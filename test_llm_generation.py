"""
test_llm_generation.py — Test the LLM generation pipeline end-to-end.

This script tests:
  1. Basic LLM invocation (single prompt → response)
  2. Itinerary planner agent (structured JSON output)
  3. Fallback mechanism (Groq → OpenRouter)
  4. Error handling & resilience

It records every test result to a JSONL log file so you can track LLM
latency, provider health, and response quality over time.

Run:
  python3 test_llm_generation.py

Log file:  test_llm_generation_log.jsonl
Report:    test_llm_generation_report.json  (latest-run summary)
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from statistics import mean, median

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agents.agents import LangTravelAgents, TravelPlanState
from config.langgraph_config import LangGraphConfig as config

# ── Paths ────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_llm_generation_log.jsonl")
REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "test_llm_generation_report.json")

# ── Colours for terminal output ──────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Global counters ──────────────────────────────────────────────
passed = 0
failed = 0

# ── Log accumulator (in-memory for this run) ─────────────────────
_current_run_entries: list[dict] = []


# ══════════════════════════════════════════════════════════════════
#  Logging helpers
# ══════════════════════════════════════════════════════════════════

def _write_log_entry(entry: dict):
    """Append one JSON line to the log file."""
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    _current_run_entries.append(entry)


def _make_entry(
    test_name: str,
    success: bool,
    elapsed_s: float,
    provider: str,
    response_length: int,
    destination: str = "",
    details: str = "",
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_name": test_name,
        "success": success,
        "elapsed_s": round(elapsed_s, 3),
        "provider": provider,
        "response_length": response_length,
        "destination": destination,
        "provider_config": {
            "primary": config.LLM_PROVIDER,
            "primary_model": config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OPENROUTER_MODEL,
            "fallback_model": config.OPENROUTER_MODEL,
        },
        "details": details,
    }


# ══════════════════════════════════════════════════════════════════
#  Historical trend analysis
# ══════════════════════════════════════════════════════════════════

def _load_history() -> list[dict]:
    """Load all previous log entries from disk."""
    if not os.path.isfile(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip corrupted lines
    return entries


def _print_history_summary():
    """Show a latency trend summary from previous runs."""
    history = _load_history()
    # Exclude entries from the current session
    current_start = _current_run_entries[0]["timestamp"] if _current_run_entries else datetime.now(timezone.utc).isoformat()
    past = [e for e in history if e.get("timestamp", "") < current_start]

    if not past:
        return

    print(f"\n  {DIM}── Historical Latency (from {len(past)} previous test runs) ──{RESET}")

    # Group by test_name
    groups: dict[str, list[float]] = {}
    for e in past:
        tn = e.get("test_name", "unknown")
        groups.setdefault(tn, []).append(e.get("elapsed_s", 0))

    for tn, times in sorted(groups.items()):
        if len(times) < 1:
            continue
        avg_t = mean(times)
        min_t = min(times)
        max_t = max(times)
        med_t = median(times)
        runs = len(times)
        # Colour based on avg latency
        if avg_t < 3:
            col = GREEN
        elif avg_t < 10:
            col = YELLOW
        else:
            col = RED
        label = tn[:55] + ".." if len(tn) > 55 else tn
        print(f"  {DIM}  {label:<57} {col}{avg_t:>5.1f}s{RESET}{DIM} avg  ({min_t:.1f}–{max_t:.1f})  med {med_t:.1f}s  n={runs}{RESET}")


def _write_report(exit_code: int):
    """Write a JSON report for this run (overwritten each run)."""
    total = passed + failed
    elapsed = 0.0
    if _current_run_entries:
        elapsed = sum(e.get("elapsed_s", 0) for e in _current_run_entries)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": config.LLM_PROVIDER,
        "model": config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OPENROUTER_MODEL,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_total": total,
        "exit_code": exit_code,
        "total_elapsed_s": round(elapsed, 3),
        "entries": _current_run_entries,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════
#  Console helpers
# ══════════════════════════════════════════════════════════════════

def ok(msg: str):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET} {msg}")


def not_ok(msg: str):
    global failed
    failed += 1
    print(f"  {RED}✗{RESET} {msg}")


def heading(title: str):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_output(label: str, content: str, max_chars: int = 800):
    """Pretty-print LLM output to the terminal."""
    print(f"  {YELLOW}── {label} ──{RESET}")
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += f"\n  {YELLOW}... (truncated, full length: {len(content)} chars){RESET}"
    print(f"  {preview}")
    print()


def timing_badge(label: str, elapsed_s: float):
    """Colour-coded timing display."""
    if elapsed_s < 2:
        col = GREEN
    elif elapsed_s < 8:
        col = YELLOW
    else:
        col = RED
    print(f"  {BOLD}⏱{RESET}  {label}: {col}{elapsed_s:.1f}s{RESET}")


def log_result(
    test_name: str,
    success: bool,
    elapsed_s: float,
    provider: str,
    response_length: int,
    destination: str = "",
    details: str = "",
):
    """Record a test result to both the JSONL log and the in-memory list."""
    entry = _make_entry(
        test_name=test_name,
        success=success,
        elapsed_s=elapsed_s,
        provider=provider,
        response_length=response_length,
        destination=destination,
        details=details,
    )
    _write_log_entry(entry)


# ══════════════════════════════════════════════════════════════════
#  1. Basic LLM Invocation Test
# ══════════════════════════════════════════════════════════════════
def test_basic_llm_invocation(agent: LangTravelAgents):
    heading("1. Basic LLM Invocation")

    print("  Sending a simple travel prompt to the LLM...\n")
    messages = [
        SystemMessage(content="You are a helpful travel assistant. Keep your response concise, under 3 sentences."),
        HumanMessage(content="What is Kyoto famous for?")
    ]

    t0 = time.time()
    try:
        response = agent._invoke_llm(messages)
        elapsed = time.time() - t0
        content = response.content if hasattr(response, "content") else str(response)

        timing_badge("Response time", elapsed)
        print_output("LLM Response", content, max_chars=600)

        # Quality checks
        checks = [
            ("Response is not empty", len(content.strip()) > 0),
            ("Response is not the generic fallback", "Unable to generate" not in content),
            ("Response mentions Kyoto (or destination)", "Kyoto" in content or "kyoto" in content.lower()),
            ("Response is at least 20 chars", len(content) >= 20),
        ]

        all_pass = all(r for _, r in checks)
        for label, result in checks:
            (ok if result else not_ok)(label)

        status = "passed" if all_pass else "some_checks_failed"
        log_result(
            test_name="1. Basic LLM Invocation",
            success=all_pass,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=len(content),
            destination="Kyoto",
            details=status,
        )
    except Exception as e:
        elapsed = time.time() - t0
        not_ok(f"LLM invocation raised an exception: {type(e).__name__}: {e}")
        log_result(
            test_name="1. Basic LLM Invocation",
            success=False,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=0,
            destination="Kyoto",
            details=f"Exception: {type(e).__name__}: {e}",
        )


# ══════════════════════════════════════════════════════════════════
#  2. Itinerary Planner Agent Test
# ══════════════════════════════════════════════════════════════════
def test_itinerary_planner_agent(agent: LangTravelAgents, destination: str, duration: int = 3):
    test_label = f"2. Itinerary — {destination} ({duration}d)"
    heading(test_label)

    state: TravelPlanState = {
        "messages": [],
        "origin": "",
        "destination": destination,
        "duration": duration,
        "budget_range": "Premier",
        "interests": ["Culture", "Gastronomy", "History"],
        "group_size": 2,
        "travel_dates": "Spring 2024",
        "current_agent": "",
        "agent_outputs": {},
        "final_plan": {},
        "iteration_count": 0,
    }

    t0 = time.time()
    try:
        print(f"  Generating itinerary for {destination}...\n")
        result_state = agent._itinerary_planner_agent(state)
        elapsed = time.time() - t0
        timing_badge("Generation time", elapsed)

        output = result_state.get("agent_outputs", {}).get("itinerary_planner", {})
        raw_response = output.get("response", "")
        parsed_output = output.get("output", {})

        # Show raw LLM response
        print_output("Raw LLM Response", raw_response, max_chars=600)

        # Show parsed JSON summary
        if parsed_output and isinstance(parsed_output, dict):
            print(f"  {YELLOW}── Parsed Itinerary Summary ──{RESET}")
            print(f"  Title:    {BOLD}{parsed_output.get('trip_title', 'N/A')}{RESET}")
            print(f"  Overview: {parsed_output.get('overview', 'N/A')[:120]}...")
            print(f"  Price:    {parsed_output.get('price_range', 'N/A')}")
            print(f"  Sustain:  {parsed_output.get('sustainability_score', 'N/A')}%")
            days = parsed_output.get("days", [])
            print(f"  Days:     {len(days)}")
            for day in days:
                acts = day.get("activities", [])
                print(f"    Day {day.get('day_number')}: {day.get('theme')} ({len(acts)} activities)")
            print()

        # Quality checks
        checks = [
            ("Response is not empty", len(raw_response.strip()) > 0),
            ("Not the generic fallback message", "Unable to generate" not in raw_response),
            ("Parsed output is a dict", isinstance(parsed_output, dict)),
        ]

        if isinstance(parsed_output, dict):
            checks.append(("Has trip_title", bool(parsed_output.get("trip_title"))))
            checks.append(("Has overview", bool(parsed_output.get("overview"))))
            checks.append(("Has days list", isinstance(parsed_output.get("days"), list)))
            days_list = parsed_output.get("days", [])
            checks.append(("Has at least 1 day", len(days_list) >= 1))
            if days_list:
                checks.append(("Day 1 has activities", len(days_list[0].get("activities", [])) > 0))
                if days_list[0].get("activities"):
                    checks.append(("Activity has title", bool(days_list[0]["activities"][0].get("title"))))
                    checks.append(("Activity has description", bool(days_list[0]["activities"][0].get("description"))))
            checks.append(("Has concierge_note", bool(parsed_output.get("concierge_note"))))

        all_pass = all(r for _, r in checks)
        for label, result in checks:
            (ok if result else not_ok)(label)

        status = "passed" if all_pass else "some_checks_failed"
        log_result(
            test_name=test_label,
            success=all_pass,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=len(raw_response),
            destination=destination,
            details=status,
        )
    except Exception as e:
        elapsed = time.time() - t0
        not_ok(f"Itinerary planner raised exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        log_result(
            test_name=test_label,
            success=False,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=0,
            destination=destination,
            details=f"Exception: {type(e).__name__}: {e}",
        )


# ══════════════════════════════════════════════════════════════════
#  3. Fallback LLM Test (Groq → OpenRouter)
# ══════════════════════════════════════════════════════════════════
def test_fallback_mechanism(agent: LangTravelAgents):
    heading("3. Fallback Mechanism Check")

    print(f"  Primary LLM provider: {YELLOW}{agent.llm_type}{RESET}")
    has_fallback = hasattr(agent, "fallback_llm") and agent.fallback_llm is not None
    print(f"  Fallback LLM available: {YELLOW}{'Yes' if has_fallback else 'No'}{RESET}")

    if has_fallback:
        ok("OpenRouter fallback LLM is initialized and ready")
    else:
        not_ok("No fallback LLM available — if Groq is at capacity, requests will fail")

    log_result(
        test_name="3a. Fallback Available",
        success=has_fallback,
        elapsed_s=0,
        provider=agent.llm_type,
        response_length=0,
        details="fallback_initialized" if has_fallback else "no_fallback",
    )

    # Test that the primary LLM actually works
    print("\n  Testing primary LLM with a quick prompt...")
    t0 = time.time()
    try:
        messages = [
            SystemMessage(content="Respond with exactly: 'OK'"),
            HumanMessage(content="Say OK")
        ]
        resp = agent._invoke_llm(messages)
        elapsed = time.time() - t0
        content = resp.content if hasattr(resp, "content") else str(resp)
        timing_badge("Response time", elapsed)
        print_output("Test Response", content, max_chars=200)
        ok(f"Primary LLM responded successfully ({elapsed:.1f}s)")

        log_result(
            test_name="3b. Primary LLM Health Check",
            success=True,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=len(content),
            details=f"Responded in {elapsed:.1f}s",
        )
    except Exception as e:
        elapsed = time.time() - t0
        not_ok(f"Primary LLM failed: {type(e).__name__}: {e}")
        log_result(
            test_name="3b. Primary LLM Health Check",
            success=False,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=0,
            details=f"Exception: {type(e).__name__}: {e}",
        )
        if has_fallback:
            print(f"  {YELLOW}  → Fallback should handle this automatically{RESET}")


# ══════════════════════════════════════════════════════════════════
#  4. Error Handling & Resilience Test
# ══════════════════════════════════════════════════════════════════
def test_error_handling(agent: LangTravelAgents):
    heading("4. Error Handling & Resilience")

    # Empty messages
    t0 = time.time()
    try:
        resp = agent._invoke_llm([])
        elapsed = time.time() - t0
        content = resp.content if hasattr(resp, "content") else str(resp)
        ok(f"Handles empty messages gracefully (got {len(content)} chars)")
        log_result(
            test_name="4a. Empty Messages",
            success=True,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=len(content),
            details="ok",
        )
    except Exception as e:
        elapsed = time.time() - t0
        not_ok(f"Empty messages caused error: {e}")
        log_result(
            test_name="4a. Empty Messages",
            success=False,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=0,
            details=f"Exception: {type(e).__name__}: {e}",
        )

    # Non-standard messages
    t0 = time.time()
    try:
        resp = agent._invoke_llm([
            HumanMessage(content="Hello"),
            AIMessage(content=""),
        ])
        elapsed = time.time() - t0
        content = resp.content if hasattr(resp, "content") else str(resp)
        ok(f"Handles empty AIMessage in history (got {len(content)} chars)")
        log_result(
            test_name="4b. Non-standard Messages",
            success=True,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=len(content),
            details="ok",
        )
    except Exception as e:
        elapsed = time.time() - t0
        not_ok(f"Empty AIMessage caused error: {e}")
        log_result(
            test_name="4b. Non-standard Messages",
            success=False,
            elapsed_s=elapsed,
            provider=agent.llm_type,
            response_length=0,
            details=f"Exception: {type(e).__name__}: {e}",
        )


# ══════════════════════════════════════════════════════════════════
#  5. Run Latency Summary
# ══════════════════════════════════════════════════════════════════
def print_latency_summary():
    """Print a summary table of response times from this run."""
    heading("5. Response Time Summary (This Run)")

    if not _current_run_entries:
        print("  No test results recorded.\n")
        return

    # Header
    print(f"  {BOLD}{'Test':<55} {'Result':<8} {'Time':>8} {'Length':>7}{RESET}")
    print(f"  {DIM}{'-' * 78}{RESET}")

    times = []
    for e in _current_run_entries:
        tn = e.get("test_name", "?")[:52]
        success = e.get("success", False)
        elapsed = e.get("elapsed_s", 0)
        rlen = e.get("response_length", 0)
        result_str = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        if elapsed >= 2 and elapsed < 8:
            time_str = f"{YELLOW}{elapsed:>6.1f}s{RESET}"
        elif elapsed >= 8:
            time_str = f"{RED}{elapsed:>6.1f}s{RESET}"
        else:
            time_str = f"{GREEN}{elapsed:>6.1f}s{RESET}"
        print(f"  {tn:<55} {result_str:<8} {time_str}  {rlen:>6}")
        if elapsed > 0:
            times.append(elapsed)

    print(f"  {DIM}{'-' * 78}{RESET}")

    if times:
        print(f"  {'':<55} {'':8} {BOLD}avg {mean(times):>5.1f}s{RESET}  {'':>6}")
        print(f"  {'':<55} {'':8} {BOLD}min {min(times):>5.1f}s{RESET}  {'':>6}")
        print(f"  {'':<55} {'':8} {BOLD}max {max(times):>5.1f}s{RESET}  {'':>6}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  XPLORA — LLM Generation Test Suite{RESET}")
    print(f"{BOLD}  Provider: {config.LLM_PROVIDER} | Model: {config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.OPENROUTER_MODEL}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    run_start = time.time()

    # 1. Validate config
    print(f"  Checking configuration...")
    if not config.validate_config():
        print(f"  {RED}✗ Configuration validation failed — check your .env file{RESET}")
        sys.exit(1)
    ok("Configuration validated")

    # 2. Create agent instance
    print(f"  Initializing LangTravelAgents...")
    try:
        agent = LangTravelAgents()
        ok("Agent initialized successfully")
    except Exception as e:
        print(f"  {RED}✗ Failed to initialize agent: {e}{RESET}")
        sys.exit(1)

    # 3. Show historical latency trends (from previous runs)
    _print_history_summary()

    # 4. Run tests
    test_basic_llm_invocation(agent)
    test_fallback_mechanism(agent)
    test_error_handling(agent)
    test_itinerary_planner_agent(agent, "Kyoto, Japan", 3)
    test_itinerary_planner_agent(agent, "Paris, France", 2)

    # 5. Print latency summary table
    print_latency_summary()

    # ── Final Summary ────────────────────────────────────────
    total = passed + failed
    suite_elapsed = time.time() - run_start
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  RESULTS: {passed}/{total} tests passed", end="")
    if failed > 0:
        print(f"  {RED}{failed} FAILED{RESET}", end="")
    print()
    print(f"{BOLD}  Suite runtime: {suite_elapsed:.1f}s{RESET}")
    print(f"{BOLD}  Log: {LOG_FILE}{RESET}")
    print(f"{BOLD}  Report: {REPORT_FILE}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    exit_code = 0 if failed == 0 else 1
    _write_report(exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

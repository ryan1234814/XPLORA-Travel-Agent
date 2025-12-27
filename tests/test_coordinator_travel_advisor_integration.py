"""Integration test for Coordinator + Travel Advisor agents.

Run:
    python test_coordinator_travel_advisor_integration.py

This test loads .env, validates GEMINI_API_KEY, then executes:
1) coordinator agent
2) travel_advisor agent
and asserts state transitions and outputs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv

# Load environment variables (must happen before importing project modules)
load_dotenv(find_dotenv())

from langchain_core.messages import HumanMessage
from agents.agents import LangTravelAgents, TravelPlanState
from config.langgraph_congfig import LangGraphConfig


def _print_messages(title: str, messages: list) -> None:
    print(f"\n{title}")
    print("-" * 80)
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content = getattr(msg, "content", msg)
        print(f"[{i}] {msg_type}:")
        print(content)
        print()
    print("-" * 80)


def test_coordinator_and_travel_advisor_together() -> bool:
    print("=" * 80)
    print("TESTING COORDINATOR + TRAVEL_ADVISOR INTEGRATION")
    print("=" * 80)

    print("\n1. Validating Configuration...")
    if not LangGraphConfig.validate_config():
        print("❌ Configuration validation failed!")
        return False
    print("✅ Configuration validated successfully")

    print("\n2. Initializing LangTravelAgents...")
    try:
        travel_agents = LangTravelAgents()
        print("✅ LangTravelAgents initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LangTravelAgents: {e}")
        return False

    print("\n3. Creating initial state...")
    state: TravelPlanState = TravelPlanState(
        messages=[HumanMessage(content="I want to plan a trip to Paris")],
        origin="",
        destination="Paris, France",
        duration=5,
        budget_range="mid-range",
        interests=["museums", "food", "architecture"],
        group_size=2,
        travel_dates="2025-06-15 to 2025-06-20",
        current_agent="",
        agent_outputs={},
        final_plan={},
        iteration_count=0,
    )

    original_message_count = len(state.get("messages", []))

    print("\n4. Running coordinator agent...")
    state_after_coord = travel_agents._coordinator_agent(state)

    _print_messages(
        "4.a Messages after coordinator:",
        state_after_coord.get("messages", []),
    )

    if state_after_coord.get("current_agent") != "coordinator":
        print(
            f"❌ Coordinator did not set current_agent correctly: {state_after_coord.get('current_agent')}"
        )
        return False

    if state_after_coord.get("iteration_count", 0) != state.get("iteration_count", 0) + 1:
        print("❌ Coordinator did not increment iteration_count")
        return False

    if len(state_after_coord.get("messages", [])) <= original_message_count:
        print("❌ Coordinator did not append a new message")
        return False

    print("✅ Coordinator step OK")

    print("\n5. Running travel_advisor agent (using coordinator-updated state)...")
    state_after_advisor = travel_agents._travel_advisor_agent(state_after_coord)

    _print_messages(
        "5.a Messages after travel_advisor:",
        state_after_advisor.get("messages", []),
    )

    if state_after_advisor.get("current_agent") != "travel_advisor":
        print(
            f"❌ Travel advisor did not set current_agent correctly: {state_after_advisor.get('current_agent')}"
        )
        return False

    agent_outputs = state_after_advisor.get("agent_outputs", {})
    if "travel_advisor" not in agent_outputs:
        print("❌ agent_outputs missing 'travel_advisor'")
        return False

    travel_advisor_output = agent_outputs.get("travel_advisor", {})
    if travel_advisor_output.get("status") != "completed":
        print(
            f"❌ travel_advisor output status unexpected: {travel_advisor_output.get('status')}"
        )
        return False

    messages_after = state_after_advisor.get("messages", [])
    if len(messages_after) <= len(state_after_coord.get("messages", [])):
        print("❌ Travel advisor did not append a new message")
        return False

    print("✅ Travel advisor step OK")

    print("\n6. Summary")
    print("- Coordinator messages:", len(state_after_coord.get("messages", [])))
    print("- After travel advisor messages:", len(messages_after))
    print("- agent_outputs keys:", list(agent_outputs.keys()))

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - COORDINATOR AND TRAVEL_ADVISOR WORK TOGETHER")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n🚀 Starting Coordinator + Travel Advisor Integration Test\n")
    ok = test_coordinator_and_travel_advisor_together()
    raise SystemExit(0 if ok else 1)

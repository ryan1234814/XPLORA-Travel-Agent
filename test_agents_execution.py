"""
Test script to verify the coordinator, travel advisor, and weather analyst agents
are working properly with the API keys from .env file
"""

# Fix for PyTorch DLL error on Windows - mock transformers to prevent import
import sys
from unittest.mock import MagicMock

# Mock transformers and torch before langchain tries to import them
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

import os
from agents.agents import LangTravelAgents, TravelPlanState
from datetime import datetime
import json

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()

def print_agent_output(agent_name, output):
    """Print formatted agent output"""
    print(f"\n{'-'*80}")
    print(f"[AGENT] {agent_name.upper()} OUTPUT")
    print(f"{'-'*80}")
    
    if isinstance(output, dict):
        print(json.dumps(output, indent=2, default=str))
    else:
        print(output)
    print(f"{'-'*80}\n")

def test_coordinator_agent():
    """Test the coordinator agent"""
    print_separator("TESTING COORDINATOR AGENT")
    
    try:
        # Initialize the travel agents system
        travel_system = LangTravelAgents()
        
        # Create initial state for testing
        initial_state = TravelPlanState(
            messages=[],
            origin="",
            destination="Paris",
            duration=5,
            budget_range="mid-range",
            interests=["museums", "food", "architecture"],
            group_size=2,
            travel_dates="2024-06-15 to 2024-06-20",
            current_agent="",
            agent_outputs={},
            final_plan={},
            iteration_count=0
        )
        
        print("[INFO] Initial State:")
        print(f"   Destination: {initial_state['destination']}")
        print(f"   Duration: {initial_state['duration']} days")
        print(f"   Budget: {initial_state['budget_range']}")
        print(f"   Interests: {', '.join(initial_state['interests'])}")
        print(f"   Group Size: {initial_state['group_size']}")
        print(f"   Travel Dates: {initial_state['travel_dates']}")
        
        # Test coordinator agent
        print("\n[RUNNING] Invoking Coordinator Agent...")
        coordinator_result = travel_system._coordinator_agent(initial_state)
        
        # Extract and display results
        print("\n[SUCCESS] Coordinator Agent Response:")
        if coordinator_result.get("messages"):
            last_message = coordinator_result["messages"][-1]
            content = getattr(last_message, "content", "No content")
            print(f"\n{content}\n")
        
        print(f"[DATA] State Updates:")
        print(f"   Current Agent: {coordinator_result.get('current_agent')}")
        print(f"   Iteration Count: {coordinator_result.get('iteration_count')}")
        print(f"   Total Messages: {len(coordinator_result.get('messages', []))}")
        
        return coordinator_result, travel_system
        
    except Exception as e:
        print(f"\n[ERROR] ERROR in Coordinator Agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def test_travel_advisor_agent(travel_system, state):
    """Test the travel advisor agent"""
    print_separator("TESTING TRAVEL ADVISOR AGENT")
    
    try:
        print("[RUNNING] Invoking Travel Advisor Agent...")
        advisor_result = travel_system._travel_advisor_agent(state)
        
        # Extract and display results
        print("\n[SUCCESS] Travel Advisor Agent Response:")
        if advisor_result.get("messages"):
            last_message = advisor_result["messages"][-1]
            content = getattr(last_message, "content", "No content")
            print(f"\n{content}\n")
        
        print(f"[DATA] Agent Output Summary:")
        agent_outputs = advisor_result.get("agent_outputs", {})
        if "travel_advisor" in agent_outputs:
            advisor_data = agent_outputs["travel_advisor"]
            print(f"   Status: {advisor_data.get('status')}")
            print(f"   Timestamp: {advisor_data.get('timestamp')}")
            
            # Display response content
            response = advisor_data.get('response')
            if response:
                response_content = getattr(response, 'content', str(response))
                print(f"\n   Response Preview:")
                print(f"   {response_content[:200]}..." if len(str(response_content)) > 200 else f"   {response_content}")
        
        return advisor_result
        
    except Exception as e:
        print(f"\n[ERROR] ERROR in Travel Advisor Agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_weather_analyst_agent(travel_system, state):
    """Test the weather analyst agent"""
    print_separator("TESTING WEATHER ANALYST AGENT")
    
    try:
        print("[RUNNING] Invoking Weather Analyst Agent...")
        weather_result = travel_system._weather_analyst_agent(state)
        
        # Extract and display results
        print("\n[SUCCESS] Weather Analyst Agent Response:")
        if weather_result.get("messages"):
            last_message = weather_result["messages"][-1]
            content = getattr(last_message, "content", "No content")
            print(f"\n{content}\n")
        
        print(f"[DATA] Agent Output Summary:")
        agent_outputs = weather_result.get("agent_outputs", {})
        if "weather_analyst" in agent_outputs:
            weather_data = agent_outputs["weather_analyst"]
            print(f"   Status: {weather_data.get('status')}")
            print(f"   Timestamp: {weather_data.get('timestamp')}")
            
            # Display response content
            response_content = weather_data.get('response')
            if response_content:
                print(f"\n   Response Preview:")
                print(f"   {response_content[:200]}..." if len(str(response_content)) > 200 else f"   {response_content}")
        
        return weather_result
        
    except Exception as e:
        print(f"\n[ERROR] ERROR in Weather Analyst Agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test execution"""
    print_separator("TRAVEL AGENT SYSTEM TEST")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Coordinator Agent
    coordinator_result, travel_system = test_coordinator_agent()
    if not coordinator_result or not travel_system:
        print("\n[ERROR] Coordinator agent test failed. Stopping tests.")
        sys.exit(1)
    
    # Test 2: Travel Advisor Agent
    advisor_result = test_travel_advisor_agent(travel_system, coordinator_result)
    if not advisor_result:
        print("\n[WARNING] Travel Advisor agent test failed.")
    
    # Test 3: Weather Analyst Agent
    weather_result = test_weather_analyst_agent(travel_system, coordinator_result)
    if not weather_result:
        print("\n[WARNING] Weather Analyst agent test failed.")
    
    # Final Summary
    print_separator("TEST SUMMARY")
    
    results = {
        "Coordinator Agent": "[PASSED]" if coordinator_result else "[FAILED]",
        "Travel Advisor Agent": "[PASSED]" if advisor_result else "[FAILED]",
        "Weather Analyst Agent": "[PASSED]" if weather_result else "[FAILED]"
    }
    
    for agent, status in results.items():
        print(f"{agent}: {status}")
    
    # Display all agent outputs collected
    if coordinator_result:
        all_outputs = {}
        if advisor_result and advisor_result.get("agent_outputs"):
            all_outputs.update(advisor_result.get("agent_outputs", {}))
        if weather_result and weather_result.get("agent_outputs"):
            all_outputs.update(weather_result.get("agent_outputs", {}))
        
        if all_outputs:
            print_separator("ALL AGENT OUTPUTS")
            for agent_name, output_data in all_outputs.items():
                print(f"\n{agent_name.upper()}:")
                print(f"  Status: {output_data.get('status')}")
                print(f"  Timestamp: {output_data.get('timestamp')}")
                
                response = output_data.get('response')
                if response:
                    content = getattr(response, 'content', str(response))
                    print(f"  Response: {content[:150]}..." if len(str(content)) > 150 else f"  Response: {content}")
    
    print_separator()
    print(f"[SUCCESS] Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()

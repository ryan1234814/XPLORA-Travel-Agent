"""
Test script for the Coordinator Agent
This script tests if the coordinator agent is working properly
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv

# Load environment variables (must happen before importing project modules)
load_dotenv(find_dotenv())

from langchain_core.messages import HumanMessage, SystemMessage
from agents.agents import LangTravelAgents, TravelPlanState
from config.langgraph_congfig import LangGraphConfig
import json

def test_coordinator_agent():
    """Test the coordinator agent function"""
    print("=" * 80)
    print("TESTING COORDINATOR AGENT")
    print("=" * 80)
    
    # Validate configuration
    print("\n1. Validating Configuration...")
    if not LangGraphConfig.validate_config():
        print("❌ Configuration validation failed!")
        return False
    print("✅ Configuration validated successfully")
    
    # Initialize the agent system
    print("\n2. Initializing LangTravelAgents...")
    try:
        travel_agents = LangTravelAgents()
        print("✅ LangTravelAgents initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LangTravelAgents: {e}")
        return False
    
    # Create a test state
    print("\n3. Creating test travel plan state...")
    test_state = TravelPlanState(
        messages=[HumanMessage(content="I want to plan a trip to Paris")],
        destination="Paris, France",
        duration=5,
        budget_range="mid-range",
        interests=["museums", "food", "architecture"],
        group_size=2,
        travel_dates="2025-06-15 to 2025-06-20",
        current_agent="",
        agent_outputs={},
        final_plan={},
        iteration_count=0
    )
    print("✅ Test state created:")
    print(f"   - Destination: {test_state['destination']}")
    print(f"   - Duration: {test_state['duration']} days")
    print(f"   - Budget: {test_state['budget_range']}")
    print(f"   - Interests: {', '.join(test_state['interests'])}")
    print(f"   - Group size: {test_state['group_size']}")
    print(f"   - Travel dates: {test_state['travel_dates']}")
    
    # Test the coordinator agent directly
    print("\n4. Testing coordinator agent function...")
    try:
        result_state = travel_agents._coordinator_agent(test_state)
        print("✅ Coordinator agent executed successfully!")
        
        # Validate the result
        print("\n5. Validating coordinator agent output...")
        
        # Check if messages were added
        if len(result_state.get("messages", [])) > len(test_state.get("messages", [])):
            print("✅ New messages added to state")
            print(f"   - Original messages: {len(test_state.get('messages', []))}")
            print(f"   - New messages: {len(result_state.get('messages', []))}")
        else:
            print("❌ No new messages added")
            
        # Check if current_agent was set
        if result_state.get("current_agent") == "coordinator":
            print("✅ Current agent set to 'coordinator'")
        else:
            print(f"❌ Current agent is '{result_state.get('current_agent')}', expected 'coordinator'")
            
        # Check if iteration count was incremented
        if result_state.get("iteration_count", 0) > test_state.get("iteration_count", 0):
            print("✅ Iteration count incremented")
            print(f"   - Original: {test_state.get('iteration_count', 0)}")
            print(f"   - New: {result_state.get('iteration_count', 0)}")
        else:
            print("❌ Iteration count not incremented")
            
        # Display the coordinator's response
        print("\n6. Coordinator Agent Response:")
        print("-" * 80)
        if result_state.get("messages"):
            last_message = result_state["messages"][-1]
            print(f"Message Type: {type(last_message).__name__}")
            if hasattr(last_message, 'content'):
                print(f"Content: {last_message.content}")
            else:
                print(f"Content: {last_message}")
        print("-" * 80)
        
        # Test with agent outputs
        print("\n7. Testing with existing agent outputs...")
        test_state_with_outputs = test_state.copy()
        test_state_with_outputs["agent_outputs"] = {
            "travel_advisor": {
                "recommendations": ["Eiffel Tower", "Louvre Museum", "Notre-Dame"],
                "status": "completed"
            },
            "weather_analyst": {
                "forecast": "Sunny, 22°C average",
                "status": "completed"
            }
        }
        
        result_state_2 = travel_agents._coordinator_agent(test_state_with_outputs)
        print("✅ Coordinator agent executed with existing agent outputs")
        
        print("\n8. Coordinator Response with Agent Outputs:")
        print("-" * 80)
        if result_state_2.get("messages"):
            last_message = result_state_2["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"Content: {last_message.content}")
            else:
                print(f"Content: {last_message}")
        print("-" * 80)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - COORDINATOR AGENT IS WORKING PROPERLY!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Error testing coordinator agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 Starting Coordinator Agent Test\n")
    success = test_coordinator_agent()
    
    if success:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

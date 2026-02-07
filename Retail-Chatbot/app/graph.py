from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from .state import AgentState
from app.agents.manager import build_agent
from app.memory.store import get_checkpointer

agent_executor = build_agent()
checkpointer = get_checkpointer()

def agent_node(state: AgentState):
    """
    The node that calls the agent. 
    Handles cases where 'output' might be missing.
    """
    try:
        # Check if we actually have messages
        if not state.get("messages"):
            return {"messages": [AIMessage(content="I didn't receive any input.")]}

        # Call the agent
        result = agent_executor.invoke({"messages": state["messages"]})
        print("AgentExecutor result:", result)  # Debug log to examine the result structure

        # Get the last AIMessage from the result
        ai_messages = [msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_msg = ai_messages[-1]
            updated_messages = state["messages"] + [last_ai_msg]
        else:
            updated_messages = state["messages"] + [AIMessage(content="Agent returned no output")]

        return {"messages": updated_messages}
    
    except Exception as e:
        print(f"CRITICAL ERROR IN AGENT NODE: {e}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

def build_graph():
    # Initialize the graph with your AgentState
    workflow = StateGraph(AgentState)

    # Add the agent node
    workflow.add_node("agent", agent_node)

    # Set the entry point
    workflow.set_entry_point("agent")

    # Define the simple path: Start -> Agent -> End
    workflow.add_edge("agent", END)

    # Compile the graph with the SQLite checkpointer for persistence
    return workflow.compile(checkpointer=checkpointer)

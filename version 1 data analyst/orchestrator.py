from langgraph.graph import StateGraph, END
from context_reader import analyze_csv
from planner_agent import generate_cleaning_plan
from executor import execute_code
import pandas as pd
from typing import Dict, Any, TypedDict , List

# Define your state schema
class AgentState(TypedDict):
    file_path: str
    current_data: pd.DataFrame
    original_data: pd.DataFrame
    schema_info: Dict[str, Any]
    cleaning_plan: List[Dict[str, Any]]
    current_step: int
    generated_code: str
    execution_result: Dict[str, Any]
    is_complete: bool

# Initialize graph with state schema
workflow = StateGraph(AgentState)

# Define nodes (keep your existing node functions exactly the same)
def read_context(state: AgentState) -> AgentState:
    try:
        file_obj = state["file_path"]
        
        # Ensure we're at the start of the file
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        # Read the file based on its type
        if hasattr(file_obj, 'read'):  # File-like object (Streamlit upload)
            df = pd.read_csv(file_obj)
        elif isinstance(file_obj, str):  # File path string
            df = pd.read_csv(file_obj)
        else:
            raise ValueError("Unsupported file input type")
            
        # Validate we got data
        if df.empty:
            raise ValueError("Uploaded file contains no data")
            
        return {
            "schema_info": analyze_csv(file_obj),
            "original_data": df,
            "current_data": df,
            **state
        }
        
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or invalid") from None
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}") from None
    

def plan(state: AgentState) -> AgentState:
    return {
        **generate_cleaning_plan({
            "current_data": state["current_data"],
            "schema_info": state["schema_info"]
        }),
        **state  # Preserve other state fields
    }

def execute(state: AgentState) -> AgentState:
    result = execute_code({
        **state,
        "current_step": state.get("current_step", 0)
    })
    
    # Mark complete if either:
    # 1. The executor says it's complete
    # 2. We've executed all steps
    is_complete = result.get("is_complete", False) or \
                 result.get("current_step", 0) >= len(result.get("cleaning_plan", []))
    
    return {
        **result,
        "is_complete": is_complete
    }

# Register nodes
workflow.add_node("read_context", read_context)
workflow.add_node("plan", plan)
workflow.add_node("execute", execute)

# Set entry point
workflow.set_entry_point("read_context")

# Define edges
workflow.add_edge("read_context", "plan")
workflow.add_edge("plan", "execute")

# Conditional edges
def should_end(state: AgentState) -> str:
    # Return "end" (not END) if complete, otherwise continue to plan
    return "end" if state.get("is_complete", False) else "plan"

workflow.add_conditional_edges(
    "execute",
    should_end,
    {
        "end": END,  # Use the END constant from LangGraph
        "continue": "plan"
    }
)

# Compile
app = workflow.compile()
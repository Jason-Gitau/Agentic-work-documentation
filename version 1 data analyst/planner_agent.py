from openai import OpenAI
# import os

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=(""),  # Set your key in environment variables
    
)
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from memory_system import memory  # Import the enhanced memory system
from cost_tracker import cost_tracker
import hashlib
import json
from cost_tracker import calculate_cost # Import the function directly


def get_data_fingerprint(df: pd.DataFrame) -> str:
    """Create a unique fingerprint for the dataset"""
    features = {
        'columns': sorted(df.columns.tolist()),
        'dtypes': {k: str(v) for k,v in df.dtypes.items()},
        'null_percentages': (df.isnull().mean() * 100).to_dict(),
        'numeric_stats': df.select_dtypes(include=np.number).describe().to_dict() if df.select_dtypes(include=np.number).shape[1] > 0 else {}
    }
    return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()

def generate_cleaning_plan(state: Dict) -> Dict:
    df = state["current_data"]
    schema = state["schema_info"]
    
    cleaning_plan = []
    
    # Handle missing values (now processes ALL columns with nulls)
    for col, null_pct in schema.get("null_percentages", {}).items():
        if null_pct > 0:  # Changed from 20% threshold to handle ALL nulls
            strategy = "drop_rows" if null_pct < 50 else "fill_with_mean"
            cleaning_plan.append({
                "action": "handle_missing_values",
                "column": col,
                "strategy": strategy,
                "estimated_cost": calculate_cost(50)
            })
    
    # New: Add data type conversion for numeric columns
    for col, dtype in schema.get("dtypes", {}).items():
        if "int" in str(dtype) or "float" in str(dtype):
            cleaning_plan.append({
                "action": "convert_dtype",
                "column": col,
                "target_type": "float64",
                "estimated_cost": calculate_cost(20)
            })
    
    return {
        **state,
        "cleaning_plan": cleaning_plan if cleaning_plan else [{
            "action": "no_op",
            "message": "No cleaning required",
            "estimated_cost": 0
        }],
        "cost_estimate": sum(a.get('estimated_cost', 0) for a in cleaning_plan),
        "from_memory": False
    }


def create_action_for_issue(issue: Dict, cost_factor: float) -> Dict:
    """Create a detailed action plan for each issue type"""
    action = {
        "type": issue["type"],
        "severity": issue["severity"],
        "priority": {"critical": 0, "high": 1, "medium": 2, "low": 3}[issue["severity"]],
        "estimated_cost": 0
    }

    if issue["type"] == "missing_values":
        action.update({
            "action": "handle_missing_values",
            "column": issue["column"],
            "strategy": suggest_missing_value_strategy(issue["percentage"], issue["column"]),
            # --- CORRECTED LINE ---
            "estimated_cost": calculate_cost(50) * cost_factor
        })
    elif issue["type"] == "duplicate_rows":
        action.update({
            "action": "remove_duplicates",
            # --- CORRECTED LINE ---
            "estimated_cost": calculate_cost(30) * cost_factor
        })

    return action

def suggest_missing_value_strategy(null_percent: float, column: str) -> str:
    """Rule-based strategy suggestion"""
    if null_percent > 80:
        return "drop_column"
    elif null_percent > 50:
        return "drop_rows" 
    elif null_percent > 20:
        return "impute_advanced"  # Will use LLM
    else:
        return "impute_simple"  # Will use simple rules

# Cost estimation helper
def estimate_llm_cost(prompt_length: int, output_length: int) -> float:
    """Estimate cost based on token counts"""
    return calculate_cost(prompt_length + output_length)

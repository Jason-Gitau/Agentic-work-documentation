import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from memory_system import memory
from cost_tracker import update_cost_tracker
import traceback
import sys
from planner_agent import get_data_fingerprint

class ExecutionError(Exception):
    """Custom exception for execution failures"""
    def __init__(self, message, error_type, details):
        super().__init__(message)
        self.error_type = error_type
        self.details = details

def execute_code(state: Dict) -> Dict:
    """
    Enhanced executor with:
    - Robust error handling and recovery
    - Detailed memory logging
    - Precise cost tracking
    - Safe execution environment
    """
    if "cleaning_plan" not in state or state["current_step"] >= len(state["cleaning_plan"]):
        return {
            **state,
            "execution_result": {
                "success": True,
                "message": "No cleaning actions required",
                "validation_passed": True
            },
            "is_complete": state["current_step"] + 1 >= len(state["cleaning_plan"]),
            "current_step": state["current_step"] + 1
        }

    action = state["cleaning_plan"][state["current_step"]]
    
    # Handle validation step specially
    if action.get("action") == "validate_data":
        return {
            **state,
            "execution_result": {
                "success": True,
                "message": action.get("message", "Data validation passed"),
                "validation_passed": True
            },
            "current_step": state["current_step"] + 1,
            "is_complete": state["current_step"] + 1 >= len(state["cleaning_plan"])
        }

    df = state["current_data"].copy()
    code = state["generated_code"]
    action = state["cleaning_plan"][state["current_step"]]
    start_time = datetime.now()
    # --- ADD THIS LINE ---
    data_fingerprint = get_data_fingerprint(df)
    cost_estimate = state.get('cost_estimate_step', 0)

    safe_globals = {
        'pd': pd,
        'np': np,
        'df': df,
        '__builtins__': {
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float
        }
    }

    try:
        df = state["current_data"].copy()
        
        if action["action"] == "handle_missing_values":
            col = action["column"]
            if action["strategy"] == "drop_rows":
                df = df.dropna(subset=[col])
            elif action["strategy"] == "fill_with_mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        
        elif action["action"] == "convert_dtype":
            col = action["column"]
            df[col] = pd.to_numeric(df[col], errors='coerce')

            
        # Execute the code
        exec(code, safe_globals)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate results
        result_df = safe_globals['df']
        validation = validate_execution(df, result_df, action)

        if not validation['is_valid']:
            raise ExecutionError(
                "Validation failed",
                "data_validation",
                validation
            )

        # --- MODIFIED LOGGING ---
        # Log successful execution
        memory.log_success(
            data_fingerprint=data_fingerprint,
            action=action,
            execution_time=execution_time,
            cost=cost_estimate,
            input_shape=df.shape,
            output_shape=result_df.shape
        )

        # Update cost tracking
        update_cost_tracker(
            operation=action['action'],
            step=state['current_step'],
            execution_time=execution_time,
            cost=cost_estimate
        )

        return {
            **state,
            "current_data": result_df,
            "execution_result": {
                "success": True,
                "execution_time": execution_time,
                "input_shape": df.shape,
                "output_shape": result_df.shape,
                "rows_affected": len(df) - len(result_df) if len(df) != len(result_df) else None,
                "columns_affected": list(set(df.columns) - set(result_df.columns)) if set(df.columns) != set(result_df.columns) else None,
                "validation": validation
            },
            "last_execution_time": execution_time
        }

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_type = get_error_type(e)
        error_details = {
            "error_type": error_type,
            "traceback": traceback.format_exc(),
            "execution_time": execution_time,
            "failed_step": state['current_step']
        }

        # --- MODIFIED LOGGING ---
        # Log failed execution
        memory.log_failure(
            data_fingerprint=data_fingerprint,
            action=action,
            error=str(e),
            execution_time=execution_time,
            cost=cost_estimate,
            input_shape=df.shape,
            code=code
        )

        # Update cost tracking for failed operation
        update_cost_tracker(
            operation=f"failed_{action['action']}",
            step=state['current_step'],
            execution_time=execution_time,
            cost=cost_estimate,
            is_success=False
        )
        
        
        return {
            **state,
            "execution_result": {
                "success": False,
                "error": str(e),
                "error_type": error_type,
                "details": error_details,
                "execution_time": execution_time
            },
            "error_message": f"{error_type}: {str(e)}",
            "last_execution_time": execution_time
        }

def validate_execution(input_df: pd.DataFrame, output_df: pd.DataFrame, action: Dict) -> Dict:
    """
    Validate execution results with rule-based checks
    """
    validation_rules = {
        "handle_missing_values": [
            ("output_has_no_new_nulls", lambda: output_df.isnull().sum().sum() <= input_df.isnull().sum().sum()),
            ("columns_preserved", lambda: set(input_df.columns).issubset(set(output_df.columns)))
        ],
        "remove_duplicates": [
            ("no_duplicates", lambda: output_df.duplicated().sum() == 0),
            ("rows_decreased_or_same", lambda: len(output_df) <= len(input_df))
        ],
        "convert_dtype": [
            ("dtype_changed", lambda: str(output_df[action['column']].dtype) == action.get('target_type', ''))
        ]
    }
    
    results = {}
    for rule_name, rule_func in validation_rules.get(action['action'], []):
        try:
            results[rule_name] = rule_func()
        except:
            results[rule_name] = False
    
    return {
        "is_valid": all(results.values()),
        "rules": results,
        "action": action['action']
    }

def get_error_type(e: Exception) -> str:
    """Classify execution errors"""
    if isinstance(e, KeyError):
        return "missing_column"
    elif isinstance(e, AttributeError):
        return "invalid_operation"
    elif isinstance(e, ValueError):
        return "value_error"
    elif isinstance(e, TypeError):
        return "type_error"
    elif isinstance(e, pd.errors.EmptyDataError):
        return "empty_data"
    elif "MemoryError" in str(e):
        return "memory_error"
    return "execution_error"

# Safe utility functions
def safe_shape(df):
    try:
        return df.shape
    except:
        return (0, 0)

def safe_columns(df):
    try:
        return list(df.columns)
    except:
        return []
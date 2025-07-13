from datetime import datetime
from typing import Dict, List, Optional , Any 
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    MISTRAL = "mistral"
    LLAMA2 = "llama-2"

# Cost per 1000 tokens (input/output)
MODEL_COSTS = {
    ModelType.GPT4: {"input": 0.03, "output": 0.06},
    ModelType.GPT35_TURBO: {"input": 0.0015, "output": 0.002},
    ModelType.MISTRAL: {"input": 0.0002, "output": 0.0002},  # Hypothetical local model cost
    ModelType.LLAMA2: {"input": 0.0001, "output": 0.0001}    # Hypothetical local model cost
}

@dataclass
class CostRecord:
    operation_id: str
    operation_type: str
    model: ModelType
    input_tokens: int
    output_tokens: int
    timestamp: datetime
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def update_cost_tracker(operation: str, step: int, execution_time: float, cost: float, is_success: bool = True):
    """Helper function to update cost tracking"""
    cost_tracker.record_operation(
        operation_type=operation,
        model=ModelType.GPT4,  # Default model, adjust as needed
        input_tokens=0,  # These can be updated if you have token counts
        output_tokens=0,
        metadata={
            "step": step,
            "execution_time": execution_time,
            "is_success": is_success
        }
    )


class CostTracker:
    def __init__(self):
        self.records: List[CostRecord] = []
        self.operation_stats: Dict[str, Dict] = {}
        self.session_costs: Dict[str, float] = {}
        self.current_session: Optional[str] = None
        
    def start_session(self, session_id: str):
        """Start a new tracking session"""
        self.current_session = session_id
        self.session_costs[session_id] = 0.0
        
    def calculate_cost(self, tokens: int, model: ModelType, is_input: bool = True) -> float:
        """Calculate cost for a given token count and model"""
        cost_per_k = MODEL_COSTS[model]["input" if is_input else "output"]
        return (tokens / 1000) * cost_per_k
        
    def record_operation(
        self,
        operation_type: str,
        model: ModelType,
        input_tokens: int,
        output_tokens: int,
        metadata: Dict[str, Any] = None
    ) -> CostRecord:
        """Record a new operation with cost calculation"""
        if metadata is None:
            metadata = {}
            
        # Generate unique operation ID
        operation_id = hashlib.md5(
            f"{operation_type}{model.value}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # Calculate costs
        input_cost = self.calculate_cost(input_tokens, model, is_input=True)
        output_cost = self.calculate_cost(output_tokens, model, is_input=False)
        total_cost = input_cost + output_cost
        
        # Create record
        record = CostRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now(),
            cost=total_cost,
            metadata=metadata
        )
        
        # Store record
        self.records.append(record)
        
        # Update session cost
        if self.current_session:
            self.session_costs[self.current_session] += total_cost
            
        # Update operation statistics
        if operation_type not in self.operation_stats:
            self.operation_stats[operation_type] = {
                "total_cost": 0.0,
                "count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0
            }
            
        self.operation_stats[operation_type]["total_cost"] += total_cost
        self.operation_stats[operation_type]["count"] += 1
        self.operation_stats[operation_type]["total_input_tokens"] += input_tokens
        self.operation_stats[operation_type]["total_output_tokens"] += output_tokens
        
        return record
        
    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session"""
        return self.session_costs.get(session_id, 0.0)
        
    def get_operation_stats(self, operation_type: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type"""
        return self.operation_stats.get(operation_type, {
            "total_cost": 0.0,
            "count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0
        })
        
    def get_total_cost(self) -> float:
        """Get total tracked cost across all sessions"""
        return sum(self.session_costs.values())
        
    def get_cost_breakdown(self) -> pd.DataFrame:
        """Get cost breakdown as a pandas DataFrame"""
        data = []
        for record in self.records:
            data.append({
                "timestamp": record.timestamp,
                "operation_id": record.operation_id,
                "operation_type": record.operation_type,
                "model": record.model.value,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost": record.cost,
                "session": self.current_session,
                **record.metadata
            })
            
        return pd.DataFrame(data)
        
    def save_to_json(self, filepath: str):
        """Save cost records to JSON file"""
        records_data = []
        for record in self.records:
            records_data.append({
                "operation_id": record.operation_id,
                "operation_type": record.operation_type,
                "model": record.model.value,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "timestamp": record.timestamp.isoformat(),
                "cost": record.cost,
                "metadata": record.metadata
            })
            
        with open(filepath, 'w') as f:
            json.dump({
                "records": records_data,
                "session_costs": self.session_costs,
                "operation_stats": self.operation_stats
            }, f, indent=2)
            
    def load_from_json(self, filepath: str):
        """Load cost records from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.records = []
        for record_data in data["records"]:
            self.records.append(CostRecord(
                operation_id=record_data["operation_id"],
                operation_type=record_data["operation_type"],
                model=ModelType(record_data["model"]),
                input_tokens=record_data["input_tokens"],
                output_tokens=record_data["output_tokens"],
                timestamp=datetime.fromisoformat(record_data["timestamp"]),
                cost=record_data["cost"],
                metadata=record_data.get("metadata", {})
            ))
            
        self.session_costs = data["session_costs"]
        self.operation_stats = data["operation_stats"]

# Global cost tracker instance
cost_tracker = CostTracker()

# Helper functions for common operations
def track_llm_call(
    operation_type: str,
    model: ModelType,
    input_tokens: int,
    output_tokens: int,
    **metadata
) -> CostRecord:
    """Convenience function for tracking LLM calls"""
    return cost_tracker.record_operation(
        operation_type=operation_type,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata
    )

def calculate_cost(tokens: int, model: ModelType = ModelType.MISTRAL) -> float:
    """
    Calculate cost for a given token count and a default model.
    This is a helper function to support cost estimation in the planning phase.
    It assumes an average of input/output costs for simplicity.
    """
    avg_cost_per_k = (MODEL_COSTS[model]["input"] + MODEL_COSTS[model]["output"]) / 2
    return (tokens / 1000) * avg_cost_per_k
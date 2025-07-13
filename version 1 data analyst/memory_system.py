import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class CleaningPattern:
    data_fingerprint: str
    actions: List[Dict[str, Any]]
    success_count: int = 0
    last_success: datetime = None
    average_time: float = 0.0
    average_cost: float = 0.0

@dataclass
class FailedExecution:
    data_fingerprint: str
    action: Dict[str, Any]
    error: str
    timestamp: datetime
    input_shape: tuple
    code: str

class PatternAwareMemory:
    def __init__(self):
        self.successful_patterns: Dict[str, CleaningPattern] = {}
        self.failed_executions: List[FailedExecution] = []
        self.data_pattern_cache = {}
        self.action_statistics = defaultdict(lambda: {
            'total_executions': 0,
            'success_count': 0,
            'total_time': 0.0,
            'total_cost': 0.0
        })

    def get_data_fingerprint(self, df: pd.DataFrame) -> str:
        """Generate a consistent fingerprint for dataframe"""
        cache_key = id(df)
        if cache_key in self.data_pattern_cache:
            return self.data_pattern_cache[cache_key]
        
        features = {
            'columns': sorted(df.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_percentages': (df.isnull().mean() * 100).round(2).to_dict(),
            'numeric_stats': self._get_numeric_stats(df),
            'categorical_stats': self._get_categorical_stats(df)
        }
        
        fingerprint = hashlib.md5(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()
        
        self.data_pattern_cache[cache_key] = fingerprint
        return fingerprint

    def _get_numeric_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for numeric columns"""
        numeric_cols = df.select_dtypes(include=np.number)
        if numeric_cols.empty:
            return {}
        
        stats = numeric_cols.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        return {
            col: {
                'mean': stats[col]['mean'],
                'std': stats[col]['std'],
                'min': stats[col]['min'],
                '25%': stats[col]['25%'],
                '50%': stats[col]['50%'],
                '75%': stats[col]['75%'],
                'max': stats[col]['max']
            }
            for col in stats
        }

    def _get_categorical_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        if categorical_cols.empty:
            return {}
        
        return {
            col: {
                'unique_count': df[col].nunique(),
                'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
                'top_frequency': (df[col] == df[col].mode()[0]).mean() if not df[col].mode().empty else 0
            }
            for col in categorical_cols
        }

    def log_success(self, data_fingerprint: str, action: Dict[str, Any], 
                   execution_time: float, cost: float, input_shape: tuple, output_shape: tuple):
        """Record a successful execution pattern"""
        # Update pattern memory
        if data_fingerprint not in self.successful_patterns:
            self.successful_patterns[data_fingerprint] = CleaningPattern(
                data_fingerprint=data_fingerprint,
                actions=[]
            )
        
        # Find or create action entry
        action_entry = None
        for existing_action in self.successful_patterns[data_fingerprint].actions:
            if existing_action['action'] == action['action'] and \
               existing_action.get('column') == action.get('column'):
                action_entry = existing_action
                break
        
        if not action_entry:
            action_entry = {
                **action,
                'execution_count': 0,
                'success_count': 0,
                'execution_times': [],
                'costs': [],
                'input_shapes': [],
                'output_shapes': []
            }
            self.successful_patterns[data_fingerprint].actions.append(action_entry)
        
        # Update statistics
        action_entry['execution_count'] += 1
        action_entry['success_count'] += 1
        action_entry['execution_times'].append(execution_time)
        action_entry['costs'].append(cost)
        action_entry['input_shapes'].append(input_shape)
        action_entry['output_shapes'].append(output_shape)
        
        # Update pattern-level stats
        pattern = self.successful_patterns[data_fingerprint]
        pattern.success_count += 1
        pattern.last_success = datetime.now()
        
        # Update global action statistics
        action_key = f"{action['action']}_{action.get('column', 'global')}"
        stats = self.action_statistics[action_key]
        stats['total_executions'] += 1
        stats['success_count'] += 1
        stats['total_time'] += execution_time
        stats['total_cost'] += cost

    def log_failure(self, data_fingerprint: str, action: Dict[str, Any], 
                   error: str, execution_time: float, cost: float, input_shape: tuple, code: str):
        """Record a failed execution"""
        self.failed_executions.append(FailedExecution(
            data_fingerprint=data_fingerprint,
            action=action,
            error=error,
            timestamp=datetime.now(),
            input_shape=input_shape,
            code=code
        ))
        
        # Update global action statistics
        action_key = f"{action['action']}_{action.get('column', 'global')}"
        stats = self.action_statistics[action_key]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['total_cost'] += cost

    def get_recommendations(self, data_fingerprint: str) -> List[Dict[str, Any]]:
        """Get recommended actions for a data pattern"""
        if data_fingerprint not in self.successful_patterns:
            return []
        
        # Return actions sorted by success rate and recency
        pattern = self.successful_patterns[data_fingerprint]
        return sorted(
            pattern.actions,
            key=lambda x: (
                x['success_count'] / x['execution_count'] if x['execution_count'] > 0 else 0,
                max(x['execution_times']) if x['execution_times'] else 0
            ),
            reverse=True
        )

    def get_similar_patterns(self, df: pd.DataFrame, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar data patterns based on schema similarity"""
        current_fingerprint = self.get_data_fingerprint(df)
        similar_patterns = []
        
        for pattern_fingerprint, pattern in self.successful_patterns.items():
            if pattern_fingerprint == current_fingerprint:
                continue
            
            similarity = self._calculate_pattern_similarity(current_fingerprint, pattern_fingerprint)
            if similarity >= similarity_threshold:
                similar_patterns.append({
                    'pattern': pattern,
                    'similarity': similarity
                })
        
        return sorted(similar_patterns, key=lambda x: x['similarity'], reverse=True)

    def _calculate_pattern_similarity(self, fingerprint1: str, fingerprint2: str) -> float:
        """Calculate similarity between two data patterns (simplified example)"""
        # In a real implementation, this would compare the actual features
        # Here we just return a placeholder value
        return 0.9 if fingerprint1[:10] == fingerprint2[:10] else 0.5

    def get_action_stats(self, action_type: str, column: str = None) -> Dict[str, Any]:
        """Get statistics for a specific action type"""
        action_key = f"{action_type}_{column if column else 'global'}"
        stats = self.action_statistics[action_key]
        
        return {
            'success_rate': stats['success_count'] / stats['total_executions'] if stats['total_executions'] > 0 else 0,
            'average_time': stats['total_time'] / stats['total_executions'] if stats['total_executions'] > 0 else 0,
            'average_cost': stats['total_cost'] / stats['total_executions'] if stats['total_executions'] > 0 else 0,
            'total_executions': stats['total_executions']
        }

# Global memory instance
memory = PatternAwareMemory()
"""
Model Performance Monitoring
Tracks accuracy, latency, and other performance metrics
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import numpy as np
from pathlib import Path


class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(
        self,
        window_size: int = 100,
        storage_path: Optional[str] = None
    ):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of recent predictions to keep in memory
            storage_path: Path to store metrics history
        """
        self.window_size = window_size
        self.storage_path = storage_path or "../mlops/data/performance_metrics.json"
        
        # In-memory storage for recent predictions
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.correct_predictions = 0
        
        # Load existing data
        self._load_history()
    
    def log_prediction(
        self,
        prediction: int,
        probability: float,
        actual: Optional[int] = None,
        latency_ms: Optional[float] = None,
        model_version: str = "v1"
    ):
        """
        Log a single prediction
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            actual: Actual label (if available)
            latency_ms: Prediction latency in milliseconds
            model_version: Model version used
        """
        timestamp = datetime.now()
        
        # Add to memory
        self.predictions.append({
            'prediction': prediction,
            'probability': probability,
            'actual': actual,
            'timestamp': timestamp.isoformat(),
            'model_version': model_version
        })
        
        if latency_ms:
            self.latencies.append(latency_ms)
        
        self.timestamps.append(timestamp)
        
        # Update counters
        self.total_predictions += 1
        if actual is not None and prediction == actual:
            self.correct_predictions += 1
        
        # Persist to disk periodically
        if self.total_predictions % 10 == 0:
            self._save_history()
    
    def get_metrics(self, period_minutes: Optional[int] = None) -> Dict:
        """
        Get performance metrics
        
        Args:
            period_minutes: Optional time period to calculate metrics for
            
        Returns:
            Dictionary of metrics
        """
        if period_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
            filtered_predictions = [
                p for p, t in zip(self.predictions, self.timestamps)
                if t > cutoff_time
            ]
        else:
            filtered_predictions = list(self.predictions)
        
        if not filtered_predictions:
            return {
                'total_predictions': 0,
                'accuracy': None,
                'avg_probability': None,
                'avg_latency_ms': None,
                'churn_rate': None
            }
        
        # Calculate metrics
        predictions_with_actual = [
            p for p in filtered_predictions if p['actual'] is not None
        ]
        
        accuracy = None
        if predictions_with_actual:
            correct = sum(
                1 for p in predictions_with_actual
                if p['prediction'] == p['actual']
            )
            accuracy = correct / len(predictions_with_actual)
        
        probabilities = [p['probability'] for p in filtered_predictions]
        predictions_list = [p['prediction'] for p in filtered_predictions]
        
        metrics = {
            'total_predictions': len(filtered_predictions),
            'accuracy': accuracy,
            'avg_probability': np.mean(probabilities) if probabilities else None,
            'avg_latency_ms': np.mean(list(self.latencies)) if self.latencies else None,
            'churn_rate': np.mean(predictions_list) if predictions_list else None,
            'predictions_per_minute': self._calculate_rate(period_minutes or 60)
        }
        
        return metrics
    
    def get_accuracy_over_time(self, intervals: int = 10) -> List[Dict]:
        """
        Get accuracy metrics over time intervals
        
        Args:
            intervals: Number of time intervals to split data
            
        Returns:
            List of metrics per interval
        """
        predictions_with_actual = [
            p for p in self.predictions if p['actual'] is not None
        ]
        
        if not predictions_with_actual:
            return []
        
        # Split into intervals
        interval_size = len(predictions_with_actual) // intervals
        if interval_size == 0:
            interval_size = 1
        
        results = []
        for i in range(0, len(predictions_with_actual), interval_size):
            chunk = predictions_with_actual[i:i+interval_size]
            if not chunk:
                continue
            
            correct = sum(1 for p in chunk if p['prediction'] == p['actual'])
            accuracy = correct / len(chunk)
            
            results.append({
                'interval': i // interval_size,
                'accuracy': accuracy,
                'size': len(chunk),
                'timestamp': chunk[0]['timestamp']
            })
        
        return results
    
    def _calculate_rate(self, period_minutes: int) -> float:
        """Calculate predictions per minute"""
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
        recent_count = sum(1 for t in self.timestamps if t > cutoff_time)
        return recent_count / period_minutes if period_minutes > 0 else 0
    
    def _save_history(self):
        """Save metrics history to disk"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'total_predictions': self.total_predictions,
                'correct_predictions': self.correct_predictions,
                'recent_predictions': list(self.predictions),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def _load_history(self):
        """Load metrics history from disk"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.total_predictions = data.get('total_predictions', 0)
                self.correct_predictions = data.get('correct_predictions', 0)
                
                # Load recent predictions
                for pred in data.get('recent_predictions', []):
                    self.predictions.append(pred)
                    if 'timestamp' in pred:
                        self.timestamps.append(
                            datetime.fromisoformat(pred['timestamp'])
                        )
        except Exception as e:
            print(f"Error loading performance history: {e}")
    
    def reset(self):
        """Reset all metrics"""
        self.predictions.clear()
        self.latencies.clear()
        self.timestamps.clear()
        self.total_predictions = 0
        self.correct_predictions = 0
        self._save_history()


# Global instance
performance_monitor = PerformanceMonitor()


if __name__ == "__main__":
    # Test performance monitor
    print("Testing Performance Monitor...")
    
    # Simulate predictions
    for i in range(50):
        performance_monitor.log_prediction(
            prediction=i % 2,
            probability=0.6 + (i % 40) / 100,
            actual=i % 2,
            latency_ms=10 + i % 20,
            model_version="v1"
        )
    
    metrics = performance_monitor.get_metrics()
    print(f"\n✓ Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    accuracy_over_time = performance_monitor.get_accuracy_over_time(5)
    print(f"\n✓ Accuracy over time: {len(accuracy_over_time)} intervals")

"""
Data Drift Detection
Detects distribution shifts in input features using statistical tests
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats
from collections import defaultdict


class DataDriftDetector:
    """Detect data drift in model inputs"""
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        threshold: float = 0.05,
        storage_path: Optional[str] = None
    ):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference/baseline data distribution
            threshold: P-value threshold for drift detection (default: 0.05)
            storage_path: Path to store drift reports
        """
        self.threshold = threshold
        self.storage_path = storage_path or "../mlops/data/drift_reports.json"
        
        # Reference data statistics
        self.reference_stats = {}
        if reference_data is not None:
            self._compute_reference_stats(reference_data)
        
        # Current window data
        self.current_window = []
        self.window_size = 100
        
        # Drift history
        self.drift_history = []
        
        self._load_history()
    
    def _compute_reference_stats(self, data: pd.DataFrame):
        """Compute statistics for reference data"""
        self.reference_stats = {}
        
        for column in data.columns:
            if data[column].dtype in ['float64', 'int64']:
                # Numerical features
                self.reference_stats[column] = {
                    'type': 'numerical',
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'values': data[column].values.tolist()
                }
            else:
                # Categorical features
                value_counts = data[column].value_counts(normalize=True)
                self.reference_stats[column] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'unique_values': data[column].unique().tolist()
                }
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set new reference data"""
        self._compute_reference_stats(data)
        print(f"✓ Reference data set: {len(data)} samples, {len(data.columns)} features")
    
    def add_sample(self, features: Dict):
        """Add a new sample to current window"""
        self.current_window.append(features)
        
        # Check if window is full
        if len(self.current_window) >= self.window_size:
            self.check_drift()
            self.current_window = []
    
    def check_drift(self) -> Dict:
        """
        Check for drift in current window
        
        Returns:
            Drift report dictionary
        """
        if not self.reference_stats:
            return {
                'error': 'No reference data set',
                'drift_detected': False
            }
        
        if len(self.current_window) < 10:
            return {
                'error': 'Not enough samples for drift detection',
                'drift_detected': False,
                'sample_count': len(self.current_window)
            }
        
        # Convert current window to DataFrame
        current_df = pd.DataFrame(self.current_window)
        
        drift_results = {}
        drifted_features = []
        
        for feature in self.reference_stats.keys():
            if feature not in current_df.columns:
                continue
            
            feature_stats = self.reference_stats[feature]
            
            if feature_stats['type'] == 'numerical':
                # Use Kolmogorov-Smirnov test for numerical features
                drift_score, p_value = self._ks_test(
                    feature_stats['values'],
                    current_df[feature].values
                )
                
                drift_detected = p_value < self.threshold
                
                drift_results[feature] = {
                    'type': 'numerical',
                    'drift_detected': drift_detected,
                    'p_value': float(p_value),
                    'drift_score': float(drift_score),
                    'reference_mean': feature_stats['mean'],
                    'current_mean': float(current_df[feature].mean()),
                    'reference_std': feature_stats['std'],
                    'current_std': float(current_df[feature].std())
                }
                
                if drift_detected:
                    drifted_features.append(feature)
            
            else:
                # Use Chi-square test for categorical features
                drift_score, p_value = self._chi_square_test(
                    feature_stats['distribution'],
                    current_df[feature]
                )
                
                drift_detected = p_value < self.threshold
                
                drift_results[feature] = {
                    'type': 'categorical',
                    'drift_detected': drift_detected,
                    'p_value': float(p_value),
                    'drift_score': float(drift_score),
                    'reference_distribution': feature_stats['distribution'],
                    'current_distribution': current_df[feature].value_counts(normalize=True).to_dict()
                }
                
                if drift_detected:
                    drifted_features.append(feature)
        
        # Create drift report
        report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'num_drifted_features': len(drifted_features),
            'total_features': len(drift_results),
            'drift_percentage': len(drifted_features) / len(drift_results) * 100 if drift_results else 0,
            'window_size': len(current_df),
            'feature_results': drift_results
        }
        
        # Save to history
        self.drift_history.append(report)
        self._save_history()
        
        return report
    
    def _ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for numerical features"""
        try:
            statistic, p_value = stats.ks_2samp(reference, current)
            return statistic, p_value
        except Exception as e:
            print(f"Error in KS test: {e}")
            return 0.0, 1.0
    
    def _chi_square_test(self, reference_dist: Dict, current_data: pd.Series) -> Tuple[float, float]:
        """Chi-square test for categorical features"""
        try:
            current_dist = current_data.value_counts(normalize=True).to_dict()
            
            # Get all unique categories
            all_categories = set(reference_dist.keys()) | set(current_dist.keys())
            
            reference_freq = [reference_dist.get(cat, 0) for cat in all_categories]
            current_freq = [current_dist.get(cat, 0) for cat in all_categories]
            
            # Chi-square test
            statistic, p_value = stats.chisquare(
                f_obs=current_freq,
                f_exp=reference_freq
            )
            
            return statistic, p_value
        except Exception as e:
            print(f"Error in Chi-square test: {e}")
            return 0.0, 1.0
    
    def get_drift_summary(self, last_n: Optional[int] = None) -> Dict:
        """
        Get summary of drift detections
        
        Args:
            last_n: Number of recent reports to summarize
            
        Returns:
            Summary dictionary
        """
        reports = self.drift_history[-last_n:] if last_n else self.drift_history
        
        if not reports:
            return {
                'total_checks': 0,
                'drift_count': 0,
                'drift_rate': 0.0
            }
        
        drift_count = sum(1 for r in reports if r['drift_detected'])
        
        # Count drifted features
        all_drifted = []
        for report in reports:
            all_drifted.extend(report['drifted_features'])
        
        from collections import Counter
        feature_drift_counts = Counter(all_drifted)
        
        return {
            'total_checks': len(reports),
            'drift_count': drift_count,
            'drift_rate': drift_count / len(reports),
            'most_drifted_features': dict(feature_drift_counts.most_common(5)),
            'last_check': reports[-1]['timestamp'] if reports else None,
            'avg_drift_percentage': np.mean([r['drift_percentage'] for r in reports])
        }
    
    def _save_history(self):
        """Save drift history to disk"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Keep only last 100 reports
            history_to_save = self.drift_history[-100:]
            
            data = {
                'reference_stats': self.reference_stats,
                'drift_history': history_to_save,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving drift history: {e}")
    
    def _load_history(self):
        """Load drift history from disk"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.reference_stats = data.get('reference_stats', {})
                self.drift_history = data.get('drift_history', [])
        except Exception as e:
            print(f"Error loading drift history: {e}")


# Global instance
drift_detector = DataDriftDetector()


if __name__ == "__main__":
    # Test drift detector
    print("Testing Data Drift Detector...")
    
    # Create reference data
    reference_data = pd.DataFrame({
        'tenure': np.random.normal(32, 25, 1000),
        'MonthlyCharges': np.random.normal(65, 30, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000)
    })
    
    drift_detector.set_reference_data(reference_data)
    
    # Simulate current data (with drift)
    for i in range(100):
        sample = {
            'tenure': float(np.random.normal(45, 25)),  # Shifted mean
            'MonthlyCharges': float(np.random.normal(65, 30)),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'])
        }
        drift_detector.add_sample(sample)
    
    # Get summary
    summary = drift_detector.get_drift_summary()
    print(f"\n✓ Drift Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

"""
Canary Deployment Manager
Handles model versioning and gradual traffic shifting
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib


class CanaryDeployment:
    """Manage canary deployment for model versions"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize canary deployment manager
        
        Args:
            config_path: Path to canary configuration file
        """
        self.config_path = config_path or "../mlops/data/canary_config.json"
        self.models = {}
        self.config = {
            'primary_version': 'v1',
            'canary_version': None,
            'traffic_split': {
                'primary': 100,
                'canary': 0
            },
            'metrics': {
                'primary': {},
                'canary': {}
            }
        }
        
        self._load_config()
    
    def register_model(
        self,
        version: str,
        model_path: str,
        description: Optional[str] = None
    ):
        """
        Register a new model version
        
        Args:
            version: Model version (e.g., 'v1', 'v2')
            model_path: Path to model file
            description: Optional description
        """
        try:
            # Load model
            model = joblib.load(model_path)
            
            self.models[version] = {
                'model': model,
                'path': model_path,
                'registered_at': datetime.now().isoformat(),
                'description': description or f"Model version {version}",
                'prediction_count': 0
            }
            
            print(f"✓ Model {version} registered from {model_path}")
            return True
        
        except Exception as e:
            print(f"❌ Error registering model {version}: {e}")
            return False
    
    def set_canary(
        self,
        canary_version: str,
        traffic_percentage: int = 10
    ):
        """
        Set canary version and traffic split
        
        Args:
            canary_version: Version to deploy as canary
            traffic_percentage: Percentage of traffic to canary (0-100)
        """
        if canary_version not in self.models:
            raise ValueError(f"Model version {canary_version} not registered")
        
        if not 0 <= traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")
        
        self.config['canary_version'] = canary_version
        self.config['traffic_split'] = {
            'primary': 100 - traffic_percentage,
            'canary': traffic_percentage
        }
        
        print(f"✓ Canary set: {canary_version} ({traffic_percentage}% traffic)")
        self._save_config()
    
    def route_request(self) -> Tuple[str, object]:
        """
        Route request to appropriate model version
        
        Returns:
            Tuple of (version, model)
        """
        if self.config['canary_version'] is None:
            # No canary, use primary
            version = self.config['primary_version']
        else:
            # Random routing based on traffic split
            rand = random.random() * 100
            
            if rand < self.config['traffic_split']['canary']:
                version = self.config['canary_version']
            else:
                version = self.config['primary_version']
        
        # Update prediction count
        if version in self.models:
            self.models[version]['prediction_count'] += 1
        
        model = self.models.get(version, {}).get('model')
        
        return version, model
    
    def promote_canary(self):
        """Promote canary to primary"""
        if self.config['canary_version'] is None:
            raise ValueError("No canary version set")
        
        old_primary = self.config['primary_version']
        new_primary = self.config['canary_version']
        
        self.config['primary_version'] = new_primary
        self.config['canary_version'] = None
        self.config['traffic_split'] = {
            'primary': 100,
            'canary': 0
        }
        
        print(f"✓ Promoted {new_primary} to primary (was {old_primary})")
        self._save_config()
    
    def rollback_canary(self):
        """Rollback canary deployment"""
        if self.config['canary_version'] is None:
            raise ValueError("No canary version to rollback")
        
        canary_version = self.config['canary_version']
        
        self.config['canary_version'] = None
        self.config['traffic_split'] = {
            'primary': 100,
            'canary': 0
        }
        
        print(f"✓ Rolled back canary {canary_version}")
        self._save_config()
    
    def gradual_rollout(
        self,
        canary_version: str,
        steps: List[int] = [10, 25, 50, 75, 100]
    ):
        """
        Gradually increase traffic to canary
        
        Args:
            canary_version: Version to rollout
            steps: List of traffic percentages
        """
        self.set_canary(canary_version, steps[0])
        
        return {
            'canary_version': canary_version,
            'rollout_steps': steps,
            'current_step': 0,
            'current_traffic': steps[0]
        }
    
    def next_rollout_step(self, steps: List[int], current_step: int):
        """Move to next rollout step"""
        if current_step + 1 >= len(steps):
            # Final step - promote to primary
            self.promote_canary()
            return {
                'completed': True,
                'promoted': True
            }
        
        next_step = current_step + 1
        next_traffic = steps[next_step]
        
        self.config['traffic_split'] = {
            'primary': 100 - next_traffic,
            'canary': next_traffic
        }
        
        self._save_config()
        
        return {
            'completed': False,
            'current_step': next_step,
            'current_traffic': next_traffic
        }
    
    def get_status(self) -> Dict:
        """Get current deployment status"""
        return {
            'primary_version': self.config['primary_version'],
            'canary_version': self.config['canary_version'],
            'traffic_split': self.config['traffic_split'],
            'registered_models': list(self.models.keys()),
            'prediction_counts': {
                v: self.models[v]['prediction_count']
                for v in self.models.keys()
            }
        }
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        """Compare metrics between two versions"""
        metrics_a = self.config['metrics'].get(version_a, {})
        metrics_b = self.config['metrics'].get(version_b, {})
        
        comparison = {}
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for metric in all_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)
            
            comparison[metric] = {
                version_a: val_a,
                version_b: val_b,
                'diff': val_b - val_a if (val_a and val_b) else None
            }
        
        return comparison
    
    def _save_config(self):
        """Save configuration to disk"""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare config without model objects
            config_to_save = {
                'primary_version': self.config['primary_version'],
                'canary_version': self.config['canary_version'],
                'traffic_split': self.config['traffic_split'],
                'metrics': self.config['metrics'],
                'models_info': {
                    v: {
                        'path': info['path'],
                        'registered_at': info['registered_at'],
                        'description': info['description'],
                        'prediction_count': info['prediction_count']
                    }
                    for v, info in self.models.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        
        except Exception as e:
            print(f"Error saving canary config: {e}")
    
    def _load_config(self):
        """Load configuration from disk"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    saved_config = json.load(f)
                
                self.config['primary_version'] = saved_config.get('primary_version', 'v1')
                self.config['canary_version'] = saved_config.get('canary_version')
                self.config['traffic_split'] = saved_config.get('traffic_split', {'primary': 100, 'canary': 0})
                self.config['metrics'] = saved_config.get('metrics', {})
                
                # Reload models
                models_info = saved_config.get('models_info', {})
                for version, info in models_info.items():
                    if Path(info['path']).exists():
                        self.register_model(version, info['path'], info.get('description'))
                        self.models[version]['prediction_count'] = info.get('prediction_count', 0)
        
        except Exception as e:
            print(f"Error loading canary config: {e}")


# Global instance
canary_deployment = CanaryDeployment()


if __name__ == "__main__":
    # Test canary deployment
    print("Testing Canary Deployment...")
    
    # Simulate model registration
    print("\n1. Registering models...")
    canary_deployment.register_model('v1', '../model/model.pkl', 'Baseline model')
    
    # Set canary
    print("\n2. Setting up canary...")
    canary_deployment.set_canary('v1', traffic_percentage=20)
    
    # Get status
    print("\n3. Current status:")
    status = canary_deployment.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

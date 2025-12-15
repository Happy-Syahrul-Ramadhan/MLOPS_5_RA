"""
MLflow Configuration and Setup
Handles MLflow server configuration, tracking, and model registry
"""

import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional
import json


class MLflowConfig:
    """MLflow configuration manager"""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5001",
        experiment_name: str = "churn-prediction-production",
        registry_uri: Optional[str] = None
    ):
        """
        Initialize MLflow configuration
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            registry_uri: Model registry URI (optional)
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri or tracking_uri
        self.experiment = None
        self.experiment_id = None
        self._initialized = False
    
    def setup_experiment(self, timeout=2):
        """Setup MLflow experiment - call this explicitly when ready"""
        if self._initialized:
            return True
            
        try:
            import socket
            # Quick check if MLflow server is accessible
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(('localhost', 5001))
            sock.close()
            
            if result != 0:
                # MLflow server not accessible
                return False
            
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)
            
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
            
            self._initialized = True
            return True
        except Exception as e:
            return False
    
    def log_model_metrics(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log model metrics to MLflow
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            metrics: Dictionary of metrics
            params: Optional parameters
            tags: Optional tags
            
        Returns:
            Run ID
        """
        if not self._initialized:
            return None
            
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log tags
                tags = tags or {}
                tags.update({
                    'model_name': model_name,
                    'model_version': model_version,
                    'deployment_timestamp': datetime.now().isoformat()
                })
                mlflow.set_tags(tags)
                
                return run.info.run_id
        except Exception:
            return None
    
    def log_prediction_metrics(
        self,
        model_version: str,
        prediction: int,
        probability: float,
        latency_ms: float,
        input_features: Dict[str, Any]
    ):
        """
        Log prediction-level metrics
        
        Args:
            model_version: Version of model used
            prediction: Prediction result
            probability: Prediction probability
            latency_ms: Prediction latency in milliseconds
            input_features: Input features used
        """
        if not self._initialized:
            return
        
        try:
            with mlflow.start_run(experiment_id=self.experiment_id):
                mlflow.log_metrics({
                    'prediction': prediction,
                    'probability': probability,
                    'latency_ms': latency_ms
                })
                
                mlflow.set_tags({
                    'model_version': model_version,
                    'prediction_timestamp': datetime.now().isoformat()
                })
                
                # Log input features as JSON artifact
                mlflow.log_dict(input_features, 'input_features.json')
        except Exception:
            pass
    
    def register_model(
        self,
        model,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float]
    ) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            model: Trained model
            model_name: Name for registered model
            model_version: Version tag
            metrics: Model metrics
            
        Returns:
            Model version number in registry
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Add version tag
            mlflow.set_tag("version", model_version)
            
            return run.info.run_id
    
    def get_model_by_version(self, model_name: str, version: str):
        """
        Load model from registry by version
        
        Args:
            model_name: Registered model name
            version: Model version
            
        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to different stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )


# Global instance
mlflow_config = MLflowConfig()


if __name__ == "__main__":
    # Test MLflow configuration
    print("Testing MLflow Configuration...")
    print(f"Tracking URI: {mlflow_config.tracking_uri}")
    print(f"Experiment ID: {mlflow_config.experiment_id}")
    
    # Test logging
    test_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88
    }
    
    run_id = mlflow_config.log_model_metrics(
        model_name="churn-model",
        model_version="v1.0",
        metrics=test_metrics
    )
    
    print(f"✓ Test run logged with ID: {run_id}")

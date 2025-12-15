"""
Logging Configuration for MLOps
Structured logging with rotation and separate channels
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json


class MLOpsLogger:
    """Centralized logging configuration"""
    
    def __init__(self, log_dir: str = "../mlops/logs"):
        """
        Initialize logging
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.prediction_logger = self._setup_logger('prediction', 'predictions.log')
        self.performance_logger = self._setup_logger('performance', 'performance.log')
        self.drift_logger = self._setup_logger('drift', 'drift.log')
        self.error_logger = self._setup_logger('error', 'errors.log', level=logging.ERROR)
        self.system_logger = self._setup_logger('system', 'system.log')
    
    def _setup_logger(
        self,
        name: str,
        filename: str,
        level: int = logging.INFO
    ) -> logging.Logger:
        """Setup individual logger with rotation"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # File handler with rotation (10MB, keep 5 backups)
        log_path = self.log_dir / filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_prediction(
        self,
        input_data: dict,
        prediction: int,
        probability: float,
        model_version: str,
        latency_ms: float
    ):
        """Log prediction event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'prediction': prediction,
            'probability': probability,
            'model_version': model_version,
            'latency_ms': latency_ms
        }
        
        self.prediction_logger.info(json.dumps(log_entry))
    
    def log_performance(
        self,
        model_version: str,
        accuracy: float,
        avg_latency: float,
        prediction_count: int,
        additional_metrics: dict = None
    ):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'prediction_count': prediction_count,
            'additional_metrics': additional_metrics or {}
        }
        
        self.performance_logger.info(json.dumps(log_entry))
    
    def log_drift(
        self,
        feature: str,
        drift_detected: bool,
        p_value: float,
        test_type: str,
        severity: str = 'INFO'
    ):
        """Log data drift detection"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'feature': feature,
            'drift_detected': drift_detected,
            'p_value': p_value,
            'test_type': test_type,
            'severity': severity
        }
        
        if drift_detected:
            self.drift_logger.warning(json.dumps(log_entry))
        else:
            self.drift_logger.info(json.dumps(log_entry))
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        traceback_info: str = None,
        context: dict = None
    ):
        """Log error event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'traceback': traceback_info,
            'context': context or {}
        }
        
        self.error_logger.error(json.dumps(log_entry))
    
    def log_system(
        self,
        event: str,
        message: str,
        level: str = 'INFO',
        metadata: dict = None
    ):
        """Log system event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'message': message,
            'metadata': metadata or {}
        }
        
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        log_level = level_map.get(level.upper(), logging.INFO)
        self.system_logger.log(log_level, json.dumps(log_entry))


# Global instance
mlops_logger = MLOpsLogger()


if __name__ == "__main__":
    # Test logging
    print("Testing MLOps Logger...")
    
    # Test prediction log
    mlops_logger.log_prediction(
        input_data={'tenure': 12, 'MonthlyCharges': 50.0},
        prediction=0,
        probability=0.85,
        model_version='v1',
        latency_ms=45.3
    )
    
    # Test performance log
    mlops_logger.log_performance(
        model_version='v1',
        accuracy=0.92,
        avg_latency=50.2,
        prediction_count=1000
    )
    
    # Test drift log
    mlops_logger.log_drift(
        feature='tenure',
        drift_detected=True,
        p_value=0.02,
        test_type='KS_test',
        severity='WARNING'
    )
    
    # Test error log
    mlops_logger.log_error(
        error_type='PredictionError',
        error_message='Invalid input data',
        context={'feature': 'tenure', 'value': 'invalid'}
    )
    
    print("✓ Logs written to mlops/logs/")

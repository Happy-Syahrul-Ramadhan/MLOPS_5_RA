"""
Training Script with MLFlow Tracking and Optuna Hyperparameter Tuning
untuk Customer Churn Prediction
"""

import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from preprocess import ChurnDataPreprocessor


class ChurnModelTrainer:
    """
    Trainer class untuk Customer Churn Model dengan MLFlow dan Optuna
    """
    
    def __init__(self, experiment_name="customer-churn-prediction"):
        """
        Initialize trainer dengan MLFlow experiment
        
        Args:
            experiment_name (str): Nama experiment di MLFlow
        """
        self.experiment_name = experiment_name
        
        # Setup MLFlow
        mlflow.set_tracking_uri("http://localhost:5001")  # MLFlow server
        mlflow.set_experiment(experiment_name)
        
        print(f"🔬 MLFlow Experiment: {experiment_name}")
        print(f"📍 Tracking URI: {mlflow.get_tracking_uri()}")
        
    def train_model(self, X_train, y_train, params):
        """
        Train Random Forest model dengan parameters tertentu
        
        Args:
            X_train: Training features
            y_train: Training target
            params (dict): Model parameters
            
        Returns:
            RandomForestClassifier: Trained model
        """
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model dan return metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        return metrics, y_pred
    
    def objective(self, trial, X_train, X_test, y_train, y_test):
        """
        Optuna objective function untuk hyperparameter tuning
        
        Args:
            trial: Optuna trial object
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            float: F1 score (metric to maximize)
        """
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        # Train model
        model = self.train_model(X_train, y_train, params)
        
        # Evaluate
        metrics, _ = self.evaluate_model(model, X_test, y_test)
        
        # Log trial to MLFlow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("trial_number", trial.number)
        
        # Return F1 score (metric to maximize)
        return metrics['f1_score']
    
    def tune_hyperparameters(self, X_train, X_test, y_train, y_test, n_trials=20):
        """
        Hyperparameter tuning menggunakan Optuna
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            n_trials (int): Number of Optuna trials
            
        Returns:
            dict: Best parameters
        """
        print("\n" + "="*60)
        print("🎯 STARTING HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*60)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name='churn-rf-optimization'
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"\n✓ Best trial: {study.best_trial.number}")
        print(f"✓ Best F1 Score: {study.best_value:.4f}")
        print(f"✓ Best parameters: {study.best_params}")
        
        return study.best_params
    
    def train_final_model(self, X_train, X_test, y_train, y_test, params, preprocessor, model_path="../model"):
        """
        Train final model dengan best parameters dan log ke MLFlow
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            params (dict): Best parameters
            preprocessor: ChurnDataPreprocessor instance
            model_path (str): Path untuk save model
            
        Returns:
            RandomForestClassifier: Final trained model
        """
        print("\n" + "="*60)
        print("🚀 TRAINING FINAL MODEL")
        print("="*60)
        
        # Create model directory if not exists
        os.makedirs(model_path, exist_ok=True)
        
        # Start MLFlow run
        with mlflow.start_run(run_name=f"final-model-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Train model
            model = self.train_model(X_train, y_train, params)
            
            # Evaluate
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("apply_smoteenn", True)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_metric("tn", int(cm[0][0]))
            mlflow.log_metric("fp", int(cm[0][1]))
            mlflow.log_metric("fn", int(cm[1][0]))
            mlflow.log_metric("tp", int(cm[1][1]))
            
            # Print classification report
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            print("\n📈 Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            # Log model to MLFlow
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="churn-random-forest"
            )
            
            # Save model locally with artifacts
            model_artifacts = {
                'model': model,
                'preprocessor': preprocessor,
                'feature_columns': preprocessor.feature_columns,
                'metrics': metrics,
                'params': params
            }
            
            model_filename = os.path.join(model_path, "churn_model.pkl")
            joblib.dump(model_artifacts, model_filename)
            
            # Log artifacts
            mlflow.log_artifact(model_filename)
            
            print(f"\n✓ Model saved to: {model_filename}")
            print(f"✓ MLFlow run ID: {mlflow.active_run().info.run_id}")
            
            return model


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*60)
    print("🎓 CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
    # 1. Preprocessing
    print("\n📦 Step 1: Data Preprocessing")
    preprocessor = ChurnDataPreprocessor()
    data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    results = preprocessor.preprocess_pipeline(
        filepath=data_path,
        apply_smoteenn=True,
        test_size=0.2,
        random_state=42
    )
    
    X_train = results['X_train']
    X_test = results['X_test']
    y_train = results['y_train']
    y_test = results['y_test']
    
    # 2. Initialize Trainer
    print("\n🔧 Step 2: Initialize Trainer")
    trainer = ChurnModelTrainer(experiment_name="customer-churn-prediction")
    
    # 3. Hyperparameter Tuning
    print("\n🎯 Step 3: Hyperparameter Tuning")
    best_params = trainer.tune_hyperparameters(
        X_train, X_test, y_train, y_test,
        n_trials=20  # Bisa ditambah untuk hasil lebih optimal
    )
    
    # 4. Train Final Model
    print("\n🚀 Step 4: Train Final Model")
    final_model = trainer.train_final_model(
        X_train, X_test, y_train, y_test,
        best_params,
        preprocessor
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Check MLFlow UI at http://localhost:5001")
    print("   2. Review experiment metrics and parameters")
    print("   3. Deploy model using Flask API")
    

if __name__ == "__main__":
    main()

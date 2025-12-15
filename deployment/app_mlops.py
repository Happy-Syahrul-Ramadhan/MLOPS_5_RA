"""
Flask API dengan MLOps Integration
Customer Churn Prediction dengan Monitoring, Drift Detection, dan Canary Deployment
"""

from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import os
from datetime import datetime
import sys
import time
import traceback as tb

# Import preprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from preprocess import ChurnDataPreprocessor

# Import MLOps modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mlops'))
from mlflow_config.config import mlflow_config
from monitoring.performance import performance_monitor
from monitoring.drift_detection import drift_detector
from versioning.canary import canary_deployment
from mlops_logging.config import mlops_logger

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
feature_columns = None

# Load model saat aplikasi start
def load_model():
    """Load model dari file"""
    global model, preprocessor, feature_columns
    
    # Coba beberapa path model
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'model', 'churn_model.pkl'),
        os.path.join('model', 'model.pkl'),
        os.path.join('..', 'model', 'model.pkl')
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        mlops_logger.log_error('ModelLoadError', 'Model file not found')
        print(f"⚠️ Model tidak ditemukan")
        return False
    
    try:
        # Load model artifacts
        artifacts = joblib.load(model_path)
        
        # Check if artifacts is a dictionary
        if isinstance(artifacts, dict):
            model = artifacts.get('model')
            feature_columns = artifacts.get('feature_columns')
            
            # Cek apakah ada preprocessor atau komponen individual
            if 'preprocessor' in artifacts:
                preprocessor = artifacts['preprocessor']
            else:
                # Buat preprocessor dari komponen individual
                preprocessor = ChurnDataPreprocessor()
                preprocessor.scaler = artifacts.get('scaler')
                preprocessor.label_encoder = artifacts.get('label_encoder')
                preprocessor.feature_columns = feature_columns
                
                # Get encoding info
                encoding_info = artifacts.get('encoding_info', {})
                preprocessor.one_hot_cols_multi = encoding_info.get('one_hot_cols_multi', [])
                preprocessor.one_hot_cols_binary = encoding_info.get('one_hot_cols_binary', [])
                preprocessor.label_encoding_cols = encoding_info.get('label_encoding_cols', ['Contract'])
                preprocessor.numeric_cols = encoding_info.get('numeric_cols', [])
                
                print(f"✓ Preprocessor dibuat dari komponen individual")
                print(f"  - Multi-class cols: {len(preprocessor.one_hot_cols_multi)}")
                print(f"  - Binary cols: {len(preprocessor.one_hot_cols_binary)}")
        else:
            # Direct model object
            model = artifacts
            preprocessor = None
            feature_columns = None
        
        # Register model with canary deployment
        canary_deployment.register_model('v1', model_path, 'Production model')
        
        # Log system event
        mlops_logger.log_system('ModelLoaded', f'Model loaded from {model_path}')
        
        print(f"✓ Model berhasil di-load dari: {model_path}")
        if preprocessor:
            print(f"✓ Preprocessor tersedia")
            print(f"✓ Feature columns: {len(feature_columns)}")
        return True
    except Exception as e:
        mlops_logger.log_error('ModelLoadError', str(e), tb.format_exc())
        print(f"❌ Error loading model: {str(e)}")
        tb.print_exc()
        return False


@app.route('/')
def home():
    """Halaman utama"""
    return render_template('index.html')


@app.route('/api/info')
def api_info():
    """Informasi API"""
    status = canary_deployment.get_status()
    metrics = performance_monitor.get_metrics()
    
    return jsonify({
        'service': 'Customer Churn Prediction API with MLOps',
        'version': '2.0',
        'status': 'ready' if model is not None else 'model not loaded',
        'deployment': {
            'primary_version': status['primary_version'],
            'canary_version': status['canary_version'],
            'traffic_split': status['traffic_split']
        },
        'performance': {
            'total_predictions': metrics['total_predictions'],
            'accuracy': metrics['accuracy'],
            'avg_latency_ms': metrics['average_latency']
        },
        'endpoints': {
            '/': 'Web Interface',
            '/api/info': 'Informasi API',
            '/api/predict': 'POST - Prediksi churn customer',
            '/api/metrics': 'GET - Performance metrics',
            '/api/drift': 'GET - Drift detection report',
            '/api/health': 'GET - Health check'
        }
    })


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    drift_summary = drift_detector.get_drift_summary()
    
    health_status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'drift_detected': drift_summary['drifted_features_count'] > 0,
        'drift_percentage': drift_summary['drift_percentage']
    }
    
    return jsonify(health_status)


@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics"""
    metrics = performance_monitor.get_metrics()
    accuracy_over_time = performance_monitor.get_accuracy_over_time()
    
    return jsonify({
        'current_metrics': metrics,
        'accuracy_over_time': accuracy_over_time
    })


@app.route('/api/drift')
def get_drift():
    """Get drift detection report"""
    drift_summary = drift_detector.get_drift_summary()
    
    return jsonify(drift_summary)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi churn dengan MLOps monitoring"""
    
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        mlops_logger.log_error('PredictionError', 'Model not loaded')
        return jsonify({'error': 'Model belum di-load'}), 503
    
    # Get JSON data
    data = request.get_json()
    
    if not data:
        mlops_logger.log_error('PredictionError', 'Empty request body')
        return jsonify({'error': 'Request body harus berupa JSON'}), 400
    
    try:
        print("\n" + "="*60)
        print("📥 INCOMING REQUEST")
        print("="*60)
        print(f"Data received: {data}")
        
        # Convert data to DataFrame
        df_input = pd.DataFrame([data])
        print(f"✓ DataFrame created: {df_input.shape}")
        
        # Preprocess data jika preprocessor tersedia
        if preprocessor:
            print("\n🔄 Preprocessing data...")
            try:
                # Transform data menggunakan preprocessor
                df_processed = preprocessor.transform_new_data(df_input)
                print(f"✓ Data preprocessed: {df_processed.shape}")
                
                # Align dengan feature columns jika ada
                if feature_columns and df_processed.shape[1] != len(feature_columns):
                    print(f"⚠️  Aligning features: {df_processed.shape[1]} -> {len(feature_columns)}")
                    # Add missing columns
                    for col in feature_columns:
                        if col not in df_processed.columns:
                            df_processed[col] = 0
                    # Keep only required columns in order
                    df_processed = df_processed[feature_columns]
                    print(f"✓ Aligned: {df_processed.shape}")
                
            except Exception as preprocess_error:
                mlops_logger.log_error('PreprocessingError', str(preprocess_error), tb.format_exc())
                print(f"❌ Preprocessing error: {str(preprocess_error)}")
                return jsonify({
                    'error': 'Preprocessing error',
                    'detail': str(preprocess_error)
                }), 500
        else:
            print("⚠️  No preprocessor - using raw input")
            df_processed = df_input
        
        # Route request to appropriate model version (canary)
        model_version, model_artifacts = canary_deployment.route_request()
        
        # Extract actual model from artifacts if it's a dict
        if model_artifacts is None:
            model_to_use = model  # Fallback to loaded model
            model_version = 'v1'
        elif isinstance(model_artifacts, dict):
            model_to_use = model_artifacts.get('model', model)
        else:
            model_to_use = model_artifacts
        
        # Make prediction
        print(f"\n🔮 Making prediction with model {model_version}...")
        prediction = model_to_use.predict(df_processed)[0]
        print(f"✓ Prediction: {prediction}")
        
        # Get probability if available
        try:
            probability = model_to_use.predict_proba(df_processed)[0].tolist()
            print(f"✓ Probability: {probability}")
        except Exception:
            probability = None
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Setup MLflow on first prediction (lazy initialization)
        if not mlflow_config._initialized:
            success = mlflow_config.setup_experiment()
            if success:
                print("✓ MLflow connected and tracking enabled")
        
        # Log prediction to performance monitor
        performance_monitor.log_prediction(
            prediction=int(prediction),
            probability=probability[1] if probability else 0.5,
            actual=None,  # Actual label akan di-update nanti
            latency_ms=latency_ms,
            model_version=model_version
        )
        
        # Add sample to drift detector
        drift_detector.add_sample(data)
        
        # Log prediction with MLOps logger
        mlops_logger.log_prediction(
            input_data=data,
            prediction=int(prediction),
            probability=probability[1] if probability else None,
            model_version=model_version,
            latency_ms=latency_ms
        )
        
        # Log metrics to MLflow (optional, non-blocking)
        if mlflow_config._initialized:
            try:
                mlflow_config.log_prediction_metrics(
                    model_version=model_version,
                    prediction=int(prediction),
                    probability=probability[1] if probability else 0.5,
                    latency_ms=latency_ms,
                    input_features=data
                )
            except Exception as e:
                pass  # Silently fail if MLflow not available
        
        # Return result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
            'model_version': model_version,
            'latency_ms': round(latency_ms, 2),
            'input_data': data
        }
        
        if probability:
            result['probability'] = {
                'no_churn': round(probability[0], 4),
                'churn': round(probability[1], 4)
            }
        
        print(f"✓ Result: {result['prediction_label']}")
        print(f"✓ Model: {model_version}")
        print(f"✓ Latency: {latency_ms:.2f}ms")
        print("="*60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        mlops_logger.log_error('PredictionError', str(e), tb.format_exc(), context={'input_data': data})
        print(f"\n❌ ERROR: {str(e)}")
        tb.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'error': 'Prediction error',
            'detail': str(e)
        }), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Endpoint untuk menerima feedback aktual (untuk monitoring akurasi)"""
    data = request.get_json()
    
    if not data or 'prediction_id' not in data or 'actual' not in data:
        return jsonify({'error': 'Memerlukan prediction_id dan actual'}), 400
    
    try:
        # Update performance monitor dengan actual value
        # Note: Implementasi ini sederhana, untuk production perlu tracking yang lebih robust
        actual = int(data['actual'])
        
        # Log feedback
        mlops_logger.log_system('FeedbackReceived', f'Actual value: {actual}', metadata=data)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received'
        })
    
    except Exception as e:
        mlops_logger.log_error('FeedbackError', str(e), tb.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 Customer Churn Prediction API with MLOps")
    print("="*60)
    
    # MLflow akan di-setup setelah Flask running
    print("⚠️  MLflow tracking disabled (akan otomatis connect ke port 5001 jika tersedia)")
    print("  Untuk enable MLflow: jalankan 'mlflow ui --port 5001' di terminal terpisah")
    
    # Load model
    if load_model():
        print("✓ API siap digunakan!")
        print("\n📊 MLOps Features:")
        print("  - Performance Monitoring")
        print("  - Data Drift Detection")
        print("  - Canary Deployment")
        print("  - MLflow Integration")
        print("  - Structured Logging")
        print("\n📍 Endpoints:")
        print("  - Web UI: http://localhost:5000")
        print("  - API Info: http://localhost:5000/api/info")
        print("  - Metrics: http://localhost:5000/api/metrics")
        print("  - Drift: http://localhost:5000/api/drift")
        print("  - Health: http://localhost:5000/api/health")
        print("="*60)
        
        mlops_logger.log_system('APIStarted', 'Flask API started with MLOps')
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Gagal load model")
        print("Pastikan file model.pkl ada di folder model/")

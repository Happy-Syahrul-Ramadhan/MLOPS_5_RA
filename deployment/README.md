# Customer Churn Prediction - Deployment

API Flask untuk memprediksi customer churn menggunakan machine learning model.

## 📁 Struktur File

```
deployment/
├── app_simple.py           # Flask application
├── preprocess.py           # Data preprocessing module
├── requirements.txt        # Python dependencies
├── test_full_api.py       # Script untuk testing API
├── Dockerfile             # Docker configuration
└── templates/
    └── index_simple.html  # Web UI
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
cd deployment
pip install -r requirements.txt
```

### 2. Jalankan Flask App

```bash
python app_simple.py
```

### 3. Akses Aplikasi

Buka browser: **http://localhost:5000**

## 📊 Fitur Input

Form input memiliki **19 field** sesuai dengan dataset:

### Demografis (4 field)
- Gender (Male/Female)
- Senior Citizen (Yes/No)
- Partner (Yes/No)
- Dependents (Yes/No)

### Layanan Telepon (2 field)
- Phone Service
- Multiple Lines

### Layanan Internet (7 field)
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Kontrak & Pembayaran (3 field)
- Contract Type
- Paperless Billing
- Payment Method

### Informasi Finansial (3 field)
- Tenure (bulan)
- Monthly Charges ($)
- Total Charges ($)

## 🔌 API Endpoints

### Web Interface
```
GET /
```
Halaman UI untuk input dan prediksi.

### API Info
```
GET /api/info
```
Informasi tentang API.

### Prediksi Churn
```
POST /api/predict
Content-Type: application/json

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1020.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": {
    "no_churn": 0.35,
    "churn": 0.65
  },
  "input_data": {...}
}
```

## 🧪 Testing

Jalankan script test:

```bash
python test_full_api.py
```

Atau test dengan cURL:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

## 📦 Model Requirements

Model harus disimpan di: `../model/model.pkl`

Format model (dictionary):
```python
{
    'model': RandomForestClassifier,
    'scaler': MinMaxScaler,
    'label_encoder': LabelEncoder,
    'feature_columns': list,
    'encoding_info': dict
}
```

## ⚙️ Teknologi

- **Flask** - Web framework
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **joblib** - Model serialization

## 🐛 Troubleshooting

### Model tidak ditemukan
Pastikan `model.pkl` ada di folder `../model/`:
```bash
ls ../model/model.pkl
```

### Port sudah digunakan
Ubah port di `app_simple.py` line terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Error preprocessing
Model memerlukan preprocessing sebelum prediksi. Pastikan semua 19 field terisi dengan benar.

## 📝 Notes

- Development server - tidak untuk production
- Untuk production, gunakan WSGI server seperti gunicorn
- Model di-train dengan scikit-learn 1.6.1

---

**Author**: MLOps Project  
**Date**: December 2025

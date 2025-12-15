# 🚀 Quick Start Guide

## Menjalankan Aplikasi

```bash
# 1. Masuk ke folder deployment
cd deployment

# 2. Install dependencies (jika belum)
pip install -r requirements.txt

# 3. Jalankan Flask app
python app.py
```

## Akses Aplikasi

**Web Interface**: http://localhost:5000

**API Endpoint**: http://localhost:5000/api/predict

## Testing

```bash
# Test dengan script Python
python test_full_api.py
```

## Struktur Folder

```
mlopssss/
├── data/                          # Dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/                         # Trained model
│   └── model.pkl
├── training/                      # Training scripts
│   ├── Curn_Customer.ipynb
│   ├── preprocess.py
│   └── train.py
└── deployment/                    # Flask application
    ├── app.py                     # Main Flask app
    ├── preprocess.py              # Preprocessing module
    ├── requirements.txt           # Dependencies
    ├── test_full_api.py          # Test script
    ├── Dockerfile                # Docker config
    ├── README.md                 # Full documentation
    └── templates/
        └── index.html            # Web UI
```

## Input Fields (19 total)

1. **Gender** - Male/Female
2. **Senior Citizen** - 0/1
3. **Partner** - Yes/No
4. **Dependents** - Yes/No
5. **Tenure** - Months (0-100)
6. **Phone Service** - Yes/No
7. **Multiple Lines** - Yes/No/No phone service
8. **Internet Service** - DSL/Fiber optic/No
9. **Online Security** - Yes/No/No internet service
10. **Online Backup** - Yes/No/No internet service
11. **Device Protection** - Yes/No/No internet service
12. **Tech Support** - Yes/No/No internet service
13. **Streaming TV** - Yes/No/No internet service
14. **Streaming Movies** - Yes/No/No internet service
15. **Contract** - Month-to-month/One year/Two year
16. **Paperless Billing** - Yes/No
17. **Payment Method** - Electronic check/Mailed check/Bank transfer/Credit card
18. **Monthly Charges** - Dollar amount
19. **Total Charges** - Dollar amount

## Output

```json
{
  "prediction": 1,                    // 0=No Churn, 1=Churn
  "prediction_label": "Churn",
  "probability": {
    "no_churn": 0.35,
    "churn": 0.65
  }
}
```

## Troubleshooting

- **Model not found**: Pastikan `model.pkl` ada di folder `model/`
- **Port sudah digunakan**: Ubah port di `app.py`
- **Import error**: Run `pip install -r requirements.txt`

---
📖 Untuk dokumentasi lengkap, lihat [README.md](README.md)

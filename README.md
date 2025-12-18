# MLOps CLI Tools for Churn Prediction Model

## 📋 Deskripsi

MLOps CLI Tools adalah aplikasi command-line interface yang komprehensif untuk mengelola Machine Learning Operations (MLOps) pada model prediksi churn. Aplikasi ini dibangun menggunakan Typer dan menyediakan fungsionalitas untuk version management, monitoring, drift detection, dan logging.

## ✨ Fitur Utama

### 1. 🎯 Model Management
- **List Models**: Menampilkan semua versi model yang terdaftar dengan status deployment
- **Model Status**: Melihat status deployment saat ini (primary dan canary)
- **Canary Deployment**: Deploy model versi baru sebagai canary dengan traffic split
- **Promote Canary**: Promosikan canary model menjadi primary model
- **Rollback**: Rollback canary deployment jika terjadi masalah

### 2. 📊 Performance Monitoring
- **Metrics**: Menampilkan metrik performa seperti akurasi, latency, dan jumlah prediksi
- **Drift Detection**: Deteksi data drift pada fitur-fitur model
- **Reset Monitoring**: Reset semua data monitoring

### 3. 📝 Logging System
- **View Logs**: Melihat log berdasarkan tipe (prediction, performance, drift, error, system)
- **Export Logs**: Export semua log ke direktori tertentu

### 4. ℹ️ System Information
- Overview lengkap status sistem MLOps

## 🏗️ Struktur Proyek

```
mlops/
├── main.py                     # CLI application
├── monitoring/
│   ├── performance.py          # Performance monitoring
│   └── drift_detection.py      # Data drift detection
├── versioning/
│   └── canary.py              # Canary deployment management
├── logging/
│   └── config.py              # Logging configuration
└── logs/                      # Log files directory
    ├── predictions.log
    ├── performance.log
    ├── drift.log
    ├── errors.log
    └── system.log
```

## 🚀 Instalasi

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install typer
```

### Setup

```bash
# Clone repository
git clone https://github.com/Happy-Syahrul-Ramadhan/MLOPS_5_RA.git
cd MLOPS_5_RA/mlops

# Install required packages
pip install -r requirements.txt
```

## 💻 Penggunaan

### 1. Model Management

#### Melihat Daftar Model
```bash
python main.py model list
```

Output:
```
📦 Registered Models:
============================================================
  • v1.0.0 [PRIMARY]
    Predictions: 1500
  • v1.1.0 [CANARY]
    Predictions: 150
```

#### Melihat Status Deployment
```bash
python main.py model status
```

Output:
```
🚀 Deployment Status:
============================================================
Primary Version:  v1.0.0 (90% traffic)
Canary Version:   v1.1.0 (10% traffic)
```

#### Deploy Model Sebagai Canary
```bash
# Deploy dengan 10% traffic (default)
python main.py model deploy v1.2.0

# Deploy dengan custom traffic percentage
python main.py model deploy v1.2.0 --traffic 20
```

#### Promote Canary ke Primary
```bash
python main.py model promote
```

#### Rollback Canary
```bash
python main.py model rollback
```

### 2. Monitoring

#### Melihat Metrik Performa
```bash
python main.py monitor metrics
```

Output:
```
📊 Performance Metrics:
============================================================
Total Predictions:    1650
Accuracy:             87.50%
Average Latency:      45.23 ms
Min Latency:          12.50 ms
Max Latency:          156.78 ms
Time Range:           2024-01-01 10:00:00 to 2024-01-15 18:30:00
```

#### Cek Data Drift
```bash
python main.py monitor drift
```

Output:
```
🔍 Data Drift Analysis:
============================================================
Total Features Tracked:   15
Features with Drift:      2
Drift Percentage:         13.3%
Current Samples:          500

Drifted Features:
  ⚠ monthly_charges
  ⚠ tenure
```

#### Reset Monitoring Data
```bash
python main.py monitor reset
```

### 3. Logging

#### Melihat Log
```bash
# Melihat 20 baris terakhir prediction log (default)
python main.py logs view prediction

# Melihat 50 baris terakhir
python main.py logs view prediction --lines 50

# Tipe log lainnya
python main.py logs view performance
python main.py logs view drift
python main.py logs view error
python main.py logs view system
```

#### Export Semua Log
```bash
# Export ke direktori default (./logs_export)
python main.py logs export

# Export ke direktori custom
python main.py logs export --output-dir /path/to/export
```

### 4. System Information

#### Melihat Overview Sistem
```bash
python main.py info
```

Output:
```
🎯 MLOps System Information
============================================================

📦 Models: 2 registered
🚀 Primary: v1.0.0
🐤 Canary: v1.1.0 (10% traffic)

📊 Performance:
   Predictions: 1650
   Accuracy: 87.50%

🔍 Drift Detection:
   Tracked Features: 15
   Drifted Features: 2
```

## 📖 Command Reference

### Model Commands

| Command | Description | Options |
|---------|-------------|---------|
| `model list` | Tampilkan semua versi model | - |
| `model status` | Status deployment saat ini | - |
| `model deploy <version>` | Deploy model sebagai canary | `--traffic`: Traffic % (default: 10) |
| `model promote` | Promote canary ke primary | - |
| `model rollback` | Rollback canary deployment | - |

### Monitor Commands

| Command | Description | Options |
|---------|-------------|---------|
| `monitor metrics` | Tampilkan metrik performa | - |
| `monitor drift` | Cek data drift | - |
| `monitor reset` | Reset data monitoring | Requires confirmation |

### Logs Commands

| Command | Description | Options |
|---------|-------------|---------|
| `logs view <type>` | Lihat log berdasarkan tipe | `--lines`: Jumlah baris (default: 20) |
| `logs export` | Export semua log | `--output-dir`: Direktori output |

### General Commands

| Command | Description |
|---------|-------------|
| `info` | Tampilkan informasi sistem |
| `--help` | Tampilkan bantuan untuk command |

## 🎨 Fitur CLI

- **Color Output**: Menggunakan warna untuk membedakan status (hijau untuk sukses, merah untuk error, kuning untuk warning)
- **Progress Indicators**: Emoji dan simbol untuk visualisasi yang lebih baik
- **Interactive Confirmation**: Konfirmasi untuk operasi yang destructive
- **Error Handling**: Error handling yang komprehensif dengan pesan yang jelas

## 🔧 Konfigurasi

### Environment Variables

Anda dapat mengkonfigurasi aplikasi melalui environment variables (jika diperlukan):

```bash
export MLOPS_LOG_LEVEL=INFO
export MLOPS_LOG_DIR=/custom/log/path
```

## 📊 Monitoring Components

### Performance Monitor
- Tracking total predictions
- Accuracy calculation
- Latency metrics (min, max, average)
- Time range tracking

### Drift Detector
- Feature-level drift detection
- Statistical analysis
- Drift percentage calculation
- Alert system untuk drifted features

### Canary Deployment
- Traffic splitting
- Version management
- Automatic rollback capability
- Prediction tracking per version

## 🐛 Troubleshooting

### Log File Tidak Ditemukan
```bash
# Pastikan direktori logs ada
mkdir -p logs

# Atau jalankan aplikasi sekali untuk auto-create
python main.py info
```

### Import Error
```bash
# Pastikan parent directory dalam Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

### Permission Error
```bash
# Berikan permission pada file log
chmod -R 755 logs/
```

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 License

Project ini menggunakan lisensi MIT - lihat file LICENSE untuk detail.

## 👥 Authors

- **Happy Syahrul Ramadhan** - 122450013
- **Nurul Alfajar Gumel**    - 122450127
- **Sahid Maulana** -122450109
- **Vita Anggraini** -122450046

## 🙏 Acknowledgments

- Typer untuk framework CLI yang luar biasa
- Community MLOps untuk best practices
- Tim pengembang yang telah berkontribusi

## 📧 Contact

Untuk pertanyaan atau saran, silakan hubungi:
- GitHub: [@Happy-Syahrul-Ramadhan](https://github.com/Happy-Syahrul-Ramadhan)
- Repository: [MLOPS_5_RA](https://github.com/Happy-Syahrul-Ramadhan/MLOPS_5_RA)

---

**Made with ❤️ for MLOps Communi

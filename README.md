# 🏦 RiskModel — Financial Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.2-yellow)
![MLflow](https://img.shields.io/badge/MLflow-2.9-blue)
![License](https://img.shields.io/badge/License-MIT-red)

A production-ready **ensemble machine learning** platform for credit and financial risk scoring. Combines **XGBoost** and **LightGBM** via soft-voting ensemble, with **SHAP explainability**, **MLflow experiment tracking**, and a **Flask REST API** with a professional web dashboard.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  Web Browser  │  │  REST Client │  │  Batch Pipeline  │  │
│   └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
└──────────┼─────────────────┼───────────────────┼────────────┘
           │                 │                   │
           ▼                 ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     FLASK API LAYER                          │
│   ┌────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│   │  /api/predict│  │/api/batch   │  │  /api/model-info  │   │
│   │  /api/health │  │/api/history │  │  /api/auth/login   │   │
│   └──────┬───────┘  └─────┬──────┘  └────────┬──────────┘   │
└──────────┼────────────────┼───────────────────┼──────────────┘
           │                │                   │
           ▼                ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     ML ENGINE LAYER                          │
│   ┌────────────────┐  ┌──────────────┐  ┌───────────────┐   │
│   │  Preprocessing │  │   Ensemble   │  │     SHAP      │   │
│   │  & Feature Eng │  │  XGB + LGBM  │  │ Explainability│   │
│   └────────────────┘  └──────────────┘  └───────────────┘   │
└──────────────────────────────────────────────────────────────┘
           │                │                   │
           ▼                ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     DATA / STORAGE LAYER                     │
│   ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│   │  SQLite   │  │  Model PKLs  │  │   MLflow Server    │    │
│   │  Database │  │  (models/)   │  │   (Experiments)    │    │
│   └──────────┘  └──────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Ensemble ML Model** — Soft-voting classifier combining XGBoost and LightGBM for robust predictions
- **Hyperparameter Tuning** — GridSearchCV-based optimization for both base models
- **SHAP Explainability** — Per-prediction feature contribution analysis with visual explanations
- **SMOTE Balancing** — Handles class imbalance in credit default datasets
- **Feature Engineering** — Automated derived feature creation (debt ratio, income-to-loan ratio, etc.)
- **MLflow Tracking** — Full experiment tracking with metrics, parameters, and artifacts
- **REST API** — Flask-based API with JWT authentication and comprehensive endpoints
- **Web Dashboard** — Professional dark-themed UI with risk gauges, charts, and prediction history
- **Batch Predictions** — Submit multiple applicants in one request
- **Docker Support** — Containerized deployment with Docker Compose

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| **ML Models** | XGBoost 2.0, LightGBM 4.2, scikit-learn 1.3 |
| **Explainability** | SHAP 0.44 |
| **Experiment Tracking** | MLflow 2.9 |
| **Backend** | Flask 3.0, Gunicorn |
| **Authentication** | PyJWT |
| **Data Processing** | Pandas, NumPy, imbalanced-learn |
| **Visualization** | Matplotlib, Seaborn, Chart.js |
| **Frontend** | Bootstrap 5.3, Jinja2 |
| **Containerization** | Docker, Docker Compose |
| **Database** | SQLite |
| **Testing** | pytest |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip
- (Optional) Docker & Docker Compose

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/riskmodel.git
cd riskmodel

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings

# Train the model
python train_model.py

# Run the application
python run.py
```

The application will be available at `http://localhost:5000`.

---

## 📡 API Documentation

### Health Check
```http
GET /api/health
```
**Response** `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Predict Risk
```http
POST /api/predict
Content-Type: application/json
Authorization: Bearer <token>
```
**Request Body**
```json
{
  "age": 35,
  "income": 65000,
  "employment_length": 8,
  "loan_amount": 15000,
  "interest_rate": 7.5,
  "credit_score": 720,
  "debt_to_income": 0.28,
  "num_credit_lines": 5,
  "num_late_payments": 0,
  "loan_purpose": "home_improvement",
  "home_ownership": "mortgage",
  "annual_revenue": 65000,
  "monthly_expenses": 3200,
  "savings_balance": 12000
}
```
**Response** `200 OK`
```json
{
  "risk_score": 0.15,
  "risk_level": "low",
  "default_probability": 0.15,
  "explanation": {
    "top_features": {
      "credit_score": -0.32,
      "debt_to_income": 0.18,
      "num_late_payments": -0.12
    }
  }
}
```

### Batch Predict
```http
POST /api/batch-predict
Content-Type: application/json
Authorization: Bearer <token>
```
**Request Body** — Array of applicant objects (same schema as single predict).

**Response** `200 OK`
```json
{
  "predictions": [
    {"risk_score": 0.15, "risk_level": "low"},
    {"risk_score": 0.72, "risk_level": "high"}
  ],
  "count": 2
}
```

### Model Info
```http
GET /api/model-info
```
**Response** `200 OK`
```json
{
  "model_type": "VotingClassifier",
  "features": ["age", "income", "..."],
  "metrics": "<real metrics loaded from reports/metrics.json>"
}
```
*Note: Real held-out test metrics are written to `reports/metrics.json` after running `train_model.py`. If the metrics file does not exist, `metrics` is `null`.*

### Authentication
```http
POST /api/auth/login
Content-Type: application/json
```
**Request Body**
```json
{
  "username": "admin",
  "password": "password"
}
```
**Response** `200 OK`
```json
{
  "token": "eyJhbGciOi...",
  "expires_in": 3600
}
```

### Prediction History
```http
GET /api/history
Authorization: Bearer <token>
```

---

## 📖 Usage Guide

### Training the Model

```bash
# Train the ensemble model with default settings
python train_model.py
```

This will:
1. Load sample data from `data/sample_credit_data.csv`
2. Preprocess features and apply SMOTE balancing
3. Train XGBoost, LightGBM, and the ensemble model
4. Evaluate all models and log metrics to MLflow
5. Generate SHAP explainability plots
6. Save models to `models/` and evaluation report to `reports/`

### Making Predictions via API

```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"age": 35, "income": 65000, "loan_amount": 15000, "credit_score": 720, ...}'
```

### Using the Web Dashboard

1. Navigate to `http://localhost:5000`
2. Log in with your credentials
3. Use the prediction form to assess individual applicants
4. View SHAP explanations for each prediction
5. Browse prediction history

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

**Services:**
- **Web App**: `http://localhost:5000`
- **MLflow UI**: `http://localhost:5001`

---

## 📁 Project Structure

```
riskmodel/
├── app.py                  # Flask application factory
├── run.py                  # Application entry point
├── train_model.py          # Model training script
├── config.py               # App configuration
├── database.py             # Database setup
├── auth.py                 # JWT authentication
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service orchestration
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # This file
│
├── api/
│   ├── __init__.py
│   └── routes.py           # API route definitions
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading & preprocessing
│   ├── feature_engineering.py  # Feature creation & selection
│   ├── model_training.py       # Model training & evaluation
│   └── explainability.py       # SHAP explanations
│
├── templates/
│   ├── base.html           # Base template
│   ├── index.html          # Dashboard
│   ├── predict.html        # Prediction form
│   ├── results.html        # Results display
│   ├── history.html        # Prediction history
│   └── login.html          # Authentication
│
├── static/
│   ├── css/style.css       # Custom styles
│   ├── js/main.js          # Frontend logic
│   └── shap/               # Generated SHAP plots
│
├── data/
│   └── sample_credit_data.csv  # Sample dataset
│
├── models/                 # Saved model artifacts
├── reports/                # Evaluation reports
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_api.py
```

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Shreyash Kashyap** | Team Member |
| **Vandana Jain** | Team Member |
| **Aryan Gupta** | Team Member |
| **Jaya Gupta** | Team Member |

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 RiskModel Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

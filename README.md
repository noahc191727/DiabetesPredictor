# Diabetes Risk Predictor

An end-to-end intern project that showcases full lifecycle skills: data preprocessing, feature engineering, model training/validation, DevOps/Docker, Streamlit UI, CI/CD, and testing.

## Overview
- Predicts Type 2 Diabetes risk using a calibrated XGBoost classifier trained on health indicators.
- Streamlit front end for interactive input, risk categories, and probability display.
- Reproducible training pipeline with preprocessing, scaling, and feature engineering to mirror inference.
- Containerized with Docker for consistent deployments; configurable for cloud or on-prem.

## What I Built
- **Data preprocessing:** cleaning, handling missing values, scaling, and train/validation splits.
- **Feature engineering:** BMI buckets, glucose/insulin categories, and other clinically relevant transforms.
- **Model training:** XGBoost with calibration (isotonic regression) and saved artifacts for inference.
- **Model evaluation:** probability-driven metrics to support risk thresholds.
- **Inference service:** `src.inference.predict` wraps preprocessing + model/scaler loading.
- **Streamlit UI:** user-friendly two-column form, risk meter, and interpretation notes.
- **DevOps:** Dockerized app with pinned Python dependencies and healthcheck; ready for registry pushes.
- **CI/CD & testing:** test scaffolding under `tests/` and requirements for local/CI environments.

## Project Structure
- `app/` — Streamlit UI (`app.py`) that imports the inference pipeline.
- `src/` — preprocessing, feature engineering, utilities, and inference code.
- `models/` — persisted model/scaler artifacts for runtime loading.
- `notebooks/` — exploratory data analysis and iterative model development.
- `tests/` — unit/integration test stubs for pipeline and inference functions.

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Run with Docker
```bash
docker build -t diabetes-predictor .
docker run -p 8501:8501 diabetes-predictor
```

## Tooling & Skills Demonstrated
- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Streamlit for rapid UI
- Docker for reproducible environments
- CI/CD-ready layout with separate dev/runtime requirements and tests
- Professional documentation and structure for handoff and maintenance

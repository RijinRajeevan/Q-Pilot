# 🚦 Q-Pilot: Quantum-Enhanced Vehicle Trajectory Prediction System

Q-Pilot is a production-quality, real-time autonomous simulation engine designed for vehicle trajectory prediction. It features a complete Explainable AI pipeline that contrasts **Classical Machine Learning Models (Linear Regression, Random Forest, GRU, LSTM)** against **Quantum Neural Networks (4-Qubit VQCs)** in real-time.

By processing the real NGSIM trajectory dataset and mapping physical scenarios (highway cruise, urban traffic, lane changes, emergency braking), Q-Pilot proves the robust capabilities of quantum models over classical counterparts when predicting nonlinear dynamics and kinematics.

---

## 🧠 Architectural Overview

Our architecture processes real physical vectors and kinematics:
1. **Data Pipeline**: Automated feature engineering and scaling over the real NGSIM dataset.
2. **Backend (FastAPI)**: Robust Python inference engine supporting `uvicorn` WebSockets, real-time trajectory simulation, and model telemetry serving.
3. **Frontend (React/Vite)**: Real-time UI built with TailwindCSS, visualizing predictions, uncertainty cones, and live metrics.
4. **Classical Models**: Fully implemented Linear Regression, Random Forest, GRU, and LSTM tracking sequences.
5. **Quantum Engine (QNN)**: 4-Qubit Variational Quantum Circuit (VQC) implemented via `qiskit` measuring phase angles for accurate non-linear predictions. 

*(Note: Prior versions used a Streamlit dashboard, which is now obsolete. The system now runs exclusively via FastAPI & Vite.)*

---

## 🛠️ Step-By-Step Setup Guide

You must establish both the Frontend (React/Vite) and Backend (PyTorch/FastAPI) environments.

### 1. Build the Python Environment
Run these commands in your project root to secure a sandbox environment and install all dependencies:
```bash
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare the Frontend
Navigate into the React UI directory and install the required node packages:
```bash
cd frontend
npm install
cd ..
```

### 3. Model Training (Optional but recommended)
You can retrain all 5 models (LR, RF, GRU, LSTM, QNN) on the dataset using the unified training script. This script automatically handles sequence building, hyperparameter tuning, and stores the artifacts in the `/models/` directory.
```bash
python train_all_models.py
```

### 4. Boot the Orchestrator!
Everything is orchestrated from a single execution script. Make sure your Python virtual environment is activated before running!
```bash
python main.py
```
> The script orchestrates both the Vite frontend (`npm run dev`) and FastAPI backend (`uvicorn`). The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`. 
> 
> You can also run them independently using `python main.py --mode frontend` or `python main.py --mode backend`.

---

## 🔬 Testing the System

A robust unit testing suite is provided in the `/tests/` directory ensuring dataset synthesis, preprocessing pipelines, classical predictors, and utility metrics (ADE, FDE, RMSE) work correctly.

Run the tests (make sure to use utf-8 encoding on Windows):
```bash
$env:PYTHONIOENCODING="utf-8"
python tests/test_system.py
```

## 📊 Explainable Mechanics 

Our pipeline acts as a pure mathematical dashboard highlighting exactly *why* Quantum Models provide stabilization in chaotic autonomous scenarios. 

- **Prediction Scenarios:** Compare models across realistic environments like High-speed Highways, Sharp Turns, and Emergency Brakes.
- **Learned Intent:** Real-time intent classification with deep `Softmax` bounds declaring intent dynamically.
- **Benchmark Diagnostics:** Compare the Average Displacement Error (`ADE`), Final Displacement Error (`FDE`), and Mean Squared Error (`MSE`) continuously matching LSTMs versus heavily entangled Quantum Subroutines dynamically in the HUD.

---
**Final Impact Statement**: Quantum Neural Networks demonstrate improved capability in capturing complex nonlinear vehicle motion patterns compared to classical machine learning models.

<div align="center">

# 🚦 Q-Pilot
### Quantum-Enhanced Vehicle Trajectory Prediction System

**A Production-Ready AI & Quantum Machine Learning Platform for Real-Time Vehicle Trajectory Prediction**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-7-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

### **Comparing Classical Machine Learning and Quantum Neural Networks for Real-Time Autonomous Vehicle Trajectory Prediction**

Built using **FastAPI • React • Vite • PyTorch • Qiskit • YOLOv8 • TypeScript**

</div>

---

# 📖 Overview

**Q-Pilot** is a production-quality research platform that demonstrates how **Quantum Machine Learning (QML)** can be applied to autonomous driving trajectory prediction.

Unlike traditional research projects that only train a model, Q-Pilot provides an **end-to-end autonomous prediction ecosystem** including:

- Real NGSIM trajectory processing
- Advanced feature engineering
- Classical Machine Learning baselines
- Deep Learning sequence models
- Quantum Neural Networks (QNN)
- Real-time inference engine
- Interactive React dashboard
- Live telemetry streaming
- Explainable AI visualizations
- Performance benchmarking

The platform enables users to compare **Linear Regression**, **Random Forest**, **GRU**, **LSTM**, and a **4-Qubit Variational Quantum Neural Network** on identical trajectory prediction tasks.

---

# 🎯 Project Goals

Q-Pilot was developed to answer one research question:

> **Can Quantum Neural Networks improve vehicle trajectory prediction over classical machine learning models for complex nonlinear driving behavior?**

The project compares multiple AI approaches using identical datasets, preprocessing pipelines, and evaluation metrics.

---

# ✨ Key Features

## 🚗 Vehicle Trajectory Prediction

- Predict future vehicle positions
- Multi-step trajectory forecasting
- Sequence-based prediction
- Real-world driving scenarios
- Highway and urban simulations

---

## 🤖 Classical AI Models

Implemented models include:

- Linear Regression
- Random Forest Regressor
- GRU Network
- LSTM Network

Each model is trained using identical datasets for fair benchmarking.

---

## ⚛️ Quantum Machine Learning

Quantum implementation includes:

- 4-Qubit Variational Quantum Circuit
- Angle Encoding
- RY Rotation Gates
- RZ Parameterized Gates
- CNOT Entanglement
- EstimatorQNN
- TorchConnector Integration
- Hybrid Quantum-Classical Optimization

---

## ⚡ Real-Time Inference

- FastAPI REST API
- WebSocket Streaming
- Live telemetry
- Real-time prediction
- Concurrent model execution
- High-performance inference

---

## 🎨 Modern Dashboard

Production-ready frontend featuring:

- React 19
- Vite
- TypeScript
- Tailwind CSS
- Recharts
- GSAP Animations
- Framer Motion
- Responsive Design

---

## 📈 Explainable AI

Interactive visualizations include:

- Trajectory Comparison
- Prediction Confidence
- Quantum Circuit Viewer
- Model Rankings
- Metrics Dashboard
- Error Analysis
- Feature Importance
- Live Performance Monitoring

---

# 🧠 System Architecture

```text
                        NGSIM Dataset
                              │
                              ▼
                   Data Preprocessing Pipeline
                              │
                              ▼
                  Feature Engineering Module
                              │
                              ▼
                 Sequence Generation (T=5)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
     Classical Models                  Quantum Model
   ┌──────────────────┐           ┌──────────────────┐
   │ Linear Regression│           │ 4-Qubit VQC      │
   │ Random Forest    │           │ Angle Encoding   │
   │ GRU              │           │ EstimatorQNN     │
   │ LSTM             │           │ TorchConnector   │
   └──────────────────┘           └──────────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    Evaluation Engine
                              │
                              ▼
                    FastAPI Backend API
                              │
                WebSockets / REST Endpoints
                              │
                              ▼
                  React + Vite Dashboard
```

---

# 🏗 Technology Stack

## Machine Learning

- PyTorch
- Scikit-Learn
- NumPy
- SciPy

---

## Quantum Computing

- Qiskit
- Qiskit Machine Learning
- EstimatorQNN
- TorchConnector

---

## Backend

- Python
- FastAPI
- Uvicorn
- WebSockets
- Joblib

---

## Frontend

- React 19
- Vite
- TypeScript
- Tailwind CSS
- Zustand
- Recharts
- GSAP
- Framer Motion

---

## Computer Vision

- OpenCV
- YOLOv8
- FilterPy

---

## Data Analytics

- Pandas
- Polars
- Matplotlib
- Plotly
- Seaborn

---

## Utilities

- TQDM
- PyPDF2

---

# 📂 Project Structure

```text
Q-Pilot
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── models/
│   ├── inference/
│   └── websocket/
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── stores/
│   └── assets/
│
├── data/
│
├── models/
│
├── notebooks/
│
├── training/
│
├── evaluation/
│
├── configs/
│
├── tests/
│
├── requirements.txt
├── package.json
├── main.py
└── README.md
```

---

# 📊 Dataset

The project primarily uses the **NGSIM Vehicle Trajectory Dataset**, one of the most widely used datasets for autonomous driving research.

Features include:

- Vehicle ID
- Frame ID
- Local X
- Local Y
- Velocity
- Acceleration
- Lane ID
- Steering Angle
- Vehicle Length
- Vehicle Width

Sequences:

```
Past Steps (T) = 5

↓

Predict

↓

Future Steps (K) = 3
```

Synthetic trajectory generation is also included for testing and demonstrations.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Q-Pilot.git

cd Q-Pilot
```

---

## Create Virtual Environment

Windows

```bash
python -m venv .venv

.\.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Frontend

```bash
cd frontend

npm install

cd ..
```

---

# 🚀 Running the Project

## Train Models

```bash
python train_all_models.py
```

This trains:

- Linear Regression
- Random Forest
- GRU
- LSTM
- Quantum Neural Network

---

## Launch Entire System

```bash
python main.py
```

---

Backend

```
http://localhost:8000
```

Frontend

```
http://localhost:5173
```

---

Run only backend

```bash
python main.py --mode backend
```

Run only frontend

```bash
python main.py --mode frontend
```

---

# 📈 Evaluation Metrics

Q-Pilot evaluates every model using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- R² Score

---

# 🌍 Driving Scenarios

The system supports multiple simulation environments.

- Highway Cruise
- Urban Driving
- Lane Change
- Heavy Traffic
- Sharp Turns
- Emergency Braking
- High-Speed Overtake

---

# 🎮 Dashboard Features

The React dashboard contains:

- Live Vehicle Simulation
- Trajectory Visualization
- Model Comparison
- Quantum Circuit Viewer
- Performance Analytics
- Confidence Cones
- Error Metrics
- Dataset Explorer
- Research Paper Analyzer

---

# 🧪 Testing

Run all tests:

Windows

```powershell
$env:PYTHONIOENCODING="utf-8"

python tests/test_system.py
```

Linux/macOS

```bash
python tests/test_system.py
```

---

# 🔬 Quantum Neural Network

The Quantum Neural Network consists of:

- 4 Physical Qubits
- Angle Encoding
- Parameterized Variational Layers
- Entanglement via CNOT Gates
- Quantum Measurements
- Hybrid Optimization using PyTorch

The hybrid architecture enables classical optimization while leveraging quantum feature representations.

---

# 📊 Research Contributions

This project demonstrates:

- Hybrid Quantum-Classical Learning
- Vehicle Trajectory Prediction
- Explainable AI
- Real-Time Autonomous Simulation
- Quantum Machine Learning Benchmarking
- Modern AI System Architecture

---

# 📸 Screenshots

> Add screenshots of:

- Landing Page
- Live Dashboard
- Trajectory Prediction
- Quantum Circuit
- Metrics Panel
- Model Comparison
- Performance Charts

---

# 🔮 Future Improvements

- Real Quantum Hardware Deployment
- Multi-Agent Prediction
- Transformer-Based Models
- Graph Neural Networks
- ROS2 Integration
- CARLA Simulator Integration
- Digital Twin Environment
- Edge AI Deployment
- Federated Learning
- Quantum Hardware Benchmarking

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push branch

```bash
git push origin feature/my-feature
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

**Rijin Rajeevan**

B.Tech Computer Science & Engineering (Artificial Intelligence)

Passionate about:

- Artificial Intelligence
- Machine Learning
- Quantum Machine Learning
- Autonomous Systems
- Computer Vision
- Deep Learning

GitHub: https://github.com/RijinRajeevan

LinkedIn: https://www.linkedin.com/in/rijin-rajeevan-246b47334

---

# 📚 References

- NGSIM Vehicle Trajectory Dataset
- PyTorch Documentation
- Qiskit Documentation
- FastAPI Documentation
- React Documentation
- YOLOv8 Documentation
- Scikit-Learn Documentation

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star!

**Q-Pilot demonstrates how Classical AI and Quantum Machine Learning can be combined to explore the future of intelligent autonomous systems.**

</div>

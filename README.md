# 🚦 Q-Pilot Live V4: Explainable Quantum Autonomous Engine

Q-Pilot V4 is a research-grade, real-time autonomous simulation engine tracking physical vehicles on the road using advanced Computer Vision, alongside an explainable AI pipeline that contrasts **Classical LSTMs** against **Quantum Neural Networks (QNNs)** in real-time.


## 🧠 Architectural Overview

Our architecture ditches synthetic data and processes real physical pixels into quantum geometries executing at **30 FPS**.
1. **Perception**: A local `YOLOv8n` + `DeepSORT` stack tracks cars.
2. **Homography Transforms**: Camera perspective is mathematically flattened to absolute 2D road space vectors (`cv2.getPerspectiveTransform`).
3. **Tracking Buffer**: Spatial behaviors form K=5 historical paths simulating instantaneous relative velocities.
4. **Bayesian PyTorch Stack**: We execute 5x parallel **Monte Carlo** forward tensor passes over `LSTM` layers preserving Dropouts to find absolute statistical **Variance**.
5. **Contextual QNN Contexts**: If targets undergo non-linear trajectory anomalies, `Qiskit` 4-qubit entangled nodes process the inputs dynamically against the LSTM paths.

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
Navigate into the React UI bucket and install the interface nodes:
```bash
cd frontend
npm install
cd ..
```

### 3. Provide a Dataset (Video Feeding)
Q-Pilot processes physical video files dynamically. 
1. Create a `data/videos/` directory in the root if it does not exist.
2. Download any standard Dashcam `.mp4` file and place it inside `data/videos/`.
3. Rename the file exactly to: `highway.mp4`.

*(If you fail to provide a video file, the system will automatically fall back to activating your local webcam!)*

### 4. Boot the Orchestrator!
Everything is orchestrated from a single execution script. Make sure your Python virtual environment is activated before running!
```bash
python main.py
```
> The architecture pulls the YOLO weights, locks the Vite interface, boots Uvicorn WebSockets under port `8000`, and opens the dashboard cleanly at `http://localhost:5174/`.

---

## 🔬 Explainable Mechanics 

Our pipeline acts as a pure mathematical dashboard highlighting exactly *why* Quantum Models provide stabilization in chaotic autonomous scenarios. 

- **Prediction Cones:** The `React` frontend draws expanding track lines based purely on the generated `Monte-Carlo Variance` extracted per frame. High-uncertainty predictions draw visibly wider, chaotic cones.
- **Learned Intent:** We discarded spatial hard-code threshold logic favoring a 3-layer `MLP Perceptron Classifier` rendering deep `Softmax` bounds declaring real-time physical intent (`Cruising`, `Braking`, `Aggressive Shift`).
- **Benchmark Diagnostics:** Compare the Average Displacement Error (`ADE`) continuously matching LSTMs versus heavily entangled Quantum Subroutines dynamically in the HUD.

## 🗑️ Code Audit Notice
If migrating from V1 builds, note that `dashboard/` (Streamlit apps) and `.streamlit/` configs are 100% obsolete. The entire tracking geometry now executes completely decoupled inside `frontend/` (TailwindCSS/Vite) & `src/inference_engine.py`.

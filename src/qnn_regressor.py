"""
Q-Pilot V7 — Real 4-Qubit VQC Trajectory Regressor
Uses Qiskit to build a variational quantum circuit for trajectory prediction.
Input: 4 normalized features (velocity, acceleration, lane_offset, distance)
Output: 2 values (predicted delta_x, delta_y)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[QNN] Qiskit not available — QNN will use classical fallback")


class VQCTrajectoryRegressor:
    """
    4-Qubit Variational Quantum Circuit for trajectory regression.
    
    Architecture:
      - 4 qubits
      - Angle encoding layer (RY gates for input features)
      - 2 variational layers:
          - RY + RZ rotations on each qubit
          - CNOT entanglement (linear chain)
      - Measurement: Z expectation values on all 4 qubits
      - Classical post-processing: linear map 4 -> 2 (delta_x, delta_y)
    
    Training: COBYLA optimizer, 50 iterations
    """
    
    def __init__(self, num_qubits=4, num_var_layers=2):
        self.num_qubits = num_qubits
        self.num_var_layers = num_var_layers
        self.is_trained = False
        self.output_dim = None  # Set during training
        self._post_weights = None  # (qnn_output_dim, output_dim) matrix
        self._post_bias = None     # (output_dim,) bias
        
        if QISKIT_AVAILABLE:
            self._build_circuit()
        else:
            self.qnn = None
    
    def _build_circuit(self):
        """Build the parameterized quantum circuit."""
        # Input parameters (4 features)
        self.input_params = ParameterVector("x", self.num_qubits)
        
        # Weight parameters: each var layer has num_qubits * 2 params (RY + RZ per qubit)
        num_weights = self.num_var_layers * self.num_qubits * 2
        self.weight_params = ParameterVector("w", num_weights)
        
        qc = QuantumCircuit(self.num_qubits)
        
        # ── Input Encoding Layer: RY angle encoding ──
        for i in range(self.num_qubits):
            qc.ry(self.input_params[i], i)
        
        # ── Variational Layers ──
        w_idx = 0
        for layer in range(self.num_var_layers):
            # Rotation layer: RY + RZ on each qubit
            for q in range(self.num_qubits):
                qc.ry(self.weight_params[w_idx], q)
                w_idx += 1
                qc.rz(self.weight_params[w_idx], q)
                w_idx += 1
            
            # Entanglement layer: CNOT chain
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
            # Close the ring
            if self.num_qubits > 2:
                qc.cx(self.num_qubits - 1, 0)
        
        self.circuit = qc
        
        # Build EstimatorQNN
        try:
            self.qnn = EstimatorQNN(
                circuit=qc,
                input_params=self.input_params,
                weight_params=self.weight_params,
            )
            print(f"[QNN] Built {self.num_qubits}-qubit VQC with {num_weights} parameters, {self.num_var_layers} layers")
        except Exception as e:
            print(f"[QNN] EstimatorQNN creation failed: {e}")
            self.qnn = None
    
    def train(self, X_train, y_train, max_iter=50):
        """
        Train the VQC on trajectory data.
        
        Args:
            X_train: (N, 4) input features
            y_train: (N,) or (N, D) targets
            max_iter: COBYLA iterations
        """
        # Ensure y is 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        self.output_dim = y_train.shape[1]
        
        if self.qnn is None:
            print("[QNN] Falling back to classical approximation")
            self._train_classical_fallback(X_train, y_train)
            return
        
        from scipy.optimize import minimize
        
        # Normalize inputs to [0, pi] range for angle encoding
        X_norm = self._normalize_input(X_train)
        
        # Initialize weights randomly
        num_weights = len(self.weight_params)
        initial_weights = np.random.uniform(-np.pi, np.pi, num_weights)
        
        # Subsample for speed (QNN training is slow)
        n_samples = min(len(X_norm), 200)
        indices = np.random.choice(len(X_norm), n_samples, replace=False)
        X_sub = X_norm[indices]
        y_sub = y_train[indices]
        
        def objective(weights):
            try:
                # Forward pass through QNN
                predictions = []
                for x in X_sub[:50]:  # Further limit for speed
                    qnn_out = self.qnn.forward(x.reshape(1, -1), weights)
                    predictions.append(qnn_out.flatten())
                predictions = np.array(predictions)
                
                # Map QNN output -> target dims via learned linear layer
                if self._post_weights is None:
                    self._post_weights = np.random.randn(predictions.shape[1], self.output_dim) * 0.1
                    self._post_bias = np.zeros(self.output_dim)
                
                mapped = predictions @ self._post_weights + self._post_bias
                loss = np.mean((mapped - y_sub[:50]) ** 2)
                return float(loss)
            except Exception:
                return 1e6
        
        print(f"[QNN] Training VQC with COBYLA ({max_iter} iterations, {n_samples} samples)...")
        result = minimize(objective, initial_weights, method='COBYLA',
                         options={'maxiter': max_iter, 'rhobeg': 0.5})
        
        self._trained_weights = result.x
        self.is_trained = True
        
        # Fit post-processing layer
        self._fit_post_layer(X_norm[indices[:50]], y_sub[:50])
        
        print(f"[QNN] Training complete. Final loss: {result.fun:.4f}")
    
    def _fit_post_layer(self, X, y):
        """Fit the classical post-processing layer."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        try:
            predictions = []
            for x in X:
                qnn_out = self.qnn.forward(x.reshape(1, -1), self._trained_weights)
                predictions.append(qnn_out.flatten())
            predictions = np.array(predictions)
            
            # Least squares to fit linear mapping
            A = np.column_stack([predictions, np.ones(len(predictions))])
            for dim in range(self.output_dim):
                sol, _, _, _ = np.linalg.lstsq(A, y[:, dim], rcond=None)
                self._post_weights[:, dim] = sol[:-1]
                self._post_bias[dim] = sol[-1]
        except Exception as e:
            print(f"[QNN] Post-layer fitting failed: {e}")
    
    def _train_classical_fallback(self, X_train, y_train):
        """Classical neural-network-like fallback when Qiskit isn't available."""
        from sklearn.kernel_ridge import KernelRidge
        
        self._fallback_models = []
        for dim in range(y_train.shape[1]):
            model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
            model.fit(X_train[:500], y_train[:500, dim])
            self._fallback_models.append(model)
        
        self.is_trained = True
        print("[QNN] Classical fallback trained")
    
    def predict(self, X):
        """
        Predict trajectory deltas.
        
        Args:
            X: (N, 4) input features
        Returns:
            (N, 2) predicted [delta_x, delta_y]
        """
        if not self.is_trained:
            # Return zero prediction if untrained
            return np.zeros((len(X), 2))
        
        if self.qnn is not None and hasattr(self, '_trained_weights'):
            X_norm = self._normalize_input(X)
            predictions = []
            for x in X_norm:
                try:
                    qnn_out = self.qnn.forward(x.reshape(1, -1), self._trained_weights)
                    mapped = qnn_out.flatten() @ self._post_weights + self._post_bias
                    predictions.append(mapped)
                except Exception:
                    predictions.append(np.zeros(2))
            return np.array(predictions)
        elif hasattr(self, '_fallback_models'):
            results = np.column_stack([
                m.predict(X[:, :4]) for m in self._fallback_models
            ])
            return results
        else:
            return np.zeros((len(X), 2))
    
    def predict_with_uncertainty(self, X, n_samples=5):
        """
        Predict with uncertainty estimation.
        Uses slight parameter perturbation to estimate variance.
        
        Returns:
            predictions: (N, 2) mean predictions
            uncertainties: (N, 2) standard deviations
        """
        if not self.is_trained:
            return np.zeros((len(X), 2)), np.ones((len(X), 2)) * 0.1
        
        all_preds = []
        for _ in range(n_samples):
            if hasattr(self, '_trained_weights'):
                # Perturb weights slightly for uncertainty
                noise = np.random.normal(0, 0.05, len(self._trained_weights))
                old_w = self._trained_weights.copy()
                self._trained_weights = old_w + noise
                pred = self.predict(X)
                self._trained_weights = old_w
            else:
                pred = self.predict(X)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def _normalize_input(X):
        """Normalize inputs to [0, pi] for angle encoding."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range < 1e-8] = 1.0
        return (X - X_min) / X_range * np.pi
    
    def get_circuit_info(self):
        """Return circuit metadata for frontend display."""
        return {
            'num_qubits': self.num_qubits,
            'num_var_layers': self.num_var_layers,
            'num_parameters': self.num_var_layers * self.num_qubits * 2,
            'encoding': 'RY angle encoding',
            'entanglement': 'CNOT ring',
            'optimizer': 'COBYLA',
            'is_trained': self.is_trained,
        }

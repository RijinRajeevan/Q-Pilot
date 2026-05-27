"""
Q-Pilot V7 — Motion Prediction Models
Defines all trajectory prediction models: LR, RF, GRU, LSTM, QNN.
Each model takes a sequence of kinematic features and predicts future positions.
"""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.qnn_regressor import VQCTrajectoryRegressor


# ═══════════════════════════════════════════════════════════════
# GRU and LSTM models (PyTorch)
# ═══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class GRUPredictor(nn.Module):
        """2-layer GRU for trajectory prediction."""
        def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_steps=5, output_dim=2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_steps * output_dim),
            )
            self.output_steps = output_steps
            self.output_dim = output_dim

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            _, h = self.gru(x)
            out = self.fc(h[-1])  # Use last hidden state
            return out.view(-1, self.output_steps, self.output_dim)

    class LSTMPredictor(nn.Module):
        """2-layer LSTM for trajectory prediction."""
        def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_steps=5, output_dim=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_steps * output_dim),
            )
            self.output_steps = output_steps
            self.output_dim = output_dim

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            out = self.fc(h[-1])
            return out.view(-1, self.output_steps, self.output_dim)


# ═══════════════════════════════════════════════════════════════
# Unified Model Manager
# ═══════════════════════════════════════════════════════════════

class MotionPredictorSuite:
    """
    Manages all trajectory prediction models.
    Trains on NGSIM data, provides real-time inference for tracked objects.
    
    Input features (per timestep): [rel_x, rel_y, vx, vy]
    Target: future [delta_x, delta_y] for next 5 frames
    """

    def __init__(self, input_steps=10, output_steps=5, input_features=4):
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.input_features = input_features

        # Models
        self.lr = None
        self.rf = None
        self.gru = None
        self.lstm = None
        self.qnn = VQCTrajectoryRegressor(num_qubits=4, num_var_layers=2)

        # Metrics per model
        self.metrics = {}
        self.is_trained = False
        self.best_model = 'linear_velocity'  # Default fallback

        # Per-scenario model rankings
        self.scenario_rankings = {}

    def train_all(self, X_sequences, y_targets, scenario='highway'):
        """
        Train all models on sequence data.
        
        Args:
            X_sequences: (N, input_steps, input_features) — trajectory sequences
            y_targets: (N, output_steps, 2) — future (delta_x, delta_y) per step
            scenario: scenario name for storing per-scenario metrics
        """
        print(f"[Models] Training all models for '{scenario}' on {len(X_sequences)} sequences...")
        t0 = time.time()

        # Flatten sequences for sklearn models: (N, input_steps * input_features)
        N = len(X_sequences)
        X_flat = X_sequences.reshape(N, -1)
        y_flat = y_targets.reshape(N, -1)  # (N, output_steps * 2)

        # Train/test split (80/20)
        split_idx = int(N * 0.8)
        X_tr_flat, X_te_flat = X_flat[:split_idx], X_flat[split_idx:]
        y_tr_flat, y_te_flat = y_flat[:split_idx], y_flat[split_idx:]
        X_tr_seq, X_te_seq = X_sequences[:split_idx], X_sequences[split_idx:]
        y_tr_seq, y_te_seq = y_targets[:split_idx], y_targets[split_idx:]

        results = {}

        # ── Linear Regression ──
        t_lr = time.time()
        self.lr = MultiOutputRegressor(LinearRegression())
        self.lr.fit(X_tr_flat, y_tr_flat)
        lr_pred = self.lr.predict(X_te_flat)
        results['linear_regression'] = self._compute_metrics(
            y_te_flat, lr_pred, y_te_seq, lr_pred.reshape(-1, self.output_steps, 2),
            time.time() - t_lr, 'Linear Regression'
        )

        # ── Random Forest ──
        t_rf = time.time()
        self.rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        )
        self.rf.fit(X_tr_flat, y_tr_flat)
        rf_pred = self.rf.predict(X_te_flat)
        results['random_forest'] = self._compute_metrics(
            y_te_flat, rf_pred, y_te_seq, rf_pred.reshape(-1, self.output_steps, 2),
            time.time() - t_rf, 'Random Forest'
        )

        # ── GRU ──
        if TORCH_AVAILABLE:
            t_gru = time.time()
            self.gru = GRUPredictor(self.input_features, hidden_size=64, output_steps=self.output_steps)
            gru_pred = self._train_torch_model(
                self.gru, X_tr_seq, y_tr_seq, X_te_seq, epochs=30, lr=0.001
            )
            results['gru'] = self._compute_metrics(
                y_te_flat, gru_pred.reshape(-1, self.output_steps * 2),
                y_te_seq, gru_pred,
                time.time() - t_gru, 'GRU'
            )

            # ── LSTM ──
            t_lstm = time.time()
            self.lstm = LSTMPredictor(self.input_features, hidden_size=64, output_steps=self.output_steps)
            lstm_pred = self._train_torch_model(
                self.lstm, X_tr_seq, y_tr_seq, X_te_seq, epochs=30, lr=0.001
            )
            results['lstm'] = self._compute_metrics(
                y_te_flat, lstm_pred.reshape(-1, self.output_steps * 2),
                y_te_seq, lstm_pred,
                time.time() - t_lstm, 'LSTM'
            )

        # ── QNN (4-Qubit VQC) ──
        t_qnn = time.time()
        # QNN uses 4 features: velocity, acceleration, lane_offset, distance
        # Extract from the last frame of each sequence
        qnn_features_tr = X_tr_seq[:, -1, :4]  # (N, 4)
        qnn_features_te = X_te_seq[:, -1, :4]
        qnn_target_tr = y_tr_seq[:, 0, :]  # Predict first future step
        qnn_target_te = y_te_seq[:, 0, :]

        self.qnn.train(qnn_features_tr, qnn_target_tr, max_iter=50)
        qnn_pred_step1 = self.qnn.predict(qnn_features_te)

        # Extend QNN prediction to all output steps (linear extrapolation from step 1)
        qnn_pred_all = np.zeros((len(X_te_seq), self.output_steps, 2))
        for step in range(self.output_steps):
            qnn_pred_all[:, step, :] = qnn_pred_step1 * (step + 1)

        results['qnn'] = self._compute_metrics(
            y_te_flat, qnn_pred_all.reshape(-1, self.output_steps * 2),
            y_te_seq, qnn_pred_all,
            time.time() - t_qnn, 'QNN (4-Qubit VQC)'
        )
        results['qnn']['circuit_info'] = self.qnn.get_circuit_info()

        # ── Determine winner ──
        # Use ADE as primary metric (lower = better)
        ade_scores = {name: r['ade'] for name, r in results.items()}
        self.best_model = min(ade_scores, key=ade_scores.get)
        
        # Compute rankings
        rankings = sorted(results.items(), key=lambda x: x[1]['ade'])
        for rank, (name, _) in enumerate(rankings):
            results[name]['rank'] = rank + 1

        self.metrics[scenario] = results
        self.scenario_rankings[scenario] = [name for name, _ in rankings]
        self.is_trained = True

        total_time = time.time() - t0
        print(f"[Models] All models trained for '{scenario}' in {total_time:.1f}s. Winner: {self.best_model}")

        return {
            'scenario': scenario,
            'results': results,
            'winner': self.best_model,
            'training_time': round(total_time, 2),
            'train_samples': split_idx,
            'test_samples': N - split_idx,
        }

    def _compute_metrics(self, y_true_flat, y_pred_flat, y_true_seq, y_pred_seq, train_time, name):
        """Compute ADE, FDE, RMSE, R² for a model."""
        # R² and RMSE on flattened
        r2 = float(r2_score(y_true_flat, y_pred_flat))
        rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
        mse = float(mean_squared_error(y_true_flat, y_pred_flat))

        # ADE: Average Displacement Error (mean Euclidean distance across all timesteps)
        displacements = np.sqrt(
            (y_true_seq[:, :, 0] - y_pred_seq[:, :, 0]) ** 2 +
            (y_true_seq[:, :, 1] - y_pred_seq[:, :, 1]) ** 2
        )
        ade = float(np.mean(displacements))

        # FDE: Final Displacement Error (Euclidean distance at last predicted step)
        fde = float(np.mean(np.sqrt(
            (y_true_seq[:, -1, 0] - y_pred_seq[:, -1, 0]) ** 2 +
            (y_true_seq[:, -1, 1] - y_pred_seq[:, -1, 1]) ** 2
        )))

        return {
            'name': name,
            'r2': round(r2, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'ade': round(ade, 4),
            'fde': round(fde, 4),
            'train_time': round(train_time, 2),
        }

    def _train_torch_model(self, model, X_tr, y_tr, X_te, epochs=30, lr=0.001):
        """Train a PyTorch model (GRU or LSTM)."""
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        X_te_t = torch.FloatTensor(X_te)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        batch_size = min(128, len(X_tr))

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_tr))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                batch_idx = indices[i:i + batch_size]
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  [{model.__class__.__name__}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")

        model.eval()
        with torch.no_grad():
            pred = model(X_te_t).numpy()
        return pred

    def predict_realtime(self, sequence, scenario='highway'):
        """
        Real-time prediction for a single object's trajectory sequence.
        Uses the best model for the current scenario.
        
        Args:
            sequence: (1, input_steps, input_features) array
            scenario: current scenario
        
        Returns:
            predictions: (output_steps, 2) future positions
            model_name: which model was used
            uncertainty: (output_steps, 2) uncertainty estimate
        """
        if not self.is_trained:
            return self._linear_velocity_predict(sequence), 'Linear Velocity', np.ones((self.output_steps, 2)) * 0.1

        best = self.scenario_rankings.get(scenario, [self.best_model])[0] if scenario in self.scenario_rankings else self.best_model
        
        try:
            if best == 'lstm' and self.lstm is not None and TORCH_AVAILABLE:
                self.lstm.eval()
                with torch.no_grad():
                    pred = self.lstm(torch.FloatTensor(sequence)).numpy()[0]
                return pred, 'LSTM', np.ones_like(pred) * 0.05

            elif best == 'gru' and self.gru is not None and TORCH_AVAILABLE:
                self.gru.eval()
                with torch.no_grad():
                    pred = self.gru(torch.FloatTensor(sequence)).numpy()[0]
                return pred, 'GRU', np.ones_like(pred) * 0.05

            elif best == 'random_forest' and self.rf is not None:
                flat = sequence.reshape(1, -1)
                pred = self.rf.predict(flat).reshape(self.output_steps, 2)
                return pred, 'Random Forest', np.ones_like(pred) * 0.08

            elif best == 'qnn' and self.qnn.is_trained:
                features = sequence[0, -1, :4].reshape(1, -1)
                pred_mean, pred_std = self.qnn.predict_with_uncertainty(features)
                pred_all = np.zeros((self.output_steps, 2))
                uncert_all = np.zeros((self.output_steps, 2))
                for step in range(self.output_steps):
                    pred_all[step] = pred_mean[0] * (step + 1)
                    uncert_all[step] = pred_std[0] * (step + 1)
                return pred_all, 'QNN (4-Qubit VQC)', uncert_all

            elif best == 'linear_regression' and self.lr is not None:
                flat = sequence.reshape(1, -1)
                pred = self.lr.predict(flat).reshape(self.output_steps, 2)
                return pred, 'Linear Regression', np.ones_like(pred) * 0.1

        except Exception as e:
            print(f"[Models] Prediction error with {best}: {e}")

        # Fallback
        return self._linear_velocity_predict(sequence), 'Linear Velocity', np.ones((self.output_steps, 2)) * 0.1

    def _linear_velocity_predict(self, sequence):
        """Simple linear velocity extrapolation as ultimate fallback."""
        last_pos = sequence[0, -1, :2]  # (rel_x, rel_y)
        last_vel = sequence[0, -1, 2:4]  # (vx, vy)
        dt = 1.0 / 20.0

        future = np.zeros((self.output_steps, 2))
        for i in range(self.output_steps):
            future[i] = last_vel * dt * (i + 1)
        return future

    def get_metrics_for_scenario(self, scenario):
        """Get all model metrics for a specific scenario."""
        return self.metrics.get(scenario, {})

    def get_ranking_for_scenario(self, scenario):
        """Get model ranking for a scenario."""
        return self.scenario_rankings.get(scenario, [])

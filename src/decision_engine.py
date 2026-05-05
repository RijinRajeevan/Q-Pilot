import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBehaviorClassifier(nn.Module):
    def __init__(self, input_size=30, num_classes=3):
        super(MLPBehaviorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        # Random initial weights simulate untrained classifier
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DecisionEngine:
    def __init__(self, risk_threshold=40.0):
        # Risk threshold is physical distance units between vehicles
        self.risk_threshold = risk_threshold
        
        # Load ML Behavior Engine
        self.behavior_classifier = MLPBehaviorClassifier()
        self.behavior_classifier.eval() # Eval mode for inference (no dropping out here)
        
        # Map integer classes to labels
        self.behavior_labels = {
            0: "Normal Cruising",
            1: "Hard Braking",
            2: "Aggressive Switch"
        }

    def extract_behavior(self, flat_tensors):
        """
        Classifies current behaviors from pure T=5 tracked buffer features
        flat_tensors: numpy array shape [batch, 30]
        """
        with torch.no_grad():
            x = torch.from_numpy(flat_tensors).float()
            logits = self.behavior_classifier(x)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).numpy()
            
        return [self.behavior_labels[int(c)] for c in predicted_class]

    def evaluate_scene(self, predictions, raw_features_batch):
        """
        Evaluates the current scene to predict behavior and track collisions
        predictions: list of dicts with current active geometries
        raw_features_batch: aligned flat tensor matching index mapped targets
        """
        if len(predictions) > 0 and raw_features_batch is not None:
            behaviors = self.extract_behavior(raw_features_batch)
            for i, p in enumerate(predictions):
                p['behavior'] = behaviors[i]
                
        for i, pred1 in enumerate(predictions):
            pred1['risk'] = 'safe'
            pred1['collision_warning'] = []
                
            # Collision Matrix (O(N^2) geometric mapping)
            for j, pred2 in enumerate(predictions):
                if i == j: continue
                
                # Check distances mapped across all K=3 future predictions (Not just END)
                # If they intersect at any temporal point, throw warning.
                is_danger = False
                for k_step in range(min(len(pred1['final']), len(pred2['final']))):
                    p1_t = pred1['final'][k_step]
                    p2_t = pred2['final'][k_step]
                    dist = math.sqrt((p1_t['x'] - p2_t['x'])**2 + (p1_t['y'] - p2_t['y'])**2)
                    
                    if dist < self.risk_threshold:
                        is_danger = True
                        break
                
                if is_danger:
                    pred1['risk'] = 'danger'
                    pred1['collision_warning'].append(pred2['id'])
                elif pred1['risk'] != 'danger':
                    # Caution if end state is somewhat close
                    p1_end = pred1['final'][-1]
                    p2_end = pred2['final'][-1]
                    end_dist = math.sqrt((p1_end['x'] - p2_end['x'])**2 + (p1_end['y'] - p2_end['y'])**2)
                    if end_dist < self.risk_threshold * 2.0:
                        pred1['risk'] = 'caution'
                    
        return predictions

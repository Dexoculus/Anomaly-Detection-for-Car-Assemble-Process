import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

class AnomalTester:
    """
    Overridden Tester class for AutoEncoder-based anomaly detection + Grid Search.
    """

    def __init__(self, model, test_loader, config, device):
        """
        Args:
            model (nn.Module): Trained AutoEncoder model.
            test_loader (DataLoader): Test data loader.
            config (dict): Experiment configuration (YAML).
            device (torch.device): CPU or GPU device.
        """
        print("[Initializing] Starting CustomTester initialization...")
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.results = {}
        print("[Initializing] CustomTester initialized successfully.")

    def test(self):
        """
        1) Reconstructs inputs by Auto-Encoder.
        2) Calculates reconstuction error.
        3) Classifies OK/NG by threshold
        4) Calculates Precision, Recall, F1.
        5) Search optimal threshold by Grid Search.
        """
        if self.test_loader is None:
            print("[Testing] No test data provided. Exiting test process.")
            return

        print("[Testing] Starting the test process for anomaly detection...")
        self.model.eval()

        reconstruction_errors = []
        true_labels = []

        threshold = self.config.get("testing", {}).get("threshold", 0.5)
        print(f"[Testing] Using threshold={threshold}")

        # Check Reconstruction Error of each Sample
        criterion = nn.MSELoss(reduction='none')

        with tqdm(total=len(self.test_loader), desc="[Testing Progress]", bar_format="{l_bar}{bar:40}{r_bar}", leave=True) as pbar:
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    recon = self.model(inputs) 
                    # Calculate MSE
                    errors = criterion(recon, inputs)  # shape: (batch, seq_len, input_dim)
                    errors = errors.mean(dim=(1, 2))   # (batch,)
                    errors = errors.cpu().numpy()

                    reconstruction_errors.extend(errors)
                    true_labels.extend(labels.cpu().numpy())
                    pbar.update(1)

        reconstruction_errors = np.array(reconstruction_errors)
        true_labels = np.array(true_labels)

        # Classifies OK(0)/NG(1) by threshold
        pred_labels = (reconstruction_errors > threshold).astype(int)

        # Precision, Recall, F1
        precision = precision_score(true_labels, pred_labels, pos_label=1)
        recall = recall_score(true_labels, pred_labels, pos_label=1)
        f1 = f1_score(true_labels, pred_labels, pos_label=1)

        print(f"[Testing] Reconstruction Errors => min={reconstruction_errors.min():.6f}, "
              f"max={reconstruction_errors.max():.6f}, mean={reconstruction_errors.mean():.6f}")
        print(f"[Testing] Current Threshold: {threshold}")
        print(f"[Testing] Precision: {precision:.4f}")
        print(f"[Testing] Recall:    {recall:.4f}")
        print(f"[Testing] F1-score: {f1:.4f}")

        # Grid Search
        best_threshold, best_metrics = self._grid_search_threshold(reconstruction_errors, true_labels)
        print("\n[Testing] Grid Search Results")
        print(f"Best Threshold: {best_threshold:.6f}")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.4f}")

        self.results = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "reconstruction_errors": reconstruction_errors.tolist(),
            "true_labels": true_labels.tolist(),
            "pred_labels": pred_labels.tolist(),
            "best_threshold": float(best_threshold),
            "best_metrics": best_metrics
        }

        print("[Testing] Test process completed with anomaly detection mode.")

    def _grid_search_threshold(self, reconstruction_errors, true_labels, num_points=50):
        """
        Search optimal threshold by Grid Search.
        Args:
            reconstruction_errors (np.ndarray): Reconstruction Error Array of (N,) shape.
            true_labels (np.ndarray): OK(0)/NG(1) Label of (N,) shape.
            num_points (int): Number of sampling for Grid Search.

        Returns:
            best_threshold (float): The Optimal Threshold maximize F1-score value.
            best_metrics (dict): Precision, Recall, F1-score at Optimal Threshold.
        """
        min_error = reconstruction_errors.min()
        max_error = reconstruction_errors.max()
        candidates = np.linspace(min_error, max_error, num_points)

        best_threshold = None
        best_f1 = 0.0
        best_metrics = {}

        for t in candidates:
            preds = (reconstruction_errors > t).astype(int)

            precision = precision_score(true_labels, preds, pos_label=1, zero_division=0)
            recall = recall_score(true_labels, preds, pos_label=1, zero_division=0)
            f1 = f1_score(true_labels, preds, pos_label=1, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_metrics = {
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1-score": float(f1)
                }

        return best_threshold, best_metrics

    def get_results(self):
        """Returns the recorded test results dictionary."""

        return self.results
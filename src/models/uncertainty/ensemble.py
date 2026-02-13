"""Ensemble uncertainty quantification."""
import numpy as np

class EnsemblePolicy:
    """Ensemble of policies for uncertainty estimation."""
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
    
    def predict_with_uncertainty(self, state):
        # Returns mean and std
        return np.zeros(10), np.ones(10)

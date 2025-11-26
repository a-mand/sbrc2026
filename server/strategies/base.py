from abc import ABC, abstractmethod
from collections import OrderedDict
import torch

class Strategy(ABC):
    """
    Abstract base class for Federated Learning aggregation strategies.
    """
    
    @abstractmethod
    def aggregate(self, updates):
        """
        Aggregates model updates from clients.
        
        Args:
            updates: List of dicts containing 'model_update' (state_dict) and 'num_samples'.
            
        Returns:
            aggregated_state_dict: The new global model state_dict.
        """
        pass
import torch
import logging
from .base import Strategy
from collections import OrderedDict

logger = logging.getLogger(__name__)

class FedAvg(Strategy):
    def aggregate(self, updates):
        if not updates:
            return None

        # 1. Calculate total samples
        total_samples = sum(u['num_samples'] for u in updates)
        if total_samples == 0:
            return None

        # 2. Initialize accumulator with the first model structure
        # We use the first update as a template
        first_state = updates[0]['model_update']
        avg_state = OrderedDict()
        
        for key, tensor in first_state.items():
            avg_state[key] = torch.zeros_like(tensor, dtype=torch.float32)

        # 3. Weighted Average
        for update in updates:
            client_state = update['model_update']
            weight = update['num_samples'] / total_samples
            
            for key in avg_state:
                if key in client_state:
                    # Ensure type matching
                    client_tensor = client_state[key].to(dtype=torch.float32)
                    avg_state[key] += client_tensor * weight

        return avg_state
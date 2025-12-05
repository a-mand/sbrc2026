import torch
from .base import Strategy
from .fedavg import FedAvg
from collections import OrderedDict

class ScaffoldStrategy(Strategy):
    def __init__(self, global_model):
        self.fedavg = FedAvg() # Reuse basic averaging logic
        self.global_c = {}
        
        # Initialize global control variate to zeros
        for name, param in global_model.named_parameters():
            self.global_c[name] = torch.zeros_like(param)

    def aggregate(self, updates):
        # 1. Aggregate Model Weights (Standard FedAvg)
        new_model_state = self.fedavg.aggregate(updates)
        if new_model_state is None:
            return None
            
        # 2. Aggregate Control Variate Updates (Delta C)
        # c_global += (1 / N) * sum(delta_c_i)
        total_clients = len(updates) # Or total participating
        
        if total_clients > 0:
            for update in updates:
                # Get the extra payload we sent from client
                delta_c = update.get("metrics", {}).get("scaffold_delta_c")
                
                if delta_c:
                    for key in self.global_c:
                        # c = c + (1/N) * delta
                        self.global_c[key] += delta_c[key] / total_clients
        
        # 3. Return COMPOSITE payload
        # Server sends { model, global_c } to clients next round
        return {
            "model_state": new_model_state,
            "extra_payload": self.global_c
        }
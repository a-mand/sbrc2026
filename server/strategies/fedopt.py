import torch
import logging
import copy
from collections import OrderedDict
from .fedavg import FedAvg

logger = logging.getLogger(__name__)

class FedOpt(FedAvg):
    """
    Federated Optimization strategy (FedAvgM, FedAdam, FedYogi).
    It uses a standard FedAvg aggregation to get a 'pseudo-gradient',
    then applies a server-side optimizer.
    """
    def __init__(self, global_model, optimizer_cls=torch.optim.SGD, **optimizer_args):
        self.global_model = global_model
        # Initialize the server-side optimizer (e.g., SGD with momentum)
        self.optimizer = optimizer_cls(self.global_model.parameters(), **optimizer_args)
        # We need to keep the model in memory to apply gradients
        self.optimizer.zero_grad()

    def aggregate(self, updates):
        # 1. Perform standard FedAvg to get the "average weight"
        avg_state = super().aggregate(updates)
        if avg_state is None:
            return None

        # 2. Compute "Pseudo-Gradient"
        # pseudo_grad = global_model - avg_model
        # This represents the direction the clients want to move
        
        current_state = self.global_model.state_dict()
        pseudo_gradients = OrderedDict()
        
        for name, param in self.global_model.named_parameters():
            if name in avg_state:
                # The 'gradient' is the difference between current and average
                # We move avg_state to the same device as param for math
                avg_tensor = avg_state[name].to(param.device)
                
                # Gradient = Current - Target (Standard optimization direction)
                # Note: Usually we want w_new = w_old - lr * grad
                # Here w_avg is the target. So (w_old - w_avg) acts like a gradient.
                pseudo_gradients[name] = (param.data - avg_tensor)

        # 3. Apply to Server Optimizer
        self.optimizer.zero_grad()
        
        for name, param in self.global_model.named_parameters():
            if name in pseudo_gradients:
                # Manually set the gradient
                param.grad = pseudo_gradients[name]

        # Step the server optimizer (Apply Momentum/Adam logic)
        self.optimizer.step()
        
        # Return the new parameters
        return self.global_model.state_dict()
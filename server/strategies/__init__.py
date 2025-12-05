from .fedavg import FedAvg
from .fedopt import FedOpt
from .scaffold import ScaffoldStrategy
import torch.optim as optim

def get_strategy(strategy_name, global_model=None, lr=1.0, momentum=0.0):
    """
    Factory to return the requested strategy.
    """
    if strategy_name == "FedAvg":
        return FedAvg()
    
    elif strategy_name == "Scaffold":   # <--- ADD THIS
        if global_model is None:
            raise ValueError("Scaffold requires global_model")
        return ScaffoldStrategy(global_model)
    
    elif strategy_name == "FedAvgM":
        # FedAvgM is just FedOpt with SGD + Momentum
        if global_model is None:
            raise ValueError("FedAvgM requires global_model instance")
        return FedOpt(
            global_model, 
            optimizer_cls=optim.SGD, 
            lr=lr, 
            momentum=momentum
        )
        
    elif strategy_name == "FedAdam":
        # Server-side Adam
        if global_model is None:
            raise ValueError("FedAdam requires global_model instance")
        return FedOpt(
            global_model,
            optimizer_cls=optim.Adam,
            lr=lr # Server learning rate
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
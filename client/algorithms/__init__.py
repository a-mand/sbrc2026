from .standard import StandardSGD
from .fedprox import FedProx
from .scaffold import Scaffold # <--- ADD THIS
import config

def get_client_algorithm(algo_name):
    if algo_name == "Standard":
        return StandardSGD()
    elif algo_name == "FedProx":
        return FedProx(mu=config.FEDPROX_MU)
    elif algo_name == "Scaffold":     # <--- ADD THIS
        return Scaffold()
    else:
        raise ValueError(f"Unknown Client Algorithm: {algo_name}")
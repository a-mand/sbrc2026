import logging
import time
import torch
import torch.cuda
import psutil
import os
import config

logger = logging.getLogger(__name__)

def train_model(model, client_data, algorithm, extra_payload=None):
    """
    Wrapper that executes the selected Client Algorithm.
    
    Args:
        model: The neural network model to train
        client_data: DataLoader containing client's training data
        algorithm: The client algorithm instance (e.g., StandardSGD, FedProx, Scaffold)
        extra_payload: Optional algorithm-specific data from server (e.g., global_c for Scaffold)
    
    Returns:
        tuple: (num_samples, peak_gpu_mb, peak_ram_mb, metrics_dict)
    """
    
    # 1. Setup Device
    forced_device = config.DEVICE.lower()
    if forced_device == "cpu":
        device = torch.device("cpu")
    elif forced_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info(f"Training with algorithm: {config.CLIENT_ALGO} on {device}")

    # 2. Track Resources - Start
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process(os.getpid())
    start_time = time.time()

    # 3. EXECUTE TRAINING
    # Pass extra_payload (e.g., Global C for Scaffold) to the algorithm
    # The algorithm's train() method should accept global_c as a kwarg
    try:
        metrics = algorithm.train(
            model, 
            client_data, 
            device, 
            config.LOCAL_EPOCHS, 
            global_c=extra_payload
        )
    except TypeError as e:
        # Fallback for algorithms that don't accept global_c parameter
        logger.warning(f"Algorithm {config.CLIENT_ALGO} doesn't accept global_c parameter. "
                      f"Calling without extra_payload.")
        metrics = algorithm.train(model, client_data, device, config.LOCAL_EPOCHS)
    
    # Ensure metrics is a dictionary
    if metrics is None:
        metrics = {}
    elif not isinstance(metrics, dict):
        logger.warning(f"Algorithm returned non-dict metrics: {type(metrics)}. Converting to empty dict.")
        metrics = {}

    # 4. Capture Metrics - End
    end_time = time.time()
    num_samples = len(client_data.dataset)
    training_time_sec = end_time - start_time
    
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    peak_gpu_mb = peak_gpu_bytes / (1024 * 1024)
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    # Log training completion
    logger.info(f"Training completed: {num_samples} samples in {training_time_sec:.2f}s")
    logger.info(f"Resource usage - GPU: {peak_gpu_mb:.2f}MB, RAM: {peak_ram_mb:.2f}MB")

    # 5. Return all metrics
    # metrics dict may contain algorithm-specific data like 'delta_c' for Scaffold
    return num_samples, peak_gpu_mb, peak_ram_mb, metrics
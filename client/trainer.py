import logging
import time
import torch
import torch.cuda
import psutil
import os
import numpy as np
import config

logger = logging.getLogger(__name__)

def calculate_entropy(probabilities):
    """Calcula a entropia de Shannon usando NumPy para evitar dependência do SciPy."""
    probs = np.array(probabilities)
    probs = probs[probs > 0]  # Filtra zeros para evitar erro no log
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs))

def train_model(model, client_data, algorithm, extra_payload=None, extra_payload2=None, strategy="treino_normal"):
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
    all_labels = []
    all_features = []
    for inputs, labels in client_data:
        all_labels.extend(labels.tolist())
        all_features.append(inputs.view(inputs.size(0), -1).mean(dim=0).numpy())

    # Qualidade de dados
    counts = np.bincount(all_labels)
    probs = counts / len(all_labels)
    entropy_val = float(calculate_entropy(probs))

    stats_summary = np.mean(all_features, axis=0).tolist()

    local_epochs = config.LOCAL_EPOCHS
    if strategy == "modelo_leve":
        local_epochs = max(1, config.LOCAL_EPOCHS // 2)
        logger.warning(f"Estratégia modelo_leve ativada. Local epochs reduzidos para {local_epochs}.")

    # 2. Track Resources - Start
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process(os.getpid())
    start_time = time.time()

    # 3. EXECUTE TRAINING
    # Pass extra_payload (e.g., Global C for Scaffold) to the algorithm
    # The algorithm's train() method should accept global_c as a kwarg
    logger.info(f"Starting training for {local_epochs} epochs.")

    train_kwargs = {
        "model": model,
        "dataloader": client_data,
        "device": device,
        "local_epochs": local_epochs
    }

    if config.CLIENT_ALGO.lower() == "scaffold" and extra_payload is not None:
        train_kwargs["global_c"] = extra_payload

    try:
        metrics = algorithm.train(**train_kwargs)
    except TypeError as e:
        # Fallback for algorithms that don't accept global_c parameter
        logger.warning(f"Algorithm {config.CLIENT_ALGO} doesn't accept global_c parameter. "
                      f"Calling without extra_payload.")
        metrics = algorithm.train(model, client_data, device, local_epochs)
    
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

    metrics.update({
        "entropy": entropy_val,
        "stats_summary": stats_summary,
        "strategy_used": strategy,
        "training_time_sec": training_time_sec,
    })
    # Log training completion
    logger.info(f"Training completed: {num_samples} samples in {training_time_sec:.2f}s, Entropy: {entropy_val:.4f}")
    logger.info(f"Resource usage - GPU: {peak_gpu_mb:.2f}MB, RAM: {peak_ram_mb:.2f}MB")

    # 5. Return all metrics
    # metrics dict may contain algorithm-specific data like 'delta_c' for Scaffold
    return num_samples, peak_gpu_mb, peak_ram_mb, metrics

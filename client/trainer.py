import torch
import torch.nn as nn
import torch.optim as optim
import logging
import config
import torch.cuda
import psutil
import os
import copy  # <--- ADDED for FedProx

# Get a logger for this module
logger = logging.getLogger(__name__)


def train_model(model, client_data):
    """
    Performs local training on the client's model with FedProx support.
    
    Returns:
        (int) num_samples: Number of samples trained on.
        (float) peak_gpu_mb: Peak GPU memory used in megabytes.
        (float) peak_ram_mb: Peak System RAM used in megabytes.
    """
    
    # --- Set device based on config ---
    forced_device = config.DEVICE.lower()
    if forced_device == "cpu":
        device = torch.device("cpu")
    elif forced_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available! Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:  # "auto" or any other value
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    logger.info(f"Training on device: {device}")
    
    # --- GPU Memory Stats ---
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
    # --- System RAM Stats ---
    process = psutil.Process(os.getpid())
    peak_ram_mb = 0.0

    data_loader = client_data 
    num_samples = len(data_loader.dataset)
    
    # --- FEDPROX: Save initial global weights ---
    # We need a frozen copy of the global model to compute the proximal term
    if config.FEDPROX_MU > 0:
        global_model_params = [p.clone().detach() for p in model.parameters()]
        logger.info(f"FedProx enabled with mu={config.FEDPROX_MU}")
    # ---------------------------------------------
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    model.train() 

    for epoch in range(config.LOCAL_EPOCHS):
        running_loss = 0.0
        
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # --- FEDPROX: Add Proximal Term ---
            # Loss += (mu / 2) * || w - w_global ||^2
            if config.FEDPROX_MU > 0:
                proximal_term = 0.0
                for w, w_global in zip(model.parameters(), global_model_params):
                    # Ensure w_global is on the same device as w
                    w_global = w_global.to(device)
                    proximal_term += (w - w_global).norm(2) ** 2
                
                loss += (config.FEDPROX_MU / 2) * proximal_term
            # ----------------------------------

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # --- Check RAM usage per batch ---
            current_ram_bytes = process.memory_info().rss
            peak_ram_mb = max(peak_ram_mb, current_ram_bytes / (1024 * 1024))
        
        epoch_loss = running_loss / len(data_loader)
        logger.info(f"Epoch {epoch+1}/{config.LOCAL_EPOCHS} - Loss: {epoch_loss:.4f}")

    logger.info("Local training complete.")
    
    # --- Get peak GPU memory usage ---
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    peak_gpu_mb = peak_gpu_bytes / (1024 * 1024)

    # Return all three metrics
    return num_samples, peak_gpu_mb, peak_ram_mb
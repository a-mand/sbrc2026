import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data():
    logger.info("--- Starting Data Preparation ---")
    
    # 1. Download CIFAR-10 (if not exists)
    # We use the same transform as training to ensure compatibility
    logger.info("Checking/Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download to ./data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Also ensure test set is downloaded for the server
    torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 2. Calculate Non-IID Partitions
    logger.info(f"Partitioning data for {config.TOTAL_CLIENTS} clients (Alpha={config.DIRICHLET_ALPHA})...")
    
    np.random.seed(config.RANDOM_SEED)
    
    num_classes = 10
    labels = np.array(train_dataset.targets)
    
    # Get indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_partitions = [[] for _ in range(config.TOTAL_CLIENTS)]

    # Dirichlet distribution logic
    for k in range(num_classes):
        img_idx_k = class_indices[k]
        np.random.shuffle(img_idx_k)
        
        # Generate proportions
        proportions = np.random.dirichlet(np.repeat(config.DIRICHLET_ALPHA, config.TOTAL_CLIENTS))
        
        # Calculate split points
        proportions_cumsum = (np.cumsum(proportions) * len(img_idx_k)).astype(int)[:-1]
        
        # Split and assign
        split_indices = np.split(img_idx_k, proportions_cumsum)
        for i in range(config.TOTAL_CLIENTS):
            client_partitions[i].extend(split_indices[i].tolist())

    # 3. Save partitions to JSON
    partition_file = './data/partitions.json'
    logger.info(f"Saving partitions to {partition_file}...")
    
    # Convert to a dictionary: "1": [indices], "2": [indices]
    partition_dict = {
        str(i + 1): indices 
        for i, indices in enumerate(client_partitions)
    }
    
    with open(partition_file, 'w') as f:
        json.dump(partition_dict, f)
        
    logger.info("✅ Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
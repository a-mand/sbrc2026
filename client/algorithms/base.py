from abc import ABC, abstractmethod

class ClientAlgorithm(ABC):
    @abstractmethod
    def train(self, model, dataloader, device, epochs):
        """
        Executes the local training loop.
        
        Args:
            model: The global model to train (will be updated in-place or returned).
            dataloader: Local data.
            device: CPU or GPU.
            epochs: Number of local epochs.
            
        Returns:
            metrics: dict (e.g., training_time, loss, etc.)
        """
        pass
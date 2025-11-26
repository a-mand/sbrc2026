import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 (3 channels).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleMLP(nn.Module):
    """
    A simple MLP for flattened inputs.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512) # CIFAR is 32x32x3
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Model Registry ---
def get_model(model_name):
    """Factory function to instantiate models by name."""
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "SimpleMLP":
        return SimpleMLP()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
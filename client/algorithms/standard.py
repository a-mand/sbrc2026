import torch
import torch.nn as nn
import torch.optim as optim
import config
from .base import ClientAlgorithm

class StandardSGD(ClientAlgorithm):
    def train(self, model, dataloader, device, epochs):
        model.to(device)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              momentum=config.MOMENTUM)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        return {} # Return empty metrics or custom logs
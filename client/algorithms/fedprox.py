import torch
import torch.nn as nn
import torch.optim as optim
import copy
import config
from .base import ClientAlgorithm

class FedProx(ClientAlgorithm):
    def __init__(self, mu=0.01):
        self.mu = mu

    def train(self, model, dataloader, device, epochs):
        model.to(device)
        model.train()
        
        # 1. Save state of global model (The "Anchor")
        global_model_params = copy.deepcopy(list(model.parameters()))
        
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              momentum=config.MOMENTUM)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                
                # 2. Calculate Standard Loss
                loss = criterion(outputs, labels)
                
                # 3. Add Proximal Term
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model_params):
                    w_t = w_t.to(device)
                    proximal_term += (w - w_t).norm(2)
                
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
        return {"fedprox_mu": self.mu}
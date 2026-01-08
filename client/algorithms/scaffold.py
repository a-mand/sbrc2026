import torch
import torch.nn as nn
import torch.optim as optim
import copy
import config
from .base import ClientAlgorithm

class Scaffold(ClientAlgorithm):
    def __init__(self):
        self.local_c = None # Persistent local control variate

    def train(self, model, dataloader, device, local_epochs, global_c=None):
        model.to(device)
        model.train()
        
        # 1. Initialize Control Variates
        if self.local_c is None:
            self.local_c = {}
            for name, param in model.named_parameters():
                self.local_c[name] = torch.zeros_like(param)
                
        if global_c is None:
            # Should not happen in proper Scaffold run, but fallback safe
            global_c = {}
            for name, param in model.named_parameters():
                global_c[name] = torch.zeros_like(param)

        # Move variates to GPU
        local_c = {k: v.to(device) for k, v in self.local_c.items()}
        global_c_gpu = {k: v.to(device) for k, v in global_c.items()}
        
        # Keep copy of initial global model (for c update rule later)
        global_model_init = copy.deepcopy(model.state_dict())

        optimizer = optim.SGD(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              momentum=0) # SCAFFOLD usually assumes no momentum

        # 2. Training Loop
        count = 0
        for local_epoch in range(local_epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                
                # --- SCAFFOLD CORRECTION ---
                # w_new = w - lr * (grads - c_i + c)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Add (c - c_i) to the gradient
                        correction = global_c_gpu[name] - local_c[name]
                        param.grad += correction
                # ---------------------------
                
                optimizer.step()
                count += 1

        # 3. Update Local Control Variate
        # c_i+ = c_i - c + (1 / (K * lr)) * (x - y_i)
        # where x = global_model, y_i = new_local_model
        
        lr = config.LEARNING_RATE
        K = count # Total steps
        
        delta_c = {} # Change to send to server
        
        new_local_c = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                # x - y_i
                model_diff = global_model_init[name].to(device) - param.data
                
                # term = (1 / (K*lr)) * (x - y)
                term = model_diff / (K * lr)
                
                # c_new = c_local - c_global + term
                new_val = local_c[name] - global_c_gpu[name] + term
                new_local_c[name] = new_val
                
                # Calculate diff to send to server (Delta C)
                delta_c[name] = (new_val - local_c[name]).cpu()

        # Update persistent local state
        self.local_c = {k: v.cpu() for k, v in new_local_c.items()}

        # Return the delta to the server
        return {"scaffold_delta_c": delta_c}
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from .actor_critic import get_activation

class Estimator(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim=[256, 128], 
                 activation="relu",
                 lr = 1e-4,
                 **kwargs):
        if kwargs:
            print("MLP.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Estimator, self).__init__()
        
        activation = get_activation(activation)
        
        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        mlp_layers.append(activation)
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                mlp_layers.append(nn.Linear(hidden_dim[i], output_dim))
            else:
                mlp_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.init_weights()
        self.lr = lr
        
        print(f"Estimator: {self.mlp}")
    
    def forward(self, x):
        return self.mlp(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def loss(self, y_hat, y):
        return torch.mean(F.mse_loss(y_hat, y))
    
    def traning_step(self, features_batch, labels_batch):
        l = self.loss(self(features_batch), labels_batch)
        return l
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer
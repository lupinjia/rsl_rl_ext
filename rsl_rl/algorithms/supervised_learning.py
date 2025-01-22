import time
import os
from collections import deque
import numpy as np
import torch
from rsl_rl.modules import Estimator
from rsl_rl.storage import EstimatorBuffer

class SupervisedLearning:
    def __init__(self, model:Estimator, data:EstimatorBuffer, max_epochs, num_mini_batches, lr_scheduler_cfg_dict):
        self.model = model
        self.data = data
        self.max_epochs = max_epochs
        self.num_mini_batches = num_mini_batches
        
        self.optimizer = self.model.configure_optimizers()
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler_cfg_dict)
    
    def fit(self):
        mean_loss = 0
        generator = self.data.mini_batch_generator(self.num_mini_batches, self.max_epochs)
        for features_batch, labels_batch in generator:
            loss = self.model.traning_step(features_batch, labels_batch)
            mean_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        num_updates = self.max_epochs * self.num_mini_batches
        mean_loss /= num_updates
        return mean_loss

def get_lr_scheduler(optimizer, lr_scheduler_cfg_dict):
    if lr_scheduler_cfg_dict["type"] == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(optimizer, lr_scheduler_cfg_dict["start_factor"],
                                                 lr_scheduler_cfg_dict["end_factor"], lr_scheduler_cfg_dict["total_iters"])
    elif lr_scheduler_cfg_dict["type"] == None:
        return None
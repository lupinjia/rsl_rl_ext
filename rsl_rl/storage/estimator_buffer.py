import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class EstimatorBuffer:
    class Transition:
        def __init__(self):
            self.feature = None
            self.label = None
        
        def clear(self):
            self.__init__()
    
    def __init__(self, num_envs, num_transitions_per_env, feature_len, label_len, device="cpu"):
        self.device = device
        self.feature_len = feature_len
        self.label_len = label_len
        self.batch_size = num_envs * num_transitions_per_env
        
        # Core
        self.features = torch.zeros(num_transitions_per_env, num_envs, feature_len, device=self.device)
        self.labels = torch.zeros(num_transitions_per_env, num_envs, label_len, device=self.device)
        
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
    
    def clear(self):
        self.step = 0
    
    def add_transition(self, feature, label):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.features[self.step].copy_(feature)
        self.labels[self.step].copy_(label)
        self.step += 1
    
    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        mini_batch_size = self.batch_size // num_mini_batches
        indices = torch.randperm(self.batch_size, requires_grad=False, device=self.device)
        
        features = self.features.flatten(0, 1)
        labels = self.labels.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i+1) * mini_batch_size
                batch_idx = indices[start:end]
                
                features_batch = features[batch_idx]
                labels_batch = labels[batch_idx]
                yield features_batch, labels_batch
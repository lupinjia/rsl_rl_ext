import time
import os
from collections import deque
import numpy as np
import torch
from rsl_rl.modules import MLP
from rsl_rl.storage import EstimatorBuffer
from tqdm import tqdm

class SupervisedLearning: # for rsl_rl compatible training
    def __init__(self, model:MLP, data:EstimatorBuffer, max_epochs, num_mini_batches, lr_scheduler_cfg_dict):
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

class SupervisedTrainer: # for custom whole-process supervised learning
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
    
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    
    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        with tqdm(total=self.max_epochs, desc='Epochs') as pbar: # 进度条
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()
                # 一个epoch后进度条更新
                pbar.update(1)
    
    def fit_epoch(self): # 训练一个epoch
        self.model.train() # 切换为训练模式
        for batch in self.train_dataloader: # 遍历所有的batch
            loss = self.model.training_step(batch) # 计算loss
            self.optim.zero_grad() # 梯度清零
            with torch.no_grad():
                loss.backward() # 反向传播
                self.optim.step() # 更新参数
            self.train_batch_idx += 1 # 训练batch数加1
        if self.val_dataloader is None:
            return
        self.model.eval() # 切换为验证模式
        for batch in self.val_dataloader: # 遍历所有的batch
            with torch.no_grad():
                self.model.validation_step(batch) # 计算验证指标
            self.val_batch_idx += 1 # 验证batch数加1
    
    def test(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.model.eval()
        self.test_batch_idx = 0
        for batch in data.test_dataloader():
            with torch.no_grad():
                self.model.test_step(batch, self.test_batch_idx)
            self.test_batch_idx += 1
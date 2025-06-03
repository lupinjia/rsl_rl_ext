import torch
import os
import pandas as pd
import numpy as np

class RegressionData:
    def __init__(self, data_dir, 
                 feature_file_name,
                 label_file_name,
                 batch_size):
        self.data_dir = data_dir
        self.feature_file_name = feature_file_name
        self.label_file_name = label_file_name
        print('Data directory: {}'.format(self.data_dir))
        self.batch_size = batch_size
        self.prepare_data() # prepare the data

    def prepare_data(self):
        # read data from local file
        feature_file_path = os.path.join(self.data_dir, self.feature_file_name)
        label_file_path = os.path.join(self.data_dir, self.label_file_name)
        features_df = pd.read_csv(feature_file_path)
        labels_df = pd.read_csv(label_file_path)
        self.num_sample, self.num_feature = features_df.shape
        num_output = labels_df.shape[1]
        self.num_train = int(self.num_sample*0.6) # 前60%为训练集, 后20%为验证集, 后20%为测试集
        self.num_val = int(self.num_sample*0.2)
        self.num_test = self.num_sample - self.num_train - self.num_val
        print('Number of train and val:', self.num_train+self.num_val)
        self.features = torch.tensor(features_df.values, dtype=torch.float32).to('cuda:0')
        self.labels = torch.tensor(labels_df.values, dtype=torch.float32).reshape(-1, num_output).to('cuda:0')
        print('Features shape: {}; Labels shape: {}'.format(self.features.shape, self.labels.shape))
        
    def get_tensorloader(self, tensor, train, indices=slice(0, None)):
        tensors = tuple(t[indices] for t in tensor)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def get_dataloader(self, train):
        train_indices = list(range(self.num_train))
        val_indices = list(range(self.num_train, self.num_train+self.num_val))
        indices = train_indices if train else val_indices
        return self.get_tensorloader((self.features, self.labels), train, indices)
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def test_dataloader(self):
        indices = list(range(self.num_train+self.num_val, self.num_sample))
        return self.get_tensorloader((self.features, self.labels), False, indices)
    
    def get_single_feature(self, index):
        return self.features[index]
    
    def get_single_label(self, index):
        return self.labels[index]
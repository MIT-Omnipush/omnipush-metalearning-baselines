from torch import nn
from torch.distributions.normal import Normal
from tqdm import tqdm as Tqdm
import json
import torch
import random
import h5py
import numpy as np
from torch.utils.data import DataLoader
from preprocess import collate_fn, omnipush, omnipush_collate_fn
import os
from tensorboardX import SummaryWriter
#Load data
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
original_n = 'data/all_shapes.hdf5'
old_1k_n = 'data/data_old_1k.hdf5'
old_ply_n = 'data/data_plywood_oldobjects.hdf5'
new_ply_n = 'data/data_plywood_newobjects.hdf5'

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1= nn.Linear(3, 256)
        self.fc2= nn.Linear(256,256)
        self.fc3= nn.Linear(256,256)
        self.fc4= nn.Linear(256, 6)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = x
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

epochs = 3000 ; global_step = 0
def evaluate(loader, name):
    for i,data in enumerate(loader):
        X, y = data
        X = X.cuda() ; y = y.cuda()
        pred = model(X)
        mu, log_std = torch.chunk(pred,2,dim=-1)
        mse = nn.MSELoss()(y, mu)
        log_p = Normal(loc=mu, scale=torch.exp(log_std)).log_prob(y)
        # print(log_p.shape)
        log_p = torch.mean(log_p)
        writer.add_scalar(name+'/RMSE', torch.sqrt(mse), global_step)
        writer.add_scalar(name+'/logP', log_p*y.shape[-1], global_step)


Data = omnipush(dir_path='./data/bias/', return_all=True)
train_datasets = []; test_datasets = []
train_loaders = []; test_loaders = []
for i in range(5):
    X = Data[1][i]
    y = Data[2][i]
    print(X.shape, y.shape)
    train_datasets.append(torch.utils.data.TensorDataset(X[:2000], y[:2000]))
    test_datasets.append(torch.utils.data.TensorDataset(X[2000:], y[2000:]))
    train_loaders.append(DataLoader(train_datasets[-1], batch_size=256,
            shuffle=True, num_workers=32))
    test_loaders.append(DataLoader(test_datasets[-1], batch_size=256,
            shuffle=True, num_workers=32))
models = [NeuralNet().cuda() for f in train_datasets]
opts = [torch.optim.Adam(m.parameters(), lr=3e-3) for m in models]

# Training and Testing
writer = SummaryWriter(comment='bias-gooddata-norm')
for epoch in Tqdm(range(epochs)):
    train_mse = 0.
    test_mse = 0.
    train_log_p = 0.
    test_log_p = 0.
    global_step += 1
    for (model, opt, train_loader, test_loader) in zip(
            models, opts, train_loaders, test_loaders):
        model.train()
        for i, data in enumerate(train_loader):
            X, y = data
            X = X.cuda() ; y = y.cuda()
            pred = model(X)
            mu, log_std = torch.chunk(pred,2,dim=-1)
            mse = nn.MSELoss()(y, mu)
            log_p = Normal(loc=mu, scale=torch.exp(log_std)).log_prob(y)
            log_p = torch.mean(log_p)
            loss = -log_p
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar('train-bias/RMSE', torch.sqrt(mse), global_step)
            writer.add_scalar('train-bias/logP',
                    log_p*y.shape[-1], global_step)
        model.train(False)
        evaluate(test_loader, 'test-bias')

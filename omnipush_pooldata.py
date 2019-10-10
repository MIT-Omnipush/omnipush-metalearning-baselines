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
model = NeuralNet().cuda()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
ctt_std = False

comment = 'baseline-ctt_std-'+str(ctt_std)
writer = SummaryWriter(comment=comment)
train_names, train_X, train_y = omnipush('./data/all_omnipush', train=True)
test_names, test_X, test_y = omnipush('./data/all_omnipush', train=False)
train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=64,
        shuffle=True, num_workers=32)
test_loader = DataLoader(test_dataset, batch_size=64,
        shuffle=True, num_workers=32)

old_names, old_X, old_y = omnipush('./data/old_shapes', return_all=True)
plywood_names, plywood_X, plywood_y = omnipush('./data/plywood',
        return_all=True)
plyold_names, plyold_X, plyold_y = omnipush('./data/plyold',
        return_all=True)
old_dataset = torch.utils.data.TensorDataset(old_X, old_y)
old_loader = DataLoader(old_dataset, batch_size=64,
        shuffle=True, num_workers=32)
plywood_dataset = torch.utils.data.TensorDataset(plywood_X, plywood_y)
plywood_loader = DataLoader(plywood_dataset, batch_size=64,
        shuffle=True, num_workers=32)
plyold_dataset = torch.utils.data.TensorDataset(plyold_X, plyold_y)
plyold_loader = DataLoader(plyold_dataset, batch_size=64,
        shuffle=True, num_workers=32)

# Training and Testing
epochs = 3000 ; global_step = 0
hard_std = torch.Tensor(np.array([1.,1.,1.])).cuda().unsqueeze(0)
def evaluate(loader, name):
    for i,data in enumerate(loader):
        X, y = data ; X = X.cuda() ; y = y.cuda()
        X = X.reshape(-1, X.shape[2])
        y = y.reshape(-1, y.shape[2])
        pred = model(X)
        mu, log_std = torch.chunk(pred,2,dim=-1)
        mse = nn.MSELoss()(y, mu)
        if ctt_std:
            log_p = Normal(loc=mu, scale=hard_std).log_prob(y)
        else:
            log_p = Normal(loc=mu,
                    scale=torch.exp(log_std)).log_prob(y)
        log_p = torch.mean(log_p)
        writer.add_scalar(name+'/RMSE', torch.sqrt(mse), global_step)
        writer.add_scalar(name+'/logP', log_p*y.shape[-1], global_step)


for epoch in Tqdm(range(epochs)):
    train_losses = []
    model.train(True)
    errors = []
    for i, data in enumerate(train_loader):
        global_step += 1
        X, y = data ; X = X.cuda() ; y = y.cuda()
        X = X.reshape(-1, X.shape[2])
        y = y.reshape(-1, y.shape[2])
        pred = model(X)
        mu, log_std = torch.chunk(pred,2,dim=-1)
        SE= (y-mu)**2
        errors.append(SE.detach())
        mse = torch.mean(SE)
        if ctt_std:
            log_p = Normal(loc=mu, scale=hard_std).log_prob(y)
        else:
            log_p = Normal(loc=mu,
                    scale=torch.exp(log_std)).log_prob(y)
        log_p = torch.mean(log_p)
        loss = -log_p
        opt.zero_grad()
        loss.backward()
        opt.step()
        writer.add_scalar('train/RMSE', torch.sqrt(mse), global_step)
        writer.add_scalar('train/logP', log_p*y.shape[-1], global_step)
    hard_std = torch.sqrt(torch.mean(
        torch.cat(errors, dim=0), dim=0, keepdim=True))
    model.train(False)
    evaluate(test_loader, 'test')
    evaluate(old_loader, 'old')
    evaluate(plywood_loader, 'plywood')
    evaluate(plyold_loader, 'plyold')

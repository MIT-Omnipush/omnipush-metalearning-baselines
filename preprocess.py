import json
import os
import torch
import torchvision.transforms
import numpy as np

def check_in_list(st, L):
    for p in range(4):
        if st[2*p:]+st[:2*p] in L: return True
    return False

def omnipush(dir_path='./data', train=True, return_all=False, normalize=True,
        checked_pattern='X_3_not_norm'):
    split_file = os.path.join(dir_path, '../split.json')
    with open(split_file, 'r') as infile:
        split_data = json.load(infile)
    names = []
    X_t = []
    y_t = []
    X_mean = torch.Tensor(np.array(split_data['X_mean']))
    X_std = torch.Tensor(np.array(split_data['X_std']))
    y_mean = torch.Tensor(np.array(split_data['y_mean']))
    y_std = torch.Tensor(np.array(split_data['y_std']))
    for f in os.listdir(dir_path):
        if checked_pattern not in f: continue
        if check_in_list(f[:8] , split_data['train_keys']):
            f_is_train = True
        else:
            assert return_all or check_in_list(f[:8], split_data['test_keys']), f[:8]
            f_is_train = False
        if (not return_all) and f_is_train != train: continue
        X_fil = os.path.join(dir_path, f)
        y_fil = os.path.join(dir_path, f.replace('X', 'y'))
        X = np.load(X_fil)
        y = np.load(y_fil)
        names.append(np.array(f[:8]))
        if X.shape[0] != 250: print(f[:8], X.shape, y.shape)
        if normalize:
            X_t.append((torch.Tensor(X)-X_mean)/X_std)
            y_t.append((torch.Tensor(y)-y_mean)/y_std)
        else:
            X_t.append(torch.Tensor(X))
            y_t.append(torch.Tensor(y))

    return np.vstack(names), torch.stack(X_t), torch.stack(y_t)


def omnipush_collate_fn(batch, shot, meta_train=False):
    context_x = []
    context_y = []
    target_x = []
    target_y = []
    for X, y in batch:
        if meta_train: shift = np.random.randint(250-shot)
        else: shift = 0
        context_x.append(X[shift:shift+shot])
        context_y.append(y[shift:shift+shot])
        target_x.append(torch.cat((X[shift+shot:], X[:shift]), dim=0))
        target_y.append(torch.cat((y[shift+shot:], y[:shift]), dim=0))
    context_x = torch.stack(context_x)
    context_y = torch.stack(context_y)
    target_x = torch.stack(target_x)
    target_y = torch.stack(target_y)
    return context_x, context_y, target_x, target_y

def omnipush_compute_normalization(dir_path='./data', train=True, return_all=False,
        checked_pattern='X_3_not_norm'):
    #############################################################################
    # Only call this function to get normalization statistics in new_split.json #
    #############################################################################
    split_file = os.path.join(dir_path, '../all_omnipush/split.json')
    new_split_file = os.path.join(dir_path, 'new_split.json')
    with open(split_file, 'r') as infile:
        split_data = json.load(infile)
    new_split = {}
    new_split['train_keys'] = []
    new_split['test_keys'] = []
    names = []
    X_t = []
    y_t = []
    X_mean = torch.Tensor(np.array(split_data['X_mean']))
    X_std = torch.Tensor(np.array(split_data['X_std']))
    y_mean = torch.Tensor(np.array(split_data['y_mean']))
    y_std = torch.Tensor(np.array(split_data['y_std']))
    for f in os.listdir(dir_path):
        if checked_pattern not in f: continue
        if check_in_list(f[:8] , split_data['train_keys']):
            new_split['train_keys'].append(f[:8])
            f_is_train = True
        else:
            assert return_all or check_in_list(f[:8], split_data['test_keys']), f[:8]
            new_split['test_keys'].append(f[:8])
            f_is_train = False
        if (not return_all) and f_is_train != train: continue
        X_fil = os.path.join(dir_path, f)
        y_fil = os.path.join(dir_path, f.replace('X', 'y'))
        X = np.load(X_fil)
        y = np.load(y_fil)
        names.append(np.array(f[:8]))
        if X.shape[0] != 250: print(f[:8], X.shape, y.shape)
        X_t.append(torch.Tensor(X[:250]))
        y_t.append(torch.Tensor(y[:250]))
    new_split['train_keys'].sort()
    new_split['test_keys'].sort()
    if train:
        #Compute meta-training mean, std
        stack_X = torch.stack(X_t)
        stack_y = torch.stack(y_t)
        stack_X = stack_X.reshape(-1, stack_X.shape[2])
        stack_y = stack_y.reshape(-1, stack_y.shape[2])
        new_split['X_mean']=torch.mean(stack_X, dim=0).detach().cpu().numpy().tolist()
        new_split['y_mean']=torch.mean(stack_y, dim=0).detach().cpu().numpy().tolist()
        new_split['X_std'] = torch.std(stack_X, dim=0).detach().cpu().numpy().tolist()
        new_split['y_std'] = torch.std(stack_y, dim=0).detach().cpu().numpy().tolist()
        with open(new_split_file, 'w') as outfile:
            json.dump(new_split, outfile)
    return np.vstack(names), torch.stack(X_t), torch.stack(y_t)

def create_omnipush_normalized_files(dir_path='./data', checked_pattern='X_3_not_norm'):
    split_file = os.path.join(dir_path, '../all_omnipush/split.json')
    with open(split_file, 'r') as infile:
        split_data = json.load(infile)
    names = []
    X_t = []
    y_t = []
    X_mean = np.array(split_data['X_mean'])
    X_std = np.array(split_data['X_std'])
    y_mean = np.array(split_data['y_mean'])
    y_std = np.array(split_data['y_std'])
    for f in os.listdir(dir_path):
        if checked_pattern not in f: continue
        X_fil = os.path.join(dir_path, f)
        y_fil = os.path.join(dir_path, f.replace('X', 'y'))
        X = np.load(X_fil)
        y = np.load(y_fil)
        norm_f = '_'.join(f.split('_')[:3]) + '_norm.npy'
        np.save(os.path.join(dir_path, norm_f), (X-X_mean)/X_std)
        np.save(os.path.join(dir_path, norm_f.replace('X','y')), (y-y_mean)/y_std)


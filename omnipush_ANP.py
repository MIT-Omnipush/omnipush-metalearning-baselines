from tqdm import tqdm
from network import LatentModel
from tensorboardX import SummaryWriter
import torchvision
import torch as t
from torch.utils.data import DataLoader
from preprocess import omnipush, omnipush_collate_fn
import numpy as np
import os

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(model, writer, global_step, loader, name):
    for i, data in enumerate(loader):
        context_x, context_y, target_x, target_y = data
        context_x = context_x.cuda()
        context_y = context_y.cuda()
        target_x = target_x.cuda()
        target_y = target_y.cuda()

        # pass through the latent model
        y_pred, kl, loss, mse, log_p = model(context_x, context_y,
                target_x, target_y)
        if mse is not None:
            writer.add_scalar(name+'/RMSE', t.sqrt(mse), global_step)
        writer.add_scalar(name+'/logP', log_p*target_y.shape[-1], global_step)
        writer.add_scalar(name+'/Loss', loss, global_step)
        writer.add_scalar(name+'/kl', kl, global_step)

def main():
    seed = 3 #1,2
    t.manual_seed(seed)
    np.random.seed(seed)
    # Load data
    shot = 50 #10
    train_names, train_X, train_y = omnipush('./data/all_omnipush', train=True)
    test_names, test_X, test_y = omnipush('./data/all_omnipush', train=False)
    train_dataset = t.utils.data.TensorDataset(train_X, train_y)
    test_dataset = t.utils.data.TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=64,
            collate_fn=lambda x :omnipush_collate_fn(x, shot, meta_train=True),
            shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=64,
            collate_fn=lambda x :omnipush_collate_fn(x, shot, meta_train=False),
            shuffle=True, num_workers=32)

    old_names, old_X, old_y = omnipush('./data/old_shapes', return_all=True)
    plywood_names, plywood_X, plywood_y = omnipush('./data/plywood',
            return_all=True)
    plyold_names, plyold_X, plyold_y = omnipush('./data/plyold',
            return_all=True)

    old_dataset = t.utils.data.TensorDataset(old_X, old_y)
    plywood_dataset = t.utils.data.TensorDataset(plywood_X, plywood_y)
    plyold_dataset = t.utils.data.TensorDataset(plyold_X, plyold_y)

    old_loader = DataLoader(old_dataset, batch_size=64,
            collate_fn=lambda x :omnipush_collate_fn(x, shot, meta_train=False),
            shuffle=True, num_workers=32)
    plywood_loader = DataLoader(plywood_dataset, batch_size=64,
            collate_fn=lambda x :omnipush_collate_fn(x, shot, meta_train=False),
            shuffle=True, num_workers=32)
    plyold_loader = DataLoader(plyold_dataset, batch_size=64,
        collate_fn=lambda x :omnipush_collate_fn(x, shot, meta_train=False),
        shuffle=True, num_workers=32)

    boolean_pred = False
    x_dim = 3
    y_dim = 3
    epochs = 5000
    model = LatentModel(num_hidden=128, x_dim=x_dim, y_dim=y_dim,
            boolean_pred=boolean_pred).cuda()

    optim = t.optim.Adam(model.parameters(), lr=3e-4)
    writer = SummaryWriter(comment='-NP'+'-finetunenotintest'+
            '-seed='+str(seed)+'-shot='+str(shot))
    global_step = 0
    ebar = tqdm(range(epochs))
    for epoch in ebar:
        model.train()
        pbar = train_loader
        mses = []
        logps = []
        kls = []
        for i, data in enumerate(pbar):
            global_step += 1
            adjust_learning_rate(optim, global_step)
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()

            # pass through the latent model
            y_pred, kl, loss, mse, log_p = model(
                    context_x, context_y, target_x, target_y)

            # Training step
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            if mse is not None:
                writer.add_scalar('train/RMSE', t.sqrt(mse), global_step)
                if epoch%50==0:
                    print(epoch, loss.item(), t.sqrt(mse).item())
            writer.add_scalar('train/logP', log_p*target_y.shape[-1], global_step)
            writer.add_scalar('train/Loss', loss, global_step)
            writer.add_scalar('train/kl', kl, global_step)

        model.train(False)
        evaluate(model, writer, global_step, test_loader, 'test')
        evaluate(model, writer, global_step, old_loader, 'old')
        evaluate(model, writer, global_step, plywood_loader, 'plywood')
        evaluate(model, writer, global_step, plyold_loader, 'plyold')

        # Save model by each epoch
        if epoch%500==0:
            t.save({'model':model.state_dict(), 'optimizer':optim.state_dict()},
                    os.path.join('./checkpoint','checkpoint_%d.pth.tar' % (epoch+1)))


if __name__ == '__main__':
    main()

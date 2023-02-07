import torch
import numpy as np

import random
import os
import pickle
import argparse


def seed_torch(RANDOM_SEED=123):
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

from model import HOIST
from utils import mse,mae,r2,ccc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Path
    parser.add_argument('--train_dynamic', type=str)
    parser.add_argument('--train_y', type=str)
    parser.add_argument('--val_dynamic', type=str)
    parser.add_argument('--val_y', type=str)
    parser.add_argument('--test_dynamic', type=str)
    parser.add_argument('--test_y', type=str)
    parser.add_argument('--static', type=str, default=None)
    parser.add_argument('--distance', type=str, default=None)
    
    # Model Parameters
    parser.add_argument('--dynamic_dims', nargs='+', type=int)
    parser.add_argument('--static_dims', nargs='+', type=int, default=None)
    parser.add_argument('--distance_dims', type=int, default=None)
    parser.add_argument('--signs', nargs='+', type=int, default=None)
    parser.add_argument('--rnn_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--save_dir', type=str, default='./model/')

    args = parser.parse_args()

    
    device = torch.device(args.device)
    train_dynamic = pickle.load(open(args.train_dynamic, 'rb'))
    train_y = pickle.load(open(args.train_y, 'rb'))

    val_dynamic = pickle.load(open(args.val_dynamic, 'rb'))
    val_y = pickle.load(open(args.val_y, 'rb'))

    test_dynamic = pickle.load(open(args.test_dynamic, 'rb'))
    test_y = pickle.load(open(args.test_y, 'rb'))

    if args.static is not None:
        static = pickle.load(open(args.static, 'rb'))
    if args.distance is not None:
        distance = pickle.load(open(args.distance, 'rb'))

    model = HOIST(args.dynamic_dims, args.static_dims, args.distance_dims, args.rnn_dim, args.output_dim, args.signs, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss(reduction='none')
    
    min_loss = 1e99
    min_epoch = 0
    batch_size = args.batch_size
    N_loc = train_dynamic[0].shape[0]

    for i in range(args.num_epoch):
        epoch_loss = []
        val_loss = []
        model.train()
        for j in range((N_loc//batch_size)+1):
            batch_dynamic = [torch.tensor(train_dynamic[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(train_dynamic))]
            batch_y = torch.tensor(train_y[j*batch_size:(j+1)*batch_size]).float().to(device)
            if args.static is not None:
                batch_static = [torch.tensor(static[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(static))]
            else:
                batch_static = None
            if args.distance is not None:
                batch_dist = torch.tensor(distance[j*batch_size:(j+1)*batch_size,:][:,j*batch_size:(j+1)*batch_size]).float().to(device)
            else:
                batch_dist = None
            optimizer.zero_grad()
            output, _ = model(batch_dynamic, batch_static, batch_dist)
            
            N, T, F = batch_y.shape
            dist = _[0]
            weights = _[1]
            cur_d = torch.cat(batch_dynamic, -1)
            y_p = (weights * cur_d).sum(-1).reshape(N,T,1)*output.detach()
            if args.static is not None or args.distance is not None:
                y_pi = y_p.reshape(N,1,T)
                y_pj = y_p.reshape(1,N,T)
                y_k = ((y_pi * y_pj) * dist.reshape(N,N,1)).sum(1).reshape(N,T,1)
            else:
                y_k = torch.zeros_like(y_p)
            ising_loss = loss_fn(y_p+y_k, batch_y).mean(1).mean()
            
            loss = loss_fn(output, batch_y).mean(-1).mean(1).mean() + ising_loss
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for j in range((N_loc//batch_size)+1):
                batch_dynamic = [torch.tensor(val_dynamic[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(val_dynamic))]
                batch_y = torch.tensor(val_y[j*batch_size:(j+1)*batch_size]).float().to(device)
                if args.static is not None:
                    batch_static = [torch.tensor(static[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(static))]
                else:
                    batch_static = None
                if args.distance is not None:
                    batch_dist = torch.tensor(distance[j*batch_size:(j+1)*batch_size,:][:,j*batch_size:(j+1)*batch_size]).float().to(device)
                else:
                    batch_dist = None
                    
                output, _ = model(batch_dynamic, batch_static, batch_dist)
                loss = loss_fn(output, batch_y).mean(-1).mean(1).mean()
                y_pred += list(output.squeeze().cpu().detach().numpy())
                y_true += list(batch_y.squeeze().cpu().detach().numpy())
                val_loss.append(loss.item())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        cur_mse = mse(y_true, y_pred)
        cur_mae = mae(y_true, y_pred)
        if i % 10 == 0:
            print('Epoch: %d, Train Loss: %.4f, Val Loss: %.4f, MSE: %.2f, MAE: %.2f'%(i, np.mean(epoch_loss), np.mean(val_loss), cur_mse, cur_mae))
        if cur_mae < min_loss:
            min_loss = cur_mae
            min_epoch = i
            torch.save(model.state_dict(), args.save_dir+'hoist.pth')
            
    y_pred = []
    y_true = []
    weight_score = []
    #Load state dict
    model.load_state_dict(torch.load(args.save_dir+'hoist.pth'))
    model.eval()

    for j in range((N_loc//batch_size)+1):
        batch_dynamic = [torch.tensor(test_dynamic[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(test_dynamic))]
        batch_y = torch.tensor(test_y[j*batch_size:(j+1)*batch_size]).float().to(device)
        if args.static is not None:
            batch_static = [torch.tensor(static[k][j*batch_size:(j+1)*batch_size]).float().to(device) for k in range(len(static))]
        else:
            batch_static = None
        if args.distance is not None:
            batch_dist = torch.tensor(distance[j*batch_size:(j+1)*batch_size,:][:,j*batch_size:(j+1)*batch_size]).float().to(device)
        else:
            batch_dist = None
        
        output, _ = model(batch_dynamic, batch_static, batch_dist)
        
        y_pred += list(output.squeeze().cpu().detach().numpy())
        y_true += list(batch_y.squeeze().cpu().detach().numpy())
        weight_score += list(_[1].squeeze().cpu().detach().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    weight_score = np.array(weight_score)

    print('Best Epoch: %d, Test MSE: %.2f, MAE: %.2f, R2: %.2f, CCC: %.2f'%(min_epoch, mse(y_true, y_pred), mae(y_true, y_pred), r2(y_true, y_pred), ccc(y_true, y_pred)))

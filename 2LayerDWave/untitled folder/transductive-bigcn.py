from __future__ import division

import time
import argparse

import torch
import os.path as osp

from torch import tensor
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import BiGCN, BiGCNMLP
from sklearn.decomposition import TruncatedSVD 
import numpy as np

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
            
    mask = (data.train_mask).nonzero(as_tuple=False)
    new_mask_index = np.random.choice(mask.reshape(-1), int(mask.shape[0] / 16))
    new_mask = torch.zeros_like(data.train_mask)
    new_mask[new_mask_index] = 1
    new_mask = new_mask > 0
    
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def run(exp_name, data, model, runs, epochs, lr, weight_decay, early_stopping, device):
    val_losses, accs, durations = [], [], []
    for run_num in range(1):
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        train_acc = 0
        val_acc = 0
        test_acc = 0
        val_loss_history = []
        epoch_count = -1
        weight_ls = []
        for epoch in range(1, epochs + 1):
            # print("epochs:",epoch)
            train(model, optimizer, data)
            weight_1 = model.convs[0].lin.weight.clone().detach()
            weight_2 = model.convs[1].lin.weight.clone().detach()
            weight_1 = torch.flatten(weight_1.sign())
            weight_2 = torch.flatten(weight_2.sign())
            eight = torch.cat((weight_1, weight_2))
            weight_ls.append(eight)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch
            print(eval_info['test_acc'])
            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
                train_acc = eval_info['train_acc']
                val_acc = eval_info['val_acc']
                epoch_count = epoch
                # torch.save(model.state_dict(), 'gcn_cora.pkl')
                # print("model saved!")

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

            if runs == 1:
                print(epoch, "train: {:.4f}, {:.4f}".format(eval_info['train_loss'], eval_info['train_acc']),
                      "val: {:.4f}, {:.4f}".format(eval_info['val_loss'],eval_info['val_acc']),
                      "test: {:.4f}".format(eval_info['test_acc']))

        res = torch.stack(weight_ls)
        torch.save(res, "/Users/max/Documents/UniMac/quantum_gnn/quantum_gnn/Bi-GCN-master/results.pt")
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print("Run: {:d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, epoch_cnt: {:d}".format(run_num+1, train_acc, val_acc, test_acc, epoch_count))

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Experiment:', exp_name)
    print('Val Loss: {:.4f}, Test Accuracy: {:.4f}, std: {:.4f}, Duration: {:.4f}'.
          format(loss.mean().item(), acc.mean().item(), acc.std().item(),
                 duration.mean().item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='0', type=str, help='gpu id')
    parser.add_argument('--exp_name', default='default_exp_name', type=str)
    parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=550)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4
    parser.add_argument('--early_stopping', type=int, default=0)  # 100
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)  # 0.5
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    root = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')
    dataset = Planetoid(root, args.dataset)

    data = dataset[0]
   


    print(data)
    print("Size of train set:", data.train_mask.sum().item())
    print("Size of val set:", data.val_mask.sum().item())
    print("Size of test set:", data.test_mask.sum().item())
    print("Num classes:", dataset.num_classes)
    print("Num features:", dataset.num_features)

    model = BiGCNMLP(dataset.num_features, args.hidden, dataset.num_classes, args.layers, args.dropout)

    run(args.exp_name, data, model, args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, device)


if __name__ == '__main__':
    main()

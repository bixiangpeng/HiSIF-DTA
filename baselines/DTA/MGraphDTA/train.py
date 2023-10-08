# %%
import os
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import get_cindex
from dataset import *
from model import MGraphDTA
from utils import *
# from log.train_logger import TrainLogger
from lifelines.utils import concordance_index


def train(model,criterion,optimizer, train_loader, device):
    for data in train_loader:
        data = data.to(device)
        pred = model(data)

        loss = criterion(pred.view(-1), data.y.view(-1))
        # cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def val(model, criterion, dataloader, device):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            # loss = criterion(pred.view(-1), data.y.view(-1))
            # label = data.y
            total_preds = torch.cat((total_preds,pred.cpu()), 0)
            total_labels = torch.cat((total_labels,data.y.view(-1, 1).cpu()), 0)

    model.train()
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    data_root = 'data'
    fpath = os.path.join(data_root, args.dataset)
    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)

    print(len(train_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=8)

    device = torch.device('cuda:1')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    result_file_name = f'./results/{args.dataset}/result.txt'
    model_file_name = f'./results/{args.dataset}/best_model.pt'
    model.train()

    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    for epoch in range(epochs):
        train(model,criterion,optimizer, train_loader, device)
        G, P = val(model, criterion, test_loader, device)
        ret = [mse(G, P), concordance_index(G, P)]
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_mse = ret[0]
            best_ci = ret[-1]
            best_epoch = epoch
            print(epoch, 'rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)
        else:
            print(epoch, ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)


if __name__ == "__main__":
    main()

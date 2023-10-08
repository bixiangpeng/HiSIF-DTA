# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from metrics import get_cindex, get_rm2
from dataset import *
from model import MGraphDTA
from utils import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex, epoch_r2, pred, label

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = f'./results/{args.dataset}/best_model.pt'

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, train=False)
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device('cuda:1')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(model_path,map_location=device))
    test_loss, test_cindex, test_r2,P,G = val(model, criterion, test_loader, device)

    df = pd.read_csv(f'./data/{args.dataset}/raw/test.csv')
    test_smile = list(df['compound_iso_smiles'])
    test_seq = list(df['target_sequence'])
    G_list = G.tolist()
    P_list = P.tolist()
    predicted_data = {
        'smile': test_smile,
        'sequence': test_seq,
        'label': G_list,
        'predicted value': P_list
    }
    df_pre = pd.DataFrame(predicted_data)
    df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_MGraphDTA_on_{args.dataset} .csv')

    msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (test_loss, test_cindex, test_r2)
    print(msg)


if __name__ == "__main__":
    main()

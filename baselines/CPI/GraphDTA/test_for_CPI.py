import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gat import GATNet
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import argparse
import pickle
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)
    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def main(args):
    dataset = args.dataset
    model_dict_ = {'GCN': GCNNet, 'GAT': GATNet,'GIN':GINConvNet}
    modeling = model_dict_[args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    with open(f"./data/Human/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)


    results = []
    for fold in range(1,6):
        df_test = pd.read_csv(f'./data/Human/test{fold}.csv')
        test_smile,test_seq,test_label = list(df_test['0']), list(df_test['1']),list(df_test['2'])
        test_seq_temp = test_seq
        test_seq = [seq_cat(seq) for seq in test_seq]

        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data = mol_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        model = modeling().to(device)
        model_file_name = f'./results/{args.dataset}/model/pretrained_model_on_{dataset}_fold{fold}.model'
        model.load_state_dict( torch.load(model_file_name) )

        G, P_value, P_label = predicting(model, device, test_loader)
        G_list = G.tolist()
        P_value_list = P_value.tolist()
        P_label_list = P_label.tolist()
        predicted_data = {
            'smile': test_smile,
            'sequence': test_seq_temp,
            'label': G_list,
            'predicted value': P_value_list,
            'predicted label': P_label_list
        }
        df_pre = pd.DataFrame(predicted_data)
        df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_GraphDTA_on_{dataset}_test{fold} .csv')

        tpr, fpr, _ = precision_recall_curve(G, P_value)
        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),recall_score(G, P_label)]
        print('Fold-{}:, auc: {:.5f} | prc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold finished: " "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'GCN')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 4)
    parser.add_argument('--dataset', type = str, default = 'Human')
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)
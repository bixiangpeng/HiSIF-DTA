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


# training function at each epoch
def train(model, device, train_loader,optimizer,loss_fn,epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output= model(data)
        # loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        current_batch_size = len(data.y)
        epoch_loss += loss.item() * current_batch_size
    return epoch_loss / len(train_loader.dataset)

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
        df_train = pd.read_csv(f'./data/Human/train{fold}.csv')
        df_test = pd.read_csv(f'./data/Human/test{fold}.csv')
        train_smile,train_seq,train_label = list(df_train['0']), list(df_train['1']),list(df_train['2'])
        test_smile,test_seq,test_label = list(df_test['0']), list(df_test['1']),list(df_test['2'])

        train_seq = [ seq_cat(seq) for seq in train_seq]
        test_seq = [seq_cat(seq) for seq in test_seq]

        train_dataset = CPIDataset(train_smile, train_seq, train_label, mol_data = mol_data)
        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data = mol_data)

        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        # training the model

        model = modeling().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        best_roc = 0
        model_file_name = f'./results/{args.dataset}/model/' + model_st + '_'+ dataset + '_fold' + str(fold) + '.model'
        for epoch in range(args.epochs):
            train_loss = train(model, device, train_loader, optimizer,loss_fn,epoch + 1)
            G, P ,P_label= predicting(model, device, test_loader)
            valid_roc = roc_auc_score(G, P)
            # print('| AUROC: {:.3f}'.format(valid_roc))
            if valid_roc > best_roc:
                best_roc = valid_roc
                torch.save(model.state_dict(), model_file_name)
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr),recall_score(G, P_label),precision_score(G, P_label)]
                print('epoch:', epoch , "train loss =", train_loss, "test auc =", ret[0], "test recall =", ret[2], "  test precision =", ret[3])
            else:
                ret = [roc_auc_score(G, P), auc(fpr, tpr), recall_score(G, P_label), precision_score(G, P_label)]
                print('epoch:', epoch, "train loss =", train_loss, "test auc =", ret[0], "test recall =", ret[2],
                      "  test precision =", ret[3])
            # reload the best model and test it on valid set again to get other metrics
        model.load_state_dict(torch.load(model_file_name))
        G, P_value, P_label = predicting(model, device, test_loader)

        tpr, fpr, _ = precision_recall_curve(G, P_value)

        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),
                         recall_score(G, P_label)]
        print('Fold-{} valid finished, auc: {:.5f} | prc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]

    print("5-fold cross validation finished. " "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))

    result_file_name = f'./results/{args.dataset}/result/' + model_st +'_'+ dataset + '.txt'  # result

    with open(result_file_name, 'w') as f:
        f.write("auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'GCN' ,help = 'GCN or GAT')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 4)
    parser.add_argument('--dataset', type = str, default = 'Human')
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)
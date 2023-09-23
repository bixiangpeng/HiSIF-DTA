# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
from models.HGCN_for_CPI import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import argparse

def train(model, device, train_loader,optimizer,ppi_adj,ppi_features,proGraph_loader,loss_fn,epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        optimizer.zero_grad()
        output= model(mol_data,pro_data,ppi_adj,ppi_features,proGraph_loader,device)
        loss = loss_fn(output, mol_data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        current_batch_size = len(mol_data.y)
        epoch_loss += loss.item() * current_batch_size

    print('Epoch {}: train_loss: {:.5f} '.format(epoch, epoch_loss / len(train_loader.dataset)), end='')

def predicting(model, device, loader,ppi_adj,ppi_features,proGraph_loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            output = model(mol_data,pro_data,ppi_adj,ppi_features,proGraph_loader,device)

            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, mol_data.y.view(-1, 1).cpu()), 0)
    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()


def main(args):
    dataset = args.dataset
    model_dict_ = {'BUNet': BUNet, 'TDNet': TDNet}
    modeling = model_dict_[args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)

    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)

    proGraph_dataset = GraphDataset(graph=pro_data, index=ppi_index, type='pro')
    if model_st == 'TDNet':
        proGraph_loader = DataLoader(proGraph_dataset, batch_size=int(args.batch/2),shuffle=False,num_workers=args.num_workers)
    else:
        proGraph_loader = DataLoader(proGraph_dataset, batch_size=args.batch, shuffle=False,num_workers=args.num_workers)

    results = []
    for fold in range(1,6):
        df_train = pd.read_csv(f'data/{dataset}/train{fold}.csv')
        df_test = pd.read_csv(f'data/{dataset}/test{fold}.csv')
        train_smile,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
        test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])

        train_dataset = CPIDataset(train_smile, train_seq, train_label, mol_data = mol_data, ppi_index = ppi_index)
        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data = mol_data, ppi_index = ppi_index)

        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        # training the model

        model = modeling().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        best_roc = 0
        best_epoch = -1
        model_file_name = f'results/{dataset}/' + model_st + '_'+ dataset + '_fold' + str(fold) + '.model'
        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, ppi_adj,ppi_features,proGraph_loader,loss_fn,epoch + 1)
            G, P ,_= predicting(model, device, test_loader, ppi_adj,ppi_features,proGraph_loader)
            valid_roc = roc_auc_score(G, P)
            print('| AUROC: {:.5f}'.format(valid_roc))
            if valid_roc > best_roc:
                best_roc = valid_roc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr)]
                test_roc = ret[0]
                test_prc = ret[1]

                print('AUROC improved at epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc),
                      '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc),
                      '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)

            # reload the best model and test it on valid set again to get other metrics
        model.load_state_dict(torch.load(model_file_name))
        G, P_value, P_label = predicting(model, device, test_loader, ppi_adj,ppi_features,proGraph_loader)

        tpr, fpr, _ = precision_recall_curve(G, P_value)

        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),
                         recall_score(G, P_label)]
        print('Fold-{} valid finished, auc: {:.5f} | prc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]

    print("5-fold cross validation finished. " "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))

    result_file_name = f'results/{dataset}/' + model_st +'_'+ dataset + '.txt'  # result

    with open(result_file_name, 'w') as f:
        f.write("auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'BUNet', choices = ['BUNet','TDNet'])
    parser.add_argument('--epochs', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'Human',choices=['Human'])
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)








# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年03月26日
"""
from models.HGCN import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse
import csv


def train(model, device, train_loader,optimizer,ppi_adj,ppi_features,pro_graph,loss_fn,args,epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        optimizer.zero_grad()
        output= model(mol_data,pro_data,ppi_adj,ppi_features,pro_graph)
        loss = loss_fn(output, mol_data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        current_batch_size = len(mol_data.y)
        epoch_loss += loss.item() * current_batch_size
    print('Epoch {}: train_loss: {:.5f} '.format(epoch, epoch_loss / len(train_loader.dataset)), end='')

def predicting(model, device, loader,ppi_adj,ppi_features,pro_graph):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            output = model(mol_data,pro_data,ppi_adj,ppi_features,pro_graph)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, mol_data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def main(args):
    dataset = args.dataset
    modeling = [BUNet,TDNet][args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)

    with open(f'data/{dataset}/PPI/ppi_data_only_domain.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)

    pro_graph = proGraph(pro_data,ppi_index,device)

    results = []
    for fold in range(1,6):
        df_train = pd.read_csv(f'data/{dataset}/5 fold/train{fold}.csv')
        df_test = pd.read_csv(f'data/{dataset}/5 fold/test{fold}.csv')
        # df_train = pd.read_csv(f'data/{dataset}/5 fold/train3.csv')
        # df_test = pd.read_csv(f'data/{dataset}/5 fold/test3.csv')
        train_smile,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
        test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])

        train_dataset = DtaDataset(train_smile, train_seq, train_label, mol_data = mol_data, ppi_index = ppi_index)
        test_dataset = DtaDataset(test_smile, test_seq, test_label, mol_data = mol_data, ppi_index = ppi_index)

        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = f'data/{dataset}/5-fold results/'+ model_st + '_' + 'drop_edge_0.2_fold' + str(fold) + '.model'
        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, ppi_adj,ppi_features,pro_graph,loss_fn,args,epoch + 1)
            G, P = predicting(model, device, test_loader, ppi_adj,ppi_features,pro_graph)
            ret = [mse(G, P), concordance_index(G, P)]
            print('| mse: {:.5f} | ci: {:.5f}'.format(ret[0],ret[1]))
            if ret[0] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                best_epoch = epoch + 1
                best_mse = ret[0]
                best_ci = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset,model_st)
            else:
                print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset,model_st)

        model.load_state_dict(torch.load(model_file_name))
        G, P = predicting(model, device, test_loader, ppi_adj,ppi_features,pro_graph)
        ret = [mse(G, P), concordance_index(G, P)]
        print('Fold-{} valid finished, mse: {:.5f} | ci: {:.5f}'.format(str(fold), ret[0],ret[1] ))
        results.append([ret[0], ret[1]])


    origin_result_file_name = f'data/{dataset}/5-fold results/' + model_st + '_drop_edge_0.2.csv'

    f = open(origin_result_file_name, 'w', newline='')
    writer = csv.writer(f)
    for i in results:
        writer.writerow(i)

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]

    print("5-fold cross validation finished. " "mse:{:.3f}±{:.4f} | ci:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1]))

    result_file_name = f'data/{dataset}/5-fold results/' + model_st + '_drop_edge_0.2.txt'  # result

    with open(result_file_name, 'w') as f:
        f.write("mse:{:.3f}±{:.4f} | ci:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = int, default = 0 ,help = '0: BUNet 1:TDNet')
    parser.add_argument('--epochs', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'davis')
    parser.add_argument('--num_workers', type= int, default = 6)
    # parser.add_argument('--output', type=str, default='ppi_graph.pkl',help = 'The best performance of current model')
    args = parser.parse_args()
    main(args)








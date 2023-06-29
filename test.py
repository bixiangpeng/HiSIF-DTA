# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月10日
"""
from models.HGCN import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse
import torch


def test(model, device, loader,ppi_adj,ppi_features,pro_graph):
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
    modeling = [BUNet, TDNet][args.model]
    model_st = modeling.__name__

    path = f'/home/sgzhang/xiangpeng/HGraphDTA/results/{dataset}/{model_st}.model'

    check_point = torch.load(path)
    model = modeling()
    model.load_state_dict(check_point)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    model =model.to(device)
    df_test = pd.read_csv(f'data/{dataset}/test.csv')
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    with open(f'data/{dataset}/PPI/ppi_data_only_domain.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)

    pro_graph = proGraph(pro_data, ppi_index, device)
    test_dataset = DtaDataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate)#,num_workers=args.num_workers)

    G, P = test(model, device, test_loader, ppi_adj, ppi_features, pro_graph)
    ret = [mse(G, P)]
    print('test_mse:', ret[0], args.dataset, model_st)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = int, default = 0 ,help = '0: BUNet 1:TDNet')
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'davis')
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)

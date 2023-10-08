import os
import sys
import torch
import numpy as np
from utils import *
from gnn import GNNNet
import pandas as pd
import pickle
from lifelines.utils import concordance_index



def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            # data = data.to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def load_model(model_path):
    model = torch.load(model_path)
    return model




if __name__ == '__main__':
    dataset = ['davis', 'kiba'][int(sys.argv[1])]  # dataset selection
    model_st = GNNNet.__name__
    print('dataset:', dataset)

    # cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]  # gpu selection
    cuda_name = 'cuda'
    print('cuda_name:', cuda_name)

    TEST_BATCH_SIZE = 512
    models_dir = 'models'
    results_dir = 'results'

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    model_file_name = 'models/model_' + model_st + '_' + dataset + '.model'
    result_file_name = 'results/result_' + model_st + '_' + dataset + '.txt'

    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    # test = create_dataset_for_test(dataset)
    with open(f'data/{dataset}/mol_data.pkl', 'rb') as file:
        smile_graph = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file:
        target_graph = pickle.load(file)

    df_test = pd.read_csv(f'data/{dataset}/test.csv')
    test_drugs, test_prot_keys, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset, xd=test_drugs, y=test_Y,
                              target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn=collate,num_workers=6)
    #
    Y, P = predicting(model, device, test_loader)
    Y_list = Y.tolist()
    P_list = P.tolist()
    predicted_data = {
        'smile': test_drugs.tolist(),
        'sequence': test_prot_keys.tolist(),
        'label': Y_list,
        'predicted value': P_list
    }
    df_pre = pd.DataFrame(predicted_data)
    df_pre.to_csv(f'./results/{dataset}/predicted_value_of_DGraphDTA_on_{dataset} .csv')
    print(f'MSE={mse(Y,P)} , CI={concordance_index(Y, P)}')


import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from torch_geometric.data import DataLoader
from lifelines.utils import concordance_index


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()



def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba'][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = f'data/{dataset}/processed/train.pt'
    processed_data_file_test = f'data/{dataset}/processed/test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root=f'data/{dataset}', dataset='train')
        test_data = TestbedDataset(root=f'data/{dataset}', dataset='test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=6)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,num_workers=6)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 2000
        best_ci = 0
        best_epoch = -1
        model_file_name = f'./results/{dataset}/model/model_' + model_st + '_' + dataset + '.model'
        result_file_name = f'./results/{dataset}/result/result_' + model_st + '_' + dataset + '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            G, P = predicting(model, device, test_loader)
            ret = [mse(G, P), concordance_index(G, P)]
            if ret[0] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[0]
                best_ci = ret[-1]
                print('epoch:', epoch , 'mse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st,dataset)
            else:
                print('epoch:', epoch , 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,model_st, dataset)
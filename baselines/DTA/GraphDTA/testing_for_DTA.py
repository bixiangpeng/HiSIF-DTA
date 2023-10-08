import pandas as pd
import sys, os
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from torch_geometric.data import DataLoader
from lifelines.utils import concordance_index

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
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

TEST_BATCH_SIZE = 512

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', dataset)
    processed_data_file_train = f'data/{dataset}/processed/train.pt'
    processed_data_file_test = f'data/{dataset}/processed/test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root=f'data/{dataset}', dataset='test')

        # make data PyTorch mini-batch processing ready
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,num_workers=6)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        path = f'results/{dataset}/model/pretrained_model_on_{dataset}.model'
        check_point = torch.load(path, map_location=device)
        model = modeling().to(device)
        model.load_state_dict(check_point)

        G, P = predicting(model, device, test_loader)
        df = pd.read_csv(f'./data/{dataset}/test.csv')
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
        df_pre.to_csv(f'./results/{dataset}/predicted_value_of_GraphDTA_on_{dataset} .csv')

        ret = [mse(G, P), concordance_index(G, P)]
        print( 'test_mse:', ret[0], 'test_ci:', ret[1])
from models.HGCN_for_CPI import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import argparse

def test(model, device, loader, ppi_adj, ppi_features, proGraph_loader):
    '''
    :param model: A model used to predict the binding affinity.
    :param device: Device for loading models and data.
    :param loader: Dataloader used to batch the input data.
    :param ppi_adj: The adjacency matrix of a Protein-Protein Interaction (PPI) graph.
    :param ppi_features: The feature matrix of a Protein-Protein Interaction (PPI) graph.
    :param pro_graph_loader: An auxiliary dataloader for batching all proteins in dataset.
    :return: Ground truth and predicted values
    '''
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
            predicted_values = torch.sigmoid(output)
            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0) #predicted values
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0) # predicted labels
            total_true_labels = torch.cat((total_true_labels, mol_data.y.view(-1, 1).cpu()), 0) #ground truth
    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()


def main(args):
    dataset = args.dataset
    model_dict_ = {'BUNet': BUNet, 'TDNet': TDNet} # Two model architecture we proposed.
    modeling = model_dict_[args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file) # Reading drug graph data from the serialized file.
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2) # Reading protein graph data from the serialized file.
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)# Reading PPI graph data from the serialized file.
    # 'ppi_index' is a dictionary that records the order of protein nodes in PPI, where the keys are protein sequences and the values are node indices

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device) # Tensorization and sparsification of the adjacency matrix of the PPI graph.
    ppi_features = torch.Tensor(ppi_features).to(device) # Tensorization of the feature matrix of the PPI graph.

    proGraph_dataset = GraphDataset(graph=pro_data, index=ppi_index)

    if model_st == 'TDNet':
        # An auxiliary dataloader for batching all proteins in the current dataset to reduce memory burden.
        proGraph_loader = DataLoader(proGraph_dataset, batch_size=int(args.batch/2),shuffle=False,num_workers=args.num_workers)
    else:
        # An auxiliary dataloader for batching all proteins in the current dataset to reduce memory burden.
        proGraph_loader = DataLoader(proGraph_dataset, batch_size=args.batch, shuffle=False,num_workers=args.num_workers)

    results = []
    for fold in range(1,6): # Test the performance of the model trained with five-fold cross-training sequentially.
        df_test = pd.read_csv(f'data/{dataset}/test{fold}.csv') #Reading test data for current fold.
        test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])
        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data = mol_data, ppi_index = ppi_index)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        model = modeling().to(device)
        model_file_name = f'results/{dataset}/pretrained_{args.model}_fold{fold}.model'
        model.load_state_dict(torch.load(model_file_name)) # Loading pre-trained model parameters into the current model.
        G, P_value, P_label = test(model, device, test_loader, ppi_adj,ppi_features,proGraph_loader)

        G_list = G.tolist()
        P_value_list = P_value.tolist()
        P_label_list = P_label.tolist()
        predicted_data = {
            'smile': test_smile,
            'sequence': test_seq,
            'label': G_list,
            'predicted value': P_value_list,
            'predicted label': P_label_list
        }
        df_pre = pd.DataFrame(predicted_data)
        df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_{args.model}_on_{args.dataset}_test{fold} .csv')
        tpr, fpr, _ = precision_recall_curve(G, P_value)
        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),recall_score(G, P_label)]
        print('Fold-{}: prc: {:.5f} | auc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])

    valid_results = np.array(results)
    #Calculating the average performance of all models.
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold results:" "prc:{:.3f}±{:.4f} | auc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))


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

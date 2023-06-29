# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年03月26日
"""
from torch_geometric.data import Batch,InMemoryDataset
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric import data as DATA
import torch
import pickle
from torch_geometric.loader import DataLoader

def collate(data):
    drug_batch = Batch.from_data_list( [item[0] for item in data])
    seq_batch = Batch.from_data_list([item[1] for item in data])
    return drug_batch,seq_batch

def collate2(data):
    seq_batch = Batch.from_data_list([item for item in data])
    return seq_batch


def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k


class DtaDataset(Dataset):
    def __init__(self,smile_list, seq_list, label_list,mol_data = None,ppi_index = None):
        super(DtaDataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.ppi_index = ppi_index

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels =self.label_list[index]

        drug_size, drug_features, drug_edge_index = self.smile_graph[smile]
        seq_size = len(seq)
        seq_index =self.ppi_index[seq]

        GCNData_smile = DATA.Data(x=torch.Tensor(drug_features), edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0), y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_size]))
        GCNData_seq = DATA.Data(y=torch.FloatTensor([labels]),seq_num =torch.LongTensor([seq_index]))
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        return GCNData_smile, GCNData_seq

class CPIDataset(Dataset):
    def __init__(self,smile_list, seq_list, label_list,mol_data = None,ppi_index = None):
        super(CPIDataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.ppi_index = ppi_index

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels =self.label_list[index]

        drug_size, drug_features, drug_edge_index = self.smile_graph[smile]
        seq_size = len(seq)
        seq_index =self.ppi_index[seq]

        GCNData_smile = DATA.Data(x=torch.Tensor(drug_features), edge_index=torch.LongTensor(drug_edge_index), y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_size]))
        GCNData_seq = DATA.Data(y=torch.FloatTensor([labels]),seq_num =torch.LongTensor([seq_index]))
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        return GCNData_smile, GCNData_seq


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', graph = None,index = None ,type=None):
        super(GraphDataset, self).__init__(root)
        self.type = type
        self.index = index
        self.process(graph,index)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graph,index):
        data_list = []
        count = 0
        for key in index:
            size, features, edge_index = graph[key]
            # if self.type == 'pro':
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index),graph_num = torch.LongTensor([count]))
            # else:
            #     GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1,0),
            #                         edge_len=torch.LongTensor([len(edge_index[0])]))
            GCNData.__setitem__('c_size', torch.LongTensor([size]))
            count += 1
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def proGraph(graph_data, index, device):
    proGraph_dataset = GraphDataset(graph=graph_data, index=index ,type = 'pro')
    proGraph_loader = DataLoader(proGraph_dataset, batch_size=len(graph_data), shuffle=False)
    pro_graph = None
    for batchid, batch in enumerate(proGraph_loader):
        pro_graph = batch.x.to(device),batch.edge_index.to(device),batch.graph_num.to(device),batch.batch.to(device)
    return pro_graph

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def mse_print(y,f):
    mse = ((y - f)**2)
    return mse

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook) #对需要的层注册hook,其主要的作用是在不改变torch网络得到情况下获取某一层的输出
        self.target_feat = None

    def save_hook(self, md, fin, fout): #hook函数，定义一个hook函数需要定义三个参数，这三个参数均是系统传过来的，不能更改。也就是说有了这个hook,我们可以拿到该module的输入和输出结果
        self.target_feat = fout

    def __call__(self, mol_data,pro_data,ppi_adj,ppi_features,proGraph):
        self.model.eval()
        _,_,_,p_batch = proGraph
        output = self.model(mol_data,pro_data,ppi_adj,ppi_features,proGraph).view(-1)
        mask = torch.eq(p_batch, pro_data.seq_num)
        indexes = torch.nonzero(mask, as_tuple=False).view(-1)
        # new_target_feat = self.target_feat[indexes]
        grad = torch.autograd.grad(output, self.target_feat)[0]
        grad = grad[indexes]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat[indexes] * channel_weight
        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)
        return output.detach().cpu().numpy(), cam
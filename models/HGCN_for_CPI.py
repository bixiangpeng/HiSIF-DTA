# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv,global_mean_pool as gep


class BUNet(nn.Module): # Bottom-Up strategy
    def __init__(self,n_output = 1,output_dim=128,num_features_xd = 78,num_features_pro = 33):
        super(BUNet, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        # GCN encoder used for extracting drug features.
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # GCN encoder used for extracting protein features.
        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GCNConv(output_dim, output_dim)
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        # GCN encoder used for extracting PPI features.
        self.ppiGconv1 = GCNConv(output_dim, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim,1024)
        self.ppiFC2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self,mol_data,pro_data,ppi_edge,ppi_features,proGraph_loader,device):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num

        # Extracting drug features
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        # Extracting protein features in batches from protein graphs to alleviate the burden on memory.
        embedding = []
        for batchid,graphData in enumerate(proGraph_loader):
            graphData = graphData.to(device)
            p_x,p_edge_index,p_batch = graphData.x,graphData.edge_index,graphData.batch
            p_x = self.relu(self.proGconv1(p_x, p_edge_index))
            p_x = self.relu(self.proGconv2(p_x, p_edge_index))
            p_x = self.relu(self.proGconv3(p_x, p_edge_index))
            p_x = gep(p_x, p_batch)
            p_x = self.dropout2(self.relu(self.proFC1(p_x)))
            p_x = self.dropout2(self.proFC2(p_x))
            embedding.append(p_x)
        p_x = torch.vstack(embedding)

        # DropEdge
        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.2, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training)
        # Extracting protein functional features from PPI graph.
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(p_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_x = self.dropout1(self.ppiFC2(ppi_x))
        ppi_x = ppi_x[seq_num]

        #combination
        xc = torch.cat((x, ppi_x), 1)
        # classifier
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        out = self.out(xc)
        return out


class TDNet(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, num_features_pro=33,num_features_ppi = 1442):
        super(TDNet, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        # GCN encoder used for extracting drug features.
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # GCN encoder used for extracting protein features.
        self.proGconv1 = GCNConv(num_features_pro,64)
        self.proGconv2 = GCNConv(output_dim,output_dim)
        self.proGconv3 = GCNConv(output_dim,output_dim)
        self.proFC1 = nn.Linear(output_dim,1024)
        self.proFC2 = nn.Linear(1024,output_dim)

        # GCN encoder used for extracting PPI features.
        self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, 64)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # # classifier
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, proGraph_loader,device):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num

        # Extracting drug features
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        # DropEdge
        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.2, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training)
        # Extracting protein functional features from PPI graph.
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(ppi_features, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_x = self.dropout1(self.ppiFC2(ppi_x))

        # Extracting protein features in batches from protein graphs to alleviate the burden on memory.
        embedding = []
        for batchid, graphData in enumerate(proGraph_loader):
            graphData = graphData.to(device)
            p_x, p_edge_index,p_graph_num, p_batch = graphData.x, graphData.edge_index, graphData.graph_num,graphData.batch
            p_x = self.relu(self.proGconv1(p_x, p_edge_index))
            ppi_xx = ppi_x[p_graph_num][p_batch]
            p_x = torch.cat((torch.add(p_x, ppi_xx), torch.sub(p_x, ppi_xx)), -1) #feature combination
            p_x = self.relu(self.proGconv2(p_x, p_edge_index))
            p_x = self.relu(self.proGconv3(p_x, p_edge_index))
            p_x = gep(p_x, p_batch)
            p_x = self.dropout2(self.relu(self.proFC1(p_x)))
            p_x = self.dropout2(self.proFC2(p_x))
            embedding.append(p_x)
        p_x = torch.vstack(embedding)
        p_x = p_x[seq_num]

        # combination
        xc = torch.cat((x, p_x), 1)
        # classifier
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        out = self.out(xc)

        return out
# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年03月26日
"""
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool as gep

class HSM(nn.Module):
    def __init__(self,n_output = 1,output_dim=128,num_features_xd = 78,num_features_pro = 33,num_features_ppi = 1442):
        super(HSM, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GCNConv(output_dim, output_dim)
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, 128)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self,mol_data,pro_data,ppi_edge,ppi_features,pro_graph):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num

        p_x,p_edge_index,p_edge_len,p_batch = pro_graph

        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        p_x = self.dropout2(self.proFC2(p_x))
        p_x = p_x[seq_num]

        #combination
        xc = torch.cat((x, p_x), 1)
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        embedding = xc
        out = self.out(xc)

        return out#,embedding


class HFS(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, num_features_pro=33,num_features_ppi = 1442):
        super(HFS, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        self.proGconv1 = GCNConv(num_features_pro,output_dim)
        self.proGconv2 = GCNConv(output_dim,output_dim)
        self.proGconv3 = GCNConv(output_dim,output_dim)
        self.proFC1 = nn.Linear(output_dim,1024)
        self.proFC2 = nn.Linear(1024,output_dim)

        self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, 128)


        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        p_x, p_edge_index, p_edge_len, p_batch = pro_graph

        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.2, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training)
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(ppi_features, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_x = self.dropout1(self.ppiFC2(ppi_x))
        ppi_x = ppi_x[seq_num]

        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        p_x = self.dropout2(self.proFC2(p_x))
        p_x = p_x[seq_num]

        f_p_x = torch.cat((p_x, ppi_x), 1)

        # combination
        xc = torch.cat((x, f_p_x), 1)
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        embedding = xc
        out = self.out(xc)

        return out  # ,embedding



class LSM(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, num_features_pro=33,num_features_ppi = 1442):
        super(LSM, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        self.proGconv1 = GCNConv(num_features_pro,64)
        self.proGconv2 = GCNConv(output_dim,output_dim)
        self.proGconv3 = GCNConv(output_dim,output_dim)
        self.proFC1 = nn.Linear(output_dim,1024)
        self.proFC2 = nn.Linear(1024,output_dim)

        self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, 128)


        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.2, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training)
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(ppi_features, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_x = self.dropout1(self.ppiFC2(ppi_x))
        ppi_x = ppi_x[seq_num]

        # combination
        xc = torch.cat((x, ppi_x), 1)
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        embedding = xc
        out = self.out(xc)

        return out#,embedding
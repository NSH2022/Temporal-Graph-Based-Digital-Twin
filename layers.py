import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch.nn import Conv1d, MaxPool1d, Flatten, Linear, ConvTranspose1d
import torch.nn.functional as F
from attention import MultiHeadAttention

class Residual(torch.nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        return tensors[0] + self.dropout(self.sublayer(*tensors))
    
    
class CNN_1(torch.nn.Module):
    def __init__(self, num_nodes, out_dim, kernel_size):
        super(CNN_1, self).__init__()
        self.conv1 = Conv1d(num_nodes , num_nodes, kernel_size, padding = 1)
        self.conv2 = Conv1d(out_dim, 1, kernel_size, padding='same')
        
    def forward(self, x):
        x = torch.stack((x[1,:],x[5,:]), dim=1) # phase2&6
        num_nodes, num_features, num_steps = x.size()
        x_reshaped = x.view(num_features, num_nodes , num_steps)
        x = self.conv1(x_reshaped)
        x = torch.relu(x) 
        x = self.conv2(x.permute(0,2,1))
        x = x.view(num_nodes, num_features)
        x = torch.relu(x)
        return x
    
    
class GNN(torch.nn.Module):
    def __init__(self, seq_length, num_nodes, hidden_size, num_heads = 2, dropout = 0.1):
        super(GNN, self).__init__()
        self.gat1 = GATConv(num_nodes, hidden_size, heads=4, concat=True)
        self.gat2 = GATConv(hidden_size * 4, hidden_size, heads=1, concat=False) 
        self.linear= Linear(hidden_size, num_nodes)
        att_size = max(seq_length // num_heads, 1)
        self.self_attention = Residual(
            MultiHeadAttention(num_heads, num_nodes, att_size),
            dimension = num_nodes,
            dropout = dropout,
            )

        
    def forward(self,  x, edge_index, edge_attr):
        x = x.unsqueeze(0) 
        x = self.self_attention(x, x, x).squeeze()
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x) 
        x = self.linear(x) 
        x = F.relu(x) 
        return x
    





class DTGNN_1(torch.nn.Module):
    def __init__(self, num_nodes, node_in_dim, edge_in_dim, hidden_dim, out_dim, kernel_size):
        super(DTGNN_1, self).__init__()
        self.cnn1 = CNN_1(num_nodes, out_dim, kernel_size)
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.edge_mlp = torch.nn.Sequential(
            Linear(edge_in_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.fc_fwd = Linear(hidden_dim*2, out_dim)
        self.fc_rev = Linear(hidden_dim*2, out_dim)


    def forward(self, data):
        x, x_tmp, edge_index, edge_attr = data.x['x_feat'], data.x['x_feat_tmp'], data.edge_index, data.edge_attr

        x_tmp = self.cnn1(x_tmp)
        x = torch.cat((x, x_tmp), dim=1)
        x = self.gat1(x, edge_index, edge_attr) 
        x = F.relu(x)
        
        # Update edge attributes (optional dynamic edge processing)
        edge_attr_new = self.edge_mlp(edge_attr)
        x = self.gat2(x, edge_index, edge_attr_new)
        x = F.relu(x)
        
        # Aggregate node features (global) 
        graph_node_features_mean = torch.mean(x, dim=0, keepdim=True)
        
        # Aggregate edge features (direction-wise)
        edge_attr_fwd, edge_attr_rev = edge_attr_new[::2], edge_attr_new[1::2]
        graph_edge_features_mean_fwd = torch.mean(edge_attr_fwd, dim=0, keepdim=True)
        graph_edge_features_mean_rev = torch.mean(edge_attr_rev, dim=0, keepdim=True)
        
        graph_features_combined_fwd = torch.cat([graph_node_features_mean, graph_edge_features_mean_fwd], dim=-1)
        graph_features_combined_rev = torch.cat([graph_node_features_mean, graph_edge_features_mean_rev], dim=-1)
        
        out_fwd = self.fc_fwd(graph_features_combined_fwd)
        out_rev = self.fc_rev(graph_features_combined_rev)
        out_fwd = torch.relu(out_fwd) 
        out_rev = torch.relu(out_rev)

        return out_fwd.double(), out_rev.double()

    
    
    
class CNN_2(nn.Module):
    def __init__(self, seq_length, kernel_size = 3, pooling_size = 2):
        super(CNN_2, self).__init__()
        dims = [32, 64, 128, 256]

        self.Encoder = nn.Sequential(
             Conv1d(in_channels = 4, out_channels = dims[0] , kernel_size = kernel_size, bias=True),
             MaxPool1d(kernel_size = pooling_size),
             Conv1d(in_channels = dims[0], out_channels = dims[1] , kernel_size = kernel_size, bias=True),
             Flatten(1,2),
             Linear(dims[2]  , dims[3], bias = True),
             Linear(dims[3] , seq_length, bias = True),
             torch.nn.ReLU())
        

    def forward(self, x):
        x = self.Encoder(x)               
        return x
    
    
    
    
class DTGNN_2(torch.nn.Module):
    def __init__(self, num_nodes, node_in_dim, edge_in_dim, hidden_dim, out_dim, kernel_size):
        super(DTGNN_2, self).__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.cnn1 = CNN_1(self.num_nodes, self.out_dim, kernel_size)
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.edge_mlp = torch.nn.Sequential(
            Linear(edge_in_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.deconv1 = ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=10)
        self.deconv2 = ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=10)
        self.deconv3 = ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=10)
        self.cnn2 = CNN_2(self.out_dim)
        self.fc_fwd = Linear(hidden_dim*2, self.out_dim)
        self.fc_rev = Linear(hidden_dim*2, self.out_dim)


    def forward(self, data):
        x, x_tmp, edge_index, edge_attr = data.x['x_feat'], data.x['x_feat_tmp'], data.edge_index, data.edge_attr

        x_ = self.cnn1(x_tmp)
        x = torch.cat((x, x_), dim=1)
        x = self.gat1(x, edge_index, edge_attr) 
        x = F.relu(x)
        
        # Update edge attributes (optional dynamic edge processing)
        edge_attr_new = self.edge_mlp(edge_attr)
        x = self.gat2(x, edge_index, edge_attr_new) 
        x = F.relu(x)
        
        # Retrive phase dimension in node features
        x = self.deconv1(x.unsqueeze(-1)).permute(1,0,2) 
        x = F.avg_pool2d(x, (self.num_nodes,1)).squeeze() 

        # Aggregate edge features (direction-wise)
        e_fwd, e_rev = edge_attr_new[::2], edge_attr_new[1::2]
        
        e_fwd = self.deconv2(e_fwd.unsqueeze(-1)).permute(1,0,2) 
        e_rev = self.deconv3(e_rev.unsqueeze(-1)).permute(1,0,2) 

        e_fwd = F.avg_pool2d(e_fwd, (self.num_nodes,1)).squeeze() 
        e_rev = F.avg_pool2d(e_rev, (self.num_nodes,1)).squeeze() 
        
        # Time series convolution with inf waveforms
        inf = x_tmp.flatten(0,1)
 
        cnn_x = torch.stack([x, inf, e_fwd, e_rev], dim=1) 
        out = self.cnn2(cnn_x).view(self.num_nodes,self.num_nodes,self.out_dim)
        

        return out.double()
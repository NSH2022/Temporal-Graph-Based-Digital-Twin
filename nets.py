import torch
from layers import *

class DTGAT(torch.nn.Module):
    def __init__(self, num_nodes=8, node_in_dim=16, edge_in_dim=19, hidden_dim=64, hidden_size=32, out_dim=10, kernel_size=3):
        super(DTGAT, self).__init__()
        self.Submodule_inflow = GNN(out_dim, num_nodes, hidden_size)
        self.Submodule_travel_time = DTGNN_1(num_nodes, node_in_dim, edge_in_dim, hidden_dim, out_dim, kernel_size)  
        self.Submodule_queue_length = DTGNN_2(num_nodes, node_in_dim, edge_in_dim, hidden_dim, out_dim, kernel_size)
        self.Submodule_waiting_time = DTGNN_2(num_nodes, node_in_dim, edge_in_dim, hidden_dim, out_dim, kernel_size)

    def forward(self, data): 
        out_tt_fwd, out_tt_rev = self.Submodule_travel_time(data)
        out_ql = self.Submodule_queue_length(data)
        out_wt = self.Submodule_waiting_time(data)

        return out_tt_fwd, out_tt_rev, out_ql, out_wt


    
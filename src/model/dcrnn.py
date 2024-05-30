import torch
from torch import nn
from src.torch_geometric_temporal.dcrnn import DCRNN as DCRNN_TG
import torch.nn.functional as F


class DCRNN(nn.Module):
    def __init__(self, node_features, out_channels, K):
        """
        Args:
            node_features (int): Number of input features.
            out_channels (int): Number of output features.
            K (int): Filter size :math:`K`.
        """

        super(DCRNN, self).__init__()
        self.recurrent = DCRNN_TG(node_features, out_channels, K)
        self.linear = torch.nn.Linear(out_channels, node_features)

    def forward(self, x, edge_index, edge_weight, h=None):
        """
        Args:
            x (Tensor): The input features [num_nodes, num_features, P]
            edge_index (LongTensor): The edge indices [2, num_edges]
            edge_weight (Tensor): The edge weights [num_edges]
            h (Tensor, optional): The hidden state [num_nodes, out_channels]
        """
        
        if len(x.shape) == 3:
            x.unsqueeze(0)
    
        batch_size, num_nodes, num_features, P = x.shape
        # num_nodes, num_features, P = x.shape
        out_seq = []

        x = x.permute(3, 1, 0, 2)  # [P, num_nodes, batch_size, num_features]

        for t in range(P):
            h = self.recurrent(x[t, :, :, :], edge_index, edge_weight, h)
            h = F.relu(h)
            out = self.linear(h)
            out_seq.append(out)

        out_seq = torch.stack(out_seq, dim=0)  # [P, num_nodes, batch_size, num_features]
        out_seq = out_seq.permute(2, 1, 3, 0) # [batch_size, num_nodes, num_features, P]
        return out_seq, h

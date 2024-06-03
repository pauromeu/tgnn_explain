import torch
from torch import nn
from src.torch_geometric_temporal.dcrnn import DCRNN as DCRNN_TG
import torch.nn.functional as F


class DCRNN(nn.Module):
    def __init__(self, node_features, out_channels, K, stash_adj_matrix=False):
        """
        Args:
            node_features (int): Number of input features.
            out_channels (int): Number of output features.
            K (int): Filter size :math:`K`.
        """

        super(DCRNN, self).__init__()
        self.recurrent_encoder = DCRNN_TG(node_features, out_channels, K, stash_adj_matrix=stash_adj_matrix)
        self.recurrent_decoder = DCRNN_TG(node_features, out_channels, K, stash_adj_matrix=stash_adj_matrix)
        self.linear = torch.nn.Linear(out_channels, node_features)

    def forward(self, x, edge_index, edge_weight, training_target = None, time_steps = None, h=None):
        """
        Args:
            x (Tensor): The input features [num_nodes, num_features, P]
            edge_index (LongTensor): The edge indices [2, num_edges]
            edge_weight (Tensor): The edge weights [num_edges]
            h (Tensor, optional): The hidden state [num_nodes, out_channels]
        """
        untimed = False
        if time_steps is not None:
            x = x.reshape(*x.shape[:-1], -1, time_steps)
            untimed = True

        unbatched = False
        if len(x.shape) == 3:
            unbatched = True
            x = x.unsqueeze(0)
    
        batch_size, num_nodes, num_features, P = x.shape
        # num_nodes, num_features, P = x.shape


        for t in range(P):
            h = self.recurrent_encoder(x[:, :, :, t], edge_index, edge_weight, h)
            h = F.relu(h)
        
        out_seq = []
        out = x[:,:,:,P-1]
        for t in range(P):
            h = self.recurrent_decoder(out, edge_index, edge_weight, h)
            out = self.linear(h) + out
            out_seq.append(out)

        out_seq = torch.stack(out_seq, dim=3)  # [batch_size, num_nodes, num_features, P]
        if unbatched:
            out_seq = out_seq.squeeze(0)
        if untimed:
            out_seq = out_seq.reshape(*out_seq.shape[:-2], -1)
        return out_seq

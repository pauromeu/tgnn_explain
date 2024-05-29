from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent.dcrnn import DCRNN as DCRNN_TG


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
        num_nodes, num_features, P = x.shape
        out_seq = []

        for t in range(P):
            h = self.recurrent(x[:, :, t], edge_index, edge_weight, h)
            h = F.relu(h)
            out = self.linear(h)
            out_seq.append(out)

        out_seq = torch.stack(out_seq, dim=0)  # [P, num_nodes, num_features]
        out_seq = out_seq.permute(1, 2, 0)  # [num_nodes, num_features, P]
        return out_seq, h

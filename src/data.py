import torch


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, temporal_data):
        self.temporal_data = temporal_data

    def __len__(self):
        return self.temporal_data.snapshot_count

    def __getitem__(self, idx):
        snapshot = self.temporal_data[idx]
        return snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y


def collate_fn(batch):
    x, edge_index, edge_weight, y = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(edge_index),
        torch.stack(edge_weight),
        torch.stack(y),
    )

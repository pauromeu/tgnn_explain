import numpy as np
import torch
import torch.nn.functional as F

from src.torch_geometric_temporal.pems_bay import PemsBayDatasetLoader
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

loader = PemsBayDatasetLoader()
dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

model = DCRNN(node_features=2, out_channels=32, K=2)

sample = train_dataset[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(100):  # Adjust the number of epochs as necessary
    for s, snapshot in enumerate(train_dataset):
        x = torch.tensor(snapshot.x, dtype=torch.float32)
        edge_index = torch.tensor(snapshot.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(snapshot.edge_attr, dtype=torch.float32)
        y = torch.tensor(snapshot.y, dtype=torch.float32)

        optimizer.zero_grad()
        h = None  # Initialize hidden state

        y_hat, h = model(x, edge_index, edge_attr, h)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        # print snapshot progress
        print(
            f"Epoch {epoch+1}, Snapshot {np.round((s+1) * 100 / train_dataset.snapshot_count, 4)} %, Loss: {loss.item()}",
            end="\r",
        )

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

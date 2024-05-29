import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data import TemporalDataset, collate_fn
from src.torch_geometric_temporal.pems_bay import PemsBayDatasetLoader
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# =====================================
# =====================================
# Train a DCRNN model on the Pems-Bay dataset
# =====================================
# =====================================


# =====================================
# Hyperparameters
# =====================================
# Model
node_features = 2
out_channels = 32
K = 2

# Data
proportion_original_dataset = 0.01  # Use 1% of the original dataset to debug

# Training
num_workers = 16
batch_size = 1
resume_training = True

# Paths
logs_path = "runs/logs"
checkpoint_path = "runs/model_checkpoint.pth"


# =====================================
# Data
# =====================================

# Uncomment the following lines to download the dataset if SSL certificate verification fails
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
loader = PemsBayDatasetLoader()
dataset = loader.get_dataset()

_, dataset = temporal_signal_split(dataset, train_ratio=1 - proportion_original_dataset)
train_dataset_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
train_dataset, val_dataset = temporal_signal_split(train_dataset_, train_ratio=0.9)

print(f"Train dataset length: {train_dataset.snapshot_count}")
print(f"Validation dataset length: {val_dataset.snapshot_count}")
print(f"Test dataset length: {test_dataset.snapshot_count}")

train_loader = DataLoader(
    TemporalDataset(train_dataset),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    TemporalDataset(val_dataset),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    TemporalDataset(test_dataset),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

# =====================================
# Model
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DCRNN(node_features=node_features, out_channels=out_channels, K=K).to(device)


# =====================================
# Training
# =====================================

start_epoch = 0
best_val_loss = float("inf")
resume_training = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()
writer = SummaryWriter(logs_path)

if resume_training and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(
        f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}"
    )


model.train()
for epoch in range(start_epoch, 100):
    running_loss = 0.0
    for s, data in enumerate(train_loader):
        x, edge_index, edge_weight, y = data
        x = x.squeeze(0).to(device)  # Remove batch dimension and move to device
        edge_index = edge_index.squeeze(0).to(device)
        edge_weight = edge_weight.squeeze(0).to(device)
        y = y.squeeze(0).to(device)

        optimizer.zero_grad()
        h = None  # Initialize hidden state

        y_hat, h = model(x, edge_index, edge_weight, h)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(
            f"Epoch {epoch+1}, Snapshot {np.round((s+1) * 100 / train_dataset.snapshot_count, 4)} %",
            end="\r",
        )
    writer.add_scalar("Training Loss", running_loss / len(train_loader), epoch)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for s, data in enumerate(test_loader):
            x, edge_index, edge_weight, y = data
            x = x.squeeze(0).to(device)
            edge_index = edge_index.squeeze(0).to(device)
            edge_weight = edge_weight.squeeze(0).to(device)
            y = y.squeeze(0).to(device)

            h = None
            y_hat, h = model(x, edge_index, edge_weight, h)
            loss = loss_function(y_hat, y)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    writer.add_scalar("Validation Loss", val_loss, epoch)

    print(
        f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}"
    )

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )
        print(f"Saved new best model with validation loss {best_val_loss:.4f}")

    model.train()

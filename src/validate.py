import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data import get_loaders, get_pems_bay_dataset, get_metr_la_dataset
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# import batch from torch_geometric
from torch_geometric.data import Batch

# =====================================
# =====================================
# Validate a DCRNN model on the METR-LA dataset
# =====================================
# =====================================


# =====================================
# Hyperparameters
# =====================================
# Model
node_features = 2
out_channels = 32
K = 3

# Data
proportion_original_dataset = 0.1  # Use 1% of the original dataset to debug

# Training
num_workers = 1
batch_size = 32
resume_training = True
tau_sampling = 3000  # should be 3000 for full training

# Paths
logs_path = "runs/logs"
checkpoint_path = "runs/model_checkpoint_dcrnn.pth"


# =====================================
# Data
# =====================================
if __name__ == "__main__":
    dataset = get_metr_la_dataset()

    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        val_ratio=0,
        test_ratio=0,
        proportion_original_dataset=proportion_original_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # =====================================
    # Model
    # =====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DCRNN(node_features=node_features, out_channels=out_channels, K=K).to(
        device
    )

    # =====================================
    # Training
    # =====================================

    start_epoch = 0
    iterations = 0
    best_val_loss = float("inf")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter(logs_path)

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        iterations = checkpoint["iterations"]
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}"
        )

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for s, data in enumerate(train_loader):
            x, edge_index, edge_weight, y = data
            x = x.to(device)  # Remove batch dimension and move to device
            edge_index = edge_index[0].to(device)
            edge_weight = edge_weight[0].to(device)
            y = y.to(device)

            h = None
            y_hat = model(x, edge_index, edge_weight, h)
            loss = loss_function(y_hat, y)
            val_loss += loss.item()

            print(f"\r {s}/{len(train_loader)}", end="")

    print()

    val_loss /= len(train_loader)

    print("Validation Loss", val_loss)

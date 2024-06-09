import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data import get_loaders, get_pems_bay_dataset, get_metr_la_dataset
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# import batch from torch_geometric
from torch_geometric.data import Batch


class Metrics:
    def __init__(self, dataset_type, raw_data_dir=os.path.join(os.getcwd(), "data")):
        self.compute_std(dataset_type, raw_data_dir)
        self.reset()
    def update(self, y, y_hat):
        # y and y_hat are torch tensors of shape (batch_size, num_nodes, num_features, num_timesteps)
        y = self.destandardize(y)
        y_hat = self.destandardize(y_hat)
        self.num_samples += y.shape[0]*y.shape[1]
        self.non_averaged_mse += torch.sum((y - y_hat) ** 2, dim=(0, 1))
        self.non_averaged_mae += torch.sum(torch.abs(y - y_hat), dim=(0, 1))
        self.non_averaged_mape += torch.sum(torch.abs((y - y_hat) / y), dim=(0, 1))
    def destandardize(self, y):
        return y * self.stds + self.means
    def compute(self):
        metrics = {
            "MSE": self.non_averaged_mse / self.num_samples,
            "RMSE": np.sqrt(self.non_averaged_mse / self.num_samples),
            "MAE": self.non_averaged_mae / self.num_samples,
            "MAPE": self.non_averaged_mape / self.num_samples,
        }
        return metrics
    def compute_std(self, dataset_type, raw_data_dir):
        if dataset_type == 'la':
            X = np.load(os.path.join(raw_data_dir, "node_values.npy")).transpose(
                (1, 2, 0)
            )
        elif dataset_type == 'bay':
            X = np.load(os.path.join(raw_data_dir, "pems_node_values.npy")).transpose(
                (1, 2, 0)
            )
        X = X.astype(np.float32)

        # should only have two values
        self.means = torch.tensor(np.mean(X, axis=(0, 2))[None,None,:,None])
        self.stds = torch.tensor(np.std(X, axis=(0, 2))[None,None,:,None])

    def reset(self):
        self.num_samples = 0
        self.non_averaged_mse = torch.zeros(2, 12)
        self.non_averaged_mae = torch.zeros(2, 12)
        self.non_averaged_mape = torch.zeros(2, 12)



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
proportion_original_dataset = 1  # Use 1% of the original dataset to debug
dataset_type = 'bay'  # 'la' or 'bay'
test_proportion_dataset = 0.2

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
    datasets = { 'la': get_metr_la_dataset, 'bay': get_pems_bay_dataset }
    dataset = datasets[dataset_type]()

    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        val_ratio=0,
        test_ratio=test_proportion_dataset,
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

    final_epoch = 0
    iterations = 0
    best_val_loss = float("inf")
    writer = SummaryWriter(logs_path)

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_epoch = checkpoint["epoch"] + 1
        iterations = checkpoint["iterations"]
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Loading trained that finished in epoch {final_epoch} with a validation loss of {best_val_loss:.4f}"
        )

    # Validation loop
    model.eval()
    metrics = Metrics(dataset_type)
    with torch.no_grad():
        for s, data in enumerate(test_loader):
            x, edge_index, edge_weight, y = data
            x = x.to(device)  # Remove batch dimension and move to device
            edge_index = edge_index[0].to(device)
            edge_weight = edge_weight[0].to(device)
            y = y.to(device)

            # print mean and std of x and y
            h = None
            y_hat = model(x, edge_index, edge_weight, h)

            metrics.update(y, y_hat)

            if s % (len(test_loader) // 100 + 1) == 0:
                print(f"\r {s}/{len(test_loader)}", end="")

    print()

    metrics = metrics.compute()

    print(f"Metrics: \n{metrics}")



import os
import torch

from src.model.dcrnn import DCRNN

# TODO: Create configuration file to enter the hyperparameters of the model, training, etc.
# TODO: Model parameters must be the same as those used for training. Save them to avoid load incompatibilities.

# Model
node_features = 2
out_channels = 32
K = 2

# Paths
model_path = "runs/model_checkpoint.pth"
logs_path = "runs/logs-explain/"
checkpoint_path = "runs/model_checkpoint.pth"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DCRNN(node_features=node_features, out_channels=out_channels, K=K).to(
        device
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from checkpoint with validation loss {checkpoint['best_val_loss']:.4f}"
        )
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

import os
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from src.data import get_loaders, get_pems_bay_dataset, get_metr_la_dataset
from src.model.dcrnn import DCRNN

# =====================================
# =====================================
# Explain a **trained** DCRNN model on the Pems-Bay dataset
# =====================================
# =====================================

# TODO: Create configuration file to enter the hyperparameters of the model, training, etc.
# TODO: Model parameters must be the same as those used for training. Save them to avoid load incompatibilities.

# Explainer
node_index = 110  # Explain node 10 as an example
time_index = 6  # Explain the first time step as an example
feature_index = 1  # Explain the first feature as an example

explanation_type = "model"  # ["model", "phenomenon"]
node_mask_type = "attributes"  # [None, "object", "common_attributes", "attributes"]
edge_mask_type = "object"  # [None, "object", "common_attributes", "attributes"]
model_config = dict(mode="regression", task_level="node", return_type="raw")

# Model
node_features = 2
out_channels = 32
K = 3

# Data
proportion_original_dataset = 0.01  # Use 1% of the original dataset to debug

# Training parameters
num_epochs_exp = 100
lr_exp = 0.01

# Evaluation running parameters
num_workers = 1
batch_size = 1

# Paths
model_path = "runs/model_checkpoint_dcrnn.pth"
logs_path = "runs/logs-explain/"
checkpoint_path = "runs/model_checkpoint_dcrnn.pth"

if __name__ == "__main__":
    # =====================================
    # Load model to explain
    # =====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DCRNN(node_features=node_features, out_channels=out_channels, K=K, predicted_time=time_index, predicted_feature=feature_index).to(
        device
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from checkpoint with validation loss {checkpoint['best_val_loss']:.4f}"
        )
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # =====================================
    # Test data to explain
    # =====================================

    dataset = get_pems_bay_dataset()

    _, _, test_loader = get_loaders(
        dataset,
        val_ratio=0.1,
        test_ratio=0.2,
        proportion_original_dataset=proportion_original_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # =====================================
    # Explainer definition
    # =====================================

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=num_epochs_exp, lr=lr_exp, log_steps=10, device=device
        ),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )

    # =====================================
    # Running explanation
    # =====================================

    for i, data in enumerate(test_loader):
        x, edge_index, edge_weight, y = data
        x = x.squeeze(0).to(device)
        edge_index = edge_index[0].to(device)
        edge_weight = edge_weight[0].to(device)
        y = y.squeeze(0).to(device)

        # need to put x into (N,F) shape
        time_steps = x.shape[2]
        x = x.view(x.shape[0], -1)

        explanation = explainer(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            # target=y,
            index=node_index,
            time_steps=time_steps,
        )
        break  # For now, only explain one batch

    print(f"Generated explanations in {explanation.available_explanations}")

    # save node mask and edge mask data using torch.save
    path = "node_mask.pt"
    torch.save(explanation.node_mask, path)
    print(f"Node mask has been saved to '{path}'")

    path = "edge_mask.pt"
    torch.save(explanation.edge_mask, path)
    print(f"Edge mask has been saved to '{path}'")

    try:
        path = "feature_importance.png"
        explanation.visualize_feature_importance(path, top_k=30)
        print(f"Feature importance plot has been saved to '{path}'")
    except Exception as e:
        print(f"Error while generating feature importance plot: {e}")

    try:
        path = "subgraph.pdf"
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")
    except Exception as e:
        print(f"Error while generating subgraph visualization plot: {e}")

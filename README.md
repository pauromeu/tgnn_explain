# XAI for Temporal Graph Neural Networks

## Requirements:

```bash
pip install torch torchvision torchaudio
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric
pip install torch-geometric-temporal
pip install matplotlib
pip install tensorboard
```

## Run the code:

### Train the model:

```bash
python -m src.train
```

### Explain the model:

```bash
python -m src.explain
```
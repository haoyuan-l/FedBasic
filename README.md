# FedBasic

A starter template for Federated Learning projects using **PyTorch** and **Flower**.

---

## 📁 Repository Structure

```
FedBasic/
├── project/              # Minimal FL template (Flower ≥1.22.0) - START HERE
│   ├── models.py         # SimpleCNN model definition
│   ├── task.py           # Data loading, training, and evaluation
│   ├── strategy.py       # FL strategy (FedAvg)
│   ├── client_app.py     # Flower ClientApp
│   ├── server_app.py     # Flower ServerApp
│   └── run.py            # Alternative Python script
│
├── example/              # Advanced example (MobileNet-V1 + Personalized FL)
│   ├── task.py
│   ├── client_app.py
│   ├── server_app.py
│   ├── strategy.py
│   └── run.py
│
├── flwr-0.6.12/          # Deprecated - Legacy Flower v0.6.12 implementation
│
└── pyproject.toml        # Dependencies and Flower configuration
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/FedBasic.git
cd FedBasic
pip install -e .
```

### Usage

**Method 1: Flower CLI (Recommended)**
```bash
flwr run .                 # Run with default settings
flwr run . local-sim-gpu   # Run with GPU support
```

**Method 2: Python Script**
```bash
cd project
python run.py
```

---

## 📂 File Structure

### `project/` - Minimal Template

| File | Description |
|------|-------------|
| `models.py` | SimpleCNN (3 conv layers, ~87K params) |
| `task.py` | Data loading (CIFAR-10 IID), train/test functions |
| `client_app.py` | Flower client with fit/evaluate methods |
| `server_app.py` | Flower server with FedAvg strategy |
| `strategy.py` | FedAvg wrapper (easy to customize) |
| `run.py` | Alternative Python execution script |

### `example/` - Advanced Template

Features MobileNet-V1, non-IID data (Dirichlet α=0.5), personalized FL with custom strategy.

---

## ⚙️ Configuration

Edit `pyproject.toml` for settings:

```toml
[tool.flwr.app.config]
num-server-rounds = 10
local-epochs = 5
fraction-train = 0.25
fraction-evaluate = 0.5

[tool.flwr.federations.local-sim]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0
```

---

## 📊 Default Setup

- **Dataset**: CIFAR-10 (IID distribution)
- **Model**: SimpleCNN (~87K params)
- **Clients**: 10
- **Rounds**: 10
- **Local epochs**: 5
- **Optimizer**: SGD (lr=0.01, momentum=0.9)

Expected accuracy: ~50-60% after 10 rounds.

---

## 📚 Resources

- [Flower Documentation](https://flower.ai/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 📝 License

Apache License 2.0

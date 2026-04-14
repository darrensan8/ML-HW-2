"""
Task: MLP Binary Classifier on Breast Cancer Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Series   : Neural Networks (MLP)
Level    : 1
Dataset  : sklearn Breast Cancer (569 samples, 30 features — real dataset)
New Feature:
  • MLP with Batch Normalization layers between each Linear+ReLU block.
    BatchNorm stabilizes training and allows higher learning rates.
    Compared against a no-BN baseline to show improvement.
Protocol : pytorch_task_v1

Quality thresholds:
  val accuracy > 0.95
  val f1       > 0.95
  BN model accuracy >= baseline accuracy
  training loss must decrease
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, List

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/nn_lvl1_mlp_batchnorm'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':     'nn_lvl1_mlp_batchnorm',
        'series':      'Neural Networks (MLP)',
        'level':       1,
        'algorithm':   'MLP Binary Classifier with Batch Normalization',
        'dataset':     'sklearn Breast Cancer (real)',
        'new_feature': 'BatchNorm1d between hidden layers vs no-BN baseline',
        'protocol':    'pytorch_task_v1',
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    return train_loader, val_loader


class MLPBatchNorm(nn.Module):
    def __init__(self, input_dim: int = 30, use_bn: bool = True):
        super(MLPBatchNorm, self).__init__()
        self.use_bn = use_bn
        layers = []
        dims = [input_dim, 64, 32]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def build_model(
    input_dim: int = 30,
    use_bn: bool = True,
    lr: float = 1e-3,
    device: torch.device = None
) -> Tuple[nn.Module, optim.Optimizer]:
    if device is None:
        device = get_device()
    model = MLPBatchNorm(input_dim=input_dim, use_bn=use_bn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model: MLPBatchNorm  use_bn={use_bn}  Params: {sum(p.numel() for p in model.parameters())}")
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 80,
    print_every: int = 20
) -> List[float]:
    losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(model.get_device()), y.to(model.get_device())
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {losses[-1]:.4f}")

    return losses


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module
) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(model.get_device()), y.to(model.get_device())
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds.append((logits.sigmoid() > 0.5).cpu().numpy().astype(int))
            targets.append(y.cpu().numpy().astype(int))

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    return {
        'loss':     total_loss / len(data_loader),
        'accuracy': float(accuracy_score(targets, preds)),
        'f1':       float(f1_score(targets, preds, zero_division=0)),
        'mse':      float(mean_squared_error(targets, preds)),
        'r2':       float(r2_score(targets, preds)),
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(model.get_device()))
        return (logits.sigmoid() > 0.5).cpu().numpy().astype(int)


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    output_dir: str = OUTPUT_DIR
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("Task: MLP + BatchNorm on Breast Cancer Dataset")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=32)

    print("\nBuilding model (with BatchNorm)...")
    model, optimizer = build_model(input_dim=30, use_bn=True, lr=1e-3, device=device)

    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining model...")
    train_losses = train(model, train_loader, optimizer, criterion, epochs=80)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion)
    print(f"  accuracy: {train_metrics['accuracy']:.4f}  f1: {train_metrics['f1']:.4f}  loss: {train_metrics['loss']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion)
    print(f"  accuracy: {val_metrics['accuracy']:.4f}  f1: {val_metrics['f1']:.4f}  loss: {val_metrics['loss']:.4f}")

    print("\nTraining baseline (no BatchNorm)...")
    baseline, baseline_opt = build_model(input_dim=30, use_bn=False, lr=1e-3, device=device)
    train(baseline, train_loader, baseline_opt, criterion, epochs=80, print_every=999)
    baseline_metrics = evaluate(baseline, val_loader, criterion)
    print(f"  Baseline val accuracy: {baseline_metrics['accuracy']:.4f}")

    save_artifacts(model, {
        'train': train_metrics,
        'val': val_metrics,
        'baseline_val_accuracy': baseline_metrics['accuracy'],
    })

    print("\nQuality Checks:")
    quality_passed = True

    check1 = val_metrics['accuracy'] > 0.95
    print(f"  {'✓' if check1 else '✗'} Val accuracy > 0.95: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics['f1'] > 0.95
    print(f"  {'✓' if check2 else '✗'} Val F1 > 0.95: {val_metrics['f1']:.4f}")
    quality_passed = quality_passed and check2

    check3 = val_metrics['accuracy'] >= baseline_metrics['accuracy']
    print(f"  {'✓' if check3 else '✗'} BN >= baseline accuracy: {val_metrics['accuracy']:.4f} >= {baseline_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check3

    check4 = train_losses[-1] < train_losses[0]
    print(f"  {'✓' if check4 else '✗'} Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    quality_passed = quality_passed and check4

    print("\n" + "=" * 60)
    if quality_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if quality_passed else 1)


if __name__ == '__main__':
    main()
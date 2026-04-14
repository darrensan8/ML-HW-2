"""
Task: MLP Regressor on Diabetes Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Series   : Neural Networks (MLP)
Level    : 1
Dataset  : sklearn Diabetes (442 samples, 10 features — real dataset)
New Feature:
  • Feedforward MLP trained with Adam optimizer and a StepLR
    learning-rate scheduler. Demonstrates how LR decay improves
    convergence vs a fixed learning rate.
Protocol : pytorch_task_v1

Quality thresholds:
  val R2  > 0.40
  val MSE < 0.60
  training loss must decrease
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, List

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/nn_lvl1_mlp_regressor'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':     'nn_lvl1_mlp_regressor',
        'series':      'Neural Networks (MLP)',
        'level':       1,
        'algorithm':   'MLP Regressor',
        'dataset':     'sklearn Diabetes (real)',
        'new_feature': 'Adam optimizer + StepLR learning-rate scheduler',
        'protocol':    'pytorch_task_v1',
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X, y = load_diabetes(return_X_y=True)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X).astype(np.float32)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    return train_loader, val_loader


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 10):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def build_model(
    input_dim: int = 10,
    lr: float = 1e-3,
    device: torch.device = None
) -> Tuple[nn.Module, optim.Optimizer]:
    if device is None:
        device = get_device()
    model = MLPRegressor(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model: MLPRegressor  Params: {sum(p.numel() for p in model.parameters())}")
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 100,
    print_every: int = 20
) -> List[float]:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
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
        scheduler.step()
        losses.append(float(np.mean(batch_losses)))

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {losses[-1]:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

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
            out = model(x)
            total_loss += criterion(out, y).item()
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    return {
        'loss': total_loss / len(data_loader),
        'mse':  float(mean_squared_error(targets, preds)),
        'r2':   float(r2_score(targets, preds)),
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(model.get_device())).cpu().numpy()


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
    print("Task: MLP Regressor on Diabetes Dataset")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=32)

    print("\nBuilding model...")
    model, optimizer = build_model(input_dim=10, lr=1e-3, device=device)

    criterion = nn.MSELoss()

    print("\nTraining model...")
    train_losses = train(model, train_loader, optimizer, criterion, epochs=100)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion)
    print(f"  loss: {train_metrics['loss']:.4f}  mse: {train_metrics['mse']:.4f}  r2: {train_metrics['r2']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion)
    print(f"  loss: {val_metrics['loss']:.4f}  mse: {val_metrics['mse']:.4f}  r2: {val_metrics['r2']:.4f}")

    save_artifacts(model, {'train': train_metrics, 'val': val_metrics})

    print("\nQuality Checks:")
    quality_passed = True

    check1 = val_metrics['r2'] > 0.40
    print(f"  {'✓' if check1 else '✗'} Val R2 > 0.40: {val_metrics['r2']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics['mse'] < 0.60
    print(f"  {'✓' if check2 else '✗'} Val MSE < 0.60: {val_metrics['mse']:.4f}")
    quality_passed = quality_passed and check2

    check3 = train_losses[-1] < train_losses[0]
    print(f"  {'✓' if check3 else '✗'} Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    quality_passed = quality_passed and check3

    print("\n" + "=" * 60)
    if quality_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if quality_passed else 1)


if __name__ == '__main__':
    main()
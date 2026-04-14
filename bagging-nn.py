"""
Task: Bagging Ensemble Classifier on Wine Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Series   : Neural Networks (MLP)
Level    : 1
Dataset  : sklearn Wine (178 samples, 13 features, 3 classes — real dataset)
New Feature:
  • Bootstrap Aggregation (Bagging) built from scratch in PyTorch.
    N_ESTIMATORS shallow MLPs are each trained on a bootstrap resample
    of the training data. Final prediction = majority vote across all
    estimators. Demonstrates that ensembling reduces variance vs a
    single MLP baseline.
Protocol : pytorch_task_v1

Quality thresholds:
  val accuracy (ensemble) > 0.90
  val accuracy (ensemble) >= val accuracy (single MLP baseline)
  val f1_macro > 0.88
  training loss must decrease
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, List

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/ens_lvl1_bagging'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':     'ens_lvl1_bagging',
        'series':      'Neural Networks (MLP)',
        'level':       1,
        'algorithm':   'Bagging (Bootstrap Aggregation)',
        'dataset':     'sklearn Wine (real)',
        'new_feature': 'Bootstrap resampling + majority-vote ensemble of MLPs',
        'protocol':    'pytorch_task_v1',
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
    X, y = load_wine(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Features: 13  Classes: 3")

    return train_loader, val_loader


class ShallowMLP(nn.Module):
    def __init__(self, in_features: int = 13, n_classes: int = 3):
        super(ShallowMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),          nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


class BaggingEnsemble:
    def __init__(self, n_estimators: int = 10, device: torch.device = None):
        self.n_estimators = n_estimators
        self.device = device or get_device()
        self.estimators: List[ShallowMLP] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 60, lr: float = 1e-2) -> List[float]:
        n = len(X_train)
        all_losses = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=n, replace=True)
            X_b = torch.tensor(X_train[idx], dtype=torch.float32).to(self.device)
            y_b = torch.tensor(y_train[idx], dtype=torch.long).to(self.device)

            model     = ShallowMLP().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                all_losses.append(loss.item())

            self.estimators.append(model)
        return all_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        votes = []
        for est in self.estimators:
            est.eval()
            with torch.no_grad():
                votes.append(est(Xt).argmax(1).cpu().numpy())
        votes = np.stack(votes)
        return np.apply_along_axis(
            lambda col: np.bincount(col, minlength=3).argmax(), axis=0, arr=votes
        )


def build_model(
    n_estimators: int = 10,
    lr: float = 1e-2,
    device: torch.device = None
) -> Tuple[BaggingEnsemble, float]:
    if device is None:
        device = get_device()
    model = BaggingEnsemble(n_estimators=n_estimators, device=device)
    print(f"Model: BaggingEnsemble  n_estimators={n_estimators}")
    return model, lr


def train(
    model: BaggingEnsemble,
    train_loader: DataLoader,
    lr: float,
    criterion: Any = None,
    epochs: int = 60,
    print_every: int = 1
) -> List[float]:
    X_list, y_list = [], []
    for x, y in train_loader:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)

    print(f"  Fitting {model.n_estimators} estimators ({epochs} epochs each)...")
    losses = model.fit(X_train, y_train, epochs=epochs, lr=lr)
    print(f"  Done. Total loss steps: {len(losses)}")
    return losses


def evaluate(
    model: BaggingEnsemble,
    data_loader: DataLoader,
    criterion: Any = None
) -> Dict[str, float]:
    X_list, y_list = [], []
    for x, y in data_loader:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    preds = model.predict(X)

    return {
        'accuracy': float(accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'mse':      float(mean_squared_error(y, preds)),
        'r2':       float(r2_score(y, preds)),
    }


def predict(model: BaggingEnsemble, data_loader: DataLoader) -> np.ndarray:
    X = np.concatenate([x.numpy() for x, _ in data_loader])
    return model.predict(X)


def save_artifacts(
    model: BaggingEnsemble,
    metrics: Dict[str, Any],
    output_dir: str = OUTPUT_DIR
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        [est.state_dict() for est in model.estimators],
        os.path.join(output_dir, 'estimators.pt')
    )
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def _train_baseline(train_loader: DataLoader, device: torch.device, epochs: int = 60, lr: float = 1e-2) -> ShallowMLP:
    X_list, y_list = [], []
    for x, y in train_loader:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    X = torch.tensor(np.concatenate(X_list), dtype=torch.float32).to(device)
    y = torch.tensor(np.concatenate(y_list), dtype=torch.long).to(device)

    baseline  = ShallowMLP().to(device)
    optimizer = optim.Adam(baseline.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        baseline.train()
        optimizer.zero_grad()
        criterion(baseline(X), y).backward()
        optimizer.step()

    return baseline


def main():
    print("=" * 60)
    print("Task: Bagging Ensemble on Wine Dataset")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=16)

    print("\nBuilding model...")
    model, lr = build_model(n_estimators=10, lr=1e-2, device=device)

    criterion = nn.CrossEntropyLoss()

    print("\nTraining model...")
    train_losses = train(model, train_loader, lr=lr, criterion=criterion, epochs=60)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion)
    print(f"  accuracy: {train_metrics['accuracy']:.4f}  f1: {train_metrics['f1_macro']:.4f}  loss: {train_metrics['mse']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion)
    print(f"  accuracy: {val_metrics['accuracy']:.4f}  f1: {val_metrics['f1_macro']:.4f}  loss: {val_metrics['mse']:.4f}")

    print("\nTraining baseline (single MLP)...")
    baseline = _train_baseline(train_loader, device, epochs=60)
    baseline.eval()
    X_val = np.concatenate([x.numpy() for x, _ in val_loader])
    y_val = np.concatenate([y.numpy() for _, y in val_loader])
    with torch.no_grad():
        baseline_preds = baseline(torch.tensor(X_val, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    baseline_acc = float(accuracy_score(y_val, baseline_preds))
    print(f"  Baseline val accuracy: {baseline_acc:.4f}")

    save_artifacts(model, {
        'train': train_metrics,
        'val': val_metrics,
        'baseline_val_accuracy': baseline_acc,
    })

    print("\nQuality Checks:")
    quality_passed = True

    check1 = val_metrics['accuracy'] > 0.90
    print(f"  {'✓' if check1 else '✗'} Val accuracy > 0.90: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics['accuracy'] >= baseline_acc
    print(f"  {'✓' if check2 else '✗'} Ensemble >= baseline accuracy: {val_metrics['accuracy']:.4f} >= {baseline_acc:.4f}")
    quality_passed = quality_passed and check2

    check3 = val_metrics['f1_macro'] > 0.88
    print(f"  {'✓' if check3 else '✗'} Val F1 macro > 0.88: {val_metrics['f1_macro']:.4f}")
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
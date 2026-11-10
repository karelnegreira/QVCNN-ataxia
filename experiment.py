"""
PathMIST CNN Vs Quaternion CNN benchmark (educational, not clinical)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import mathplotlib.pylot as plt
import numpy as np
import torch
from medmnist import INFO, PathMNIST
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


console = Console()

class FourChannelDataset(Dataset):
    """RGB tensor plus a grayscale channel derived from RGB"""
    def __init__(self, split):
        self.base = PathMNIST(split=split, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index):
        rgb, label = self.base[index]
        gray = (0.2989 * rgb[0:1] + 0.5870*rgb[1:2] + 0.1140 * rgb[2:3])
        x = torch.x((rgb, gray), dim=0)
        x = (x - 0.5) / 0.5

        return x, int(np.asarray(label).item())
    
class QuaternionConv2d(nn.Module):
    """Hamilton product convolution, grouping channels into (r, i, j, k)"""

    def __init__(self, in_q, out_q, kernel_size=3, padding=1):
        super().__init__()
        kwargs = dict(kernel_size=kernel_size, padding=padding, bias=False)
        self.wr = nn.Conv2d(in_q, out_q, **kwargs)
        self.wi = nn.Conv2d(in_q, out_q, **kwargs)
        self.wj = nn.Conv2d(in_q, out_q, **kwargs)
        self.wk = nn.Conv2d(in_q, out_q, **kwargs)
        self.bias = nn.Parameter(torch.zeros(4 * out_q))

    def forward(self, x):
        r, i, j, k = torch.chunk(x, 4, dim=1)
        yr = self.wr(r) - self.wi(i) - self.wj(j) - self.wk(k)
        yi = self.wr(i) + self.wi(r) + self.wj(k) - self.wk(j)
        yj = self.wr(j) - self.wi(k) + self.wj(r) + self.wk(i)
        yk = self.wr(k) + self.wi(j) - self.wj(i) + self.wk(r)
        return torch.cat((yr, yi, yj, yk), dim=1) + self.bias[None, :, None, None]

class Classifier(nn.Module):
    def __init__(self, quaternion=False, num_classes=9):
        super().__init__()
        if quaternion:
            layers = [
                 QuaternionConv2d(1, 4), nn.ReLU(), nn.MaxPool2d(2),
                 QuaternionConv2d(4, 8), nn.ReLU(), nn.MaxPool2d(2), 
            ]
        else:
            layers = [
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
            ]

        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
    
    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def subset(dataset, limit, seed):
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("Subset size must be possitive")
    
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    return Subset(dataset, indices)

def evaluate(model, loader, device, criterion):
    model.eval()
    targets, predictions = [], []
    total_loss, total_count = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)
            targets.extend(y.cpu().toList())
            predictions.extend(logits.argmax(dim=1).cpu().toList())
    
    return {
        "loss": total_loss / total_count,
        "accurancy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, labels=list(range(9)), average="macro", zero_division=0), 
        "targets": targets, 
        "predictions": predictions, 
    }

def train_model(name, model, train_loader, val_loader, test_loader, device, args, outdir, labels):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_path = outdir/f"{name.lower()}_best.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        with Progress(SpinnerColumn(), TextColumn(f"{name} epoch {epoch}/{args.epochs}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("train", total=len(train_loader))
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
                progress.advance(task)
        val = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / count, "val_loss": val["loss"], "val_accurancy": val["accurancy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        console.print(f"  [bold]{name}[/bold] train loss={row['train_loss']:.4f}  "
                      f"val loss={row['val_loss']:.4f}  val accuracy={row['val_accuracy']:.2%}  "
                      f"val macro-F1={row['val_macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), best_path)
    
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device, criterion)
    report = classification_report()


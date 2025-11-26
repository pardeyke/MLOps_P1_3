# %% Imports
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from PIL import Image

# %% Constants
IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
DEFAULT_DATA_DIR = Path("infos/mlops_biomass_data")
DEFAULT_IMAGE_DIR = DEFAULT_DATA_DIR / "images_med_res"
DEFAULT_LABELS = DEFAULT_DATA_DIR / "digital_biomass_labels.xlsx"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
LOG_FILE = Path("training.log")


# %% Utility helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return result.decode("utf-8").strip()
    except Exception:
        return None


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


# %% Dataset definition
class BiomassDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_path = self.image_dir / row["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        with Image.open(image_path).convert("RGB") as image:
            tensor = self.transform(image)
        target = torch.tensor(row["fresh_weight_total"], dtype=torch.float32)
        return tensor, target


# %% Data preparation
def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_excel(labels_path)
    df = df[df["fresh_weight_total"].notna()].copy()
    df["plant_number"] = df["plant_number"].fillna(-1).astype(int)
    return df


def filter_missing_images(df: pd.DataFrame, image_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    def exists(row: pd.Series) -> bool:
        return (image_dir / row["filename"]).exists()

    mask = df.apply(exists, axis=1)
    missing = (~mask).sum()
    if missing:
        logger.warning("Dropping %d rows with missing images.", missing)
    return df[mask].reset_index(drop=True)


def create_splits(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["plant_number"].astype(int)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms()
    train_dataset = BiomassDataset(train_df, image_dir, transform=train_transform)
    val_dataset = BiomassDataset(val_df, image_dir, transform=eval_transform)
    loader_args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader


# %% Model setup
def build_model(model_name: str, freeze_backbone: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT
        backbone = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'.")

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1),
    )
    return backbone


# %% Training utilities
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_samples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: List[float] = []
    targets_buffer: List[float] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds.extend(outputs.squeeze(1).cpu().tolist())
        targets_buffer.extend(targets.squeeze(1).cpu().tolist())
    mse = total_loss / max(1, total_samples)
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "preds": preds, "targets": targets_buffer}


def save_training_curves(train_losses: List[float], val_losses: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


# %% Main training orchestration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-based biomass regressor.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--model_name", type=str, default="resnet18", help="ResNet variant to use.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with images and metadata.")
    parser.add_argument("--labels_path", type=Path, default=None, help="Path to metadata Excel file.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory with images.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet backbone weights.")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Override to unfreeze all layers.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto/cpu/cuda).")
    parser.add_argument("--model_out", type=Path, default=MODELS_DIR / "biomass_resnet.pt", help="Path to save model.")
    parser.add_argument("--curves_path", type=Path, default=RESULTS_DIR / "training_curves.png", help="Loss plot path.")
    parser.add_argument("--metrics_path", type=Path, default=RESULTS_DIR / "metrics.txt", help="Metrics JSON path.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    return device


def main() -> None:
    args = parse_args()
    args.data_dir = Path(args.data_dir)
    if args.labels_path is None:
        args.labels_path = args.data_dir / "digital_biomass_labels.xlsx"
    if args.image_dir is None:
        args.image_dir = args.data_dir / "images_med_res"
    args.model_out = Path(args.model_out)
    args.curves_path = Path(args.curves_path)
    args.metrics_path = Path(args.metrics_path)
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone
    if args.freeze_backbone is False and args.unfreeze_backbone is False:
        freeze_backbone = True

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.curves_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger = setup_logging()
    logger.info("Starting training run.")
    logger.info("Arguments: %s", vars(args))
    git_hash = get_git_hash()
    if git_hash:
        logger.info("Git commit: %s", git_hash)

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    df = load_labels(args.labels_path)
    df = filter_missing_images(df, args.image_dir, logger)
    train_df, val_df = create_splits(df, args.val_split, args.seed)
    logger.info("Training samples: %d | Validation samples: %d", len(train_df), len(val_df))

    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_name, freeze_backbone=freeze_backbone).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_metrics: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)
        val_loss = metrics["mse"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            "Epoch %d/%d - train_loss: %.5f - val_loss: %.5f - val_rmse: %.5f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            metrics["rmse"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {"val_mse": val_loss, "val_rmse": metrics["rmse"], "epoch": epoch}
            torch.save(model.state_dict(), args.model_out)
            logger.info("Saved new best model to %s", args.model_out)

    save_training_curves(train_losses, val_losses, args.curves_path)
    logger.info("Saved training curves to %s", args.curves_path)
    save_metrics(best_metrics, args.metrics_path)
    logger.info("Saved metrics to %s", args.metrics_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()

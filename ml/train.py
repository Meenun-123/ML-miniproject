import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.dataset import ChessMateDataset

WEIGHTS_DIR = PROJECT_ROOT / "weights"


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int = 128, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.
    Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> SE -> Skip Add -> ReLU.
    """

    def __init__(self, channels: int = 128, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class ChessTacticsResNet(nn.Module):
    """
    ResNet-8 backbone with Squeeze-and-Excitation attention and dual heads:
    - Policy Head: 4096 discrete move logits
    - Value Head: Scalar in [-1, 1] predicting active player's win certainty
    """

    def __init__(
        self, in_channels: int = 18, num_blocks: int = 4, channels: int = 128
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy Head (Move probabilities over 4,096 discrete classes)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 4096),
        )

        # Value Head (Auxiliary loss: mate win certainty [-1, 1])
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# Backwards compatibility alias
ChessTacticsCNN = ChessTacticsResNet


def topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int = 1
) -> float:
    """Compute top-k categorical accuracy."""
    _, topk_indices = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
    return correct.float().sum().item() / labels.size(0)


def export_onnx(model: nn.Module, device: torch.device):
    """Export best model checkpoint to ONNX."""
    best_pt_path = WEIGHTS_DIR / "chess_mate_cnn.pt"
    if best_pt_path.exists():
        print(f"Exporting best model checkpoint ({best_pt_path}) to ONNX...")
        model.load_state_dict(
            torch.load(str(best_pt_path), weights_only=True, map_location=device)
        )
    model.eval()

    dummy_input = torch.randn(1, 18, 8, 8, device=device)
    onnx_out_path = str(WEIGHTS_DIR / "chess_mate_cnn.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_out_path,
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
            opset_version=18,
        )
        print(f"Successfully exported updated weights to {onnx_out_path}")
    except Exception as e:
        print(f"ONNX export encountered error: {e}")


def train(
    data_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 2e-3,
    num_workers: int = 4,
    finetune_blunders: str | None = None,
    opening_concepts: str | None = "data/opening_concepts.csv",
    freeze_backbone: bool = False,
    blunder_ratio: float = 0.85,
    early_stopping: int = 2,
    scheduler_type: str = "onecycle",
    restart_epochs: int = 5,
    num_blocks: int = 4,
    channels: int = 128,
):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dataset Construction
    if finetune_blunders and Path(finetune_blunders).exists():
        print(f"--- Fine-Tuning Mode Activated ---")
        print(f"Loading mined blunders: {finetune_blunders}")
        df_blunders = pd.read_csv(finetune_blunders)
        print(f"Loaded {len(df_blunders)} blunder refutations.")

        print(f"Loading base puzzle dataset: {data_path}")
        df_puzzles = pd.read_csv(data_path)

        has_opening = opening_concepts and Path(opening_concepts).exists()
        if has_opening:
            print(f"Loading strategic opening concepts: {opening_concepts}")
            df_opening = pd.read_csv(opening_concepts)
            print(f"Loaded {len(df_opening)} opening mastery samples.")

            # Construct 40/40/20 Curriculum Blended Dataset
            target_total = max(3500, len(df_blunders) * 5)
            n_opening = int(target_total * 0.40)
            n_blunder = int(target_total * 0.40)
            n_puzzle = target_total - n_opening - n_blunder

            df_opening_samples = df_opening.sample(
                n=n_opening, replace=(len(df_opening) < n_opening), random_state=42
            )
            df_blunder_samples = df_blunders.sample(
                n=n_blunder, replace=True, random_state=42
            )
            df_puzzle_samples = df_puzzles.sample(
                n=min(len(df_puzzles), n_puzzle),
                replace=(len(df_puzzles) < n_puzzle),
                random_state=42,
            )

            df_mixed = pd.concat([df_opening_samples, df_blunder_samples, df_puzzle_samples], ignore_index=True)
            df_mixed = df_mixed.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(
                f"Created 40/40/20 Curriculum Blended Dataset: {len(df_mixed)} samples "
                f"(40% Opening Concepts / 40% Blunder Defense / 20% Mate Tactics)"
            )
        else:
            # Construct 85/15 blended dataset
            target_total = max(3000, len(df_blunders) * 5)
            n_blunder_samples = int(target_total * blunder_ratio)
            n_puzzle_samples = target_total - n_blunder_samples

            df_blunder_samples = df_blunders.sample(
                n=n_blunder_samples, replace=True, random_state=42
            )
            df_puzzle_samples = df_puzzles.sample(
                n=min(len(df_puzzles), n_puzzle_samples),
                replace=(len(df_puzzles) < n_puzzle_samples),
                random_state=42,
            )

            df_mixed = pd.concat([df_blunder_samples, df_puzzle_samples], ignore_index=True)
            df_mixed = df_mixed.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(
                f"Created blended dataset: {len(df_mixed)} samples "
                f"({blunder_ratio*100:.0f}% Blunder Corrections / {(1-blunder_ratio)*100:.0f}% Mate Puzzles)"
            )

        train_dataset = ChessMateDataset(df_mixed, split="train", augment=True)
        val_dataset = ChessMateDataset(df_mixed, split="val", augment=False)
    else:
        print(f"Loading base datasets from {data_path}...")
        train_dataset = ChessMateDataset(data_path, split="train", augment=True)
        val_dataset = ChessMateDataset(data_path, split="val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ChessTacticsResNet(
        in_channels=18, num_blocks=num_blocks, channels=channels
    ).to(device)

    # If fine-tuning, load existing checkpoint
    if finetune_blunders:
        pt_checkpoint = WEIGHTS_DIR / "chess_mate_cnn.pt"
        if pt_checkpoint.exists():
            print(f"Loading pre-trained checkpoint from {pt_checkpoint}...")
            model.load_state_dict(
                torch.load(pt_checkpoint, map_location=device, weights_only=True)
            )

        if freeze_backbone:
            print("Freezing ResNet backbone (Stem + Residual Blocks). Training policy & value heads only.")
            for param in model.stem.parameters():
                param.requires_grad = False
            for param in model.res_blocks.parameters():
                param.requires_grad = False

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse_criterion = nn.MSELoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.99)
    )

    if scheduler_type == "cosine_restarts":
        print(
            f"Using CosineAnnealingWarmRestarts scheduler (Period: {restart_epochs} epochs, eta_min: 1e-6)"
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, restart_epochs * len(train_loader)),
            T_mult=1,
            eta_min=1e-6,
        )
    else:
        total_steps = max(1, epochs * len(train_loader))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
        )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_p_loss = 0.0
        train_v_loss = 0.0

        for batch_idx, (inputs, labels, values) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            values = values.to(
                device, dtype=torch.float32, non_blocking=True
            ).unsqueeze(1)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_preds, value_preds = model(inputs)
                p_loss = ce_criterion(policy_preds, labels)
                v_loss = mse_criterion(value_preds, values)
                loss = p_loss + 0.5 * v_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()

            train_loss += loss.item()
            train_p_loss += p_loss.item()
            train_v_loss += v_loss.item()

        n_batches = max(1, len(train_loader))
        train_loss /= n_batches
        train_p_loss /= n_batches
        train_v_loss /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top3 = 0.0

        with torch.no_grad():
            for inputs, labels, values in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                values = values.to(
                    device, dtype=torch.float32, non_blocking=True
                ).unsqueeze(1)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    policy_preds, value_preds = model(inputs)
                    p_loss = ce_criterion(policy_preds, labels)
                    v_loss = mse_criterion(value_preds, values)
                    loss = p_loss + 0.5 * v_loss

                val_loss += loss.item()
                val_top1 += (
                    topk_accuracy(policy_preds, labels, k=1) * inputs.size(0)
                )
                val_top3 += (
                    topk_accuracy(policy_preds, labels, k=3) * inputs.size(0)
                )

        n_val_batches = max(1, len(val_loader))
        val_loss /= n_val_batches
        val_top1 /= len(val_dataset)
        val_top3 /= len(val_dataset)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f}")
        print(
            f"  Train Loss: {train_loss:.4f} (Policy: {train_p_loss:.4f}, Value: {train_v_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.4f} | Val Top-3: {val_top3:.4f}"
        )

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            epochs_no_improve = 0
            torch.save(
                model.state_dict(), str(WEIGHTS_DIR / "chess_mate_cnn.pt")
            )
            print(f"  --> Saved new best model (Top-1: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping and finetune_blunders:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    # Automated ONNX Export of best checkpoint
    export_onnx(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SE-ResNet-8 Training and Targeted Blunder Fine-Tuning"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "mate_puzzles.csv"),
        help="Path to filtered puzzles CSV",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--finetune_blunders",
        type=str,
        default=None,
        help="Path to mined blunders CSV for targeted fine-tuning",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze stem and ResNet blocks, training only policy and value heads",
    )
    parser.add_argument(
        "--blunder_ratio",
        type=float,
        default=0.85,
        help="Ratio of blunder corrections vs checkmate puzzles in blended training set (default: 0.85, 15%% puzzles)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=2,
        help="Number of epochs without improvement before early stopping",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["onecycle", "cosine_restarts"],
        default="onecycle",
        help="Learning rate scheduler: 'onecycle' or 'cosine_restarts' (CosineAnnealingWarmRestarts)",
    )
    parser.add_argument(
        "--restart_epochs",
        type=int,
        default=5,
        help="Number of epochs per cosine restart period when using cosine_restarts (default: 5)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help="Number of residual blocks (4 for SE-ResNet-8, 6 for SE-ResNet-12, 8 for SE-ResNet-16)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=128,
        help="Channel dimension of residual blocks (default: 128, or 192 for widened capacity)",
    )
    parser.add_argument(
        "--opening_concepts",
        type=str,
        default="data/opening_concepts.csv",
        help="Path to opening concepts CSV for 40/40/20 curriculum blending",
    )

    args = parser.parse_args()

    # Determine default epochs and lr based on fine-tuning vs full training
    if args.finetune_blunders:
        epochs = args.epochs if args.epochs is not None else 3
        lr = args.lr if args.lr is not None else (1e-4 if args.freeze_backbone else 5e-5)
    else:
        epochs = args.epochs if args.epochs is not None else 20
        lr = args.lr if args.lr is not None else 2e-3

    train(
        data_path=args.data_path,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=lr,
        finetune_blunders=args.finetune_blunders,
        opening_concepts=args.opening_concepts,
        freeze_backbone=args.freeze_backbone,
        blunder_ratio=args.blunder_ratio,
        early_stopping=args.early_stopping,
        scheduler_type=args.scheduler,
        restart_epochs=args.restart_epochs,
        num_blocks=args.num_blocks,
        channels=args.channels,
    )

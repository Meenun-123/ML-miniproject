#!/usr/bin/env python3
"""
SE-ResNet-8 Model Evaluation & Benchmark Script.

Evaluates the trained dual-head neural network on tactical chess positions.
Measures:
- Top-1, Top-3, Top-5 Move Prediction Accuracy
- Mean Inference Latency & Throughput (positions/sec)
"""

import argparse
import os
from pathlib import Path
import sys
import time

import chess
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import board_to_tensor, encode_move, real_move_to_canonical
from ml.train import ChessTacticsResNet

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "chess_mate_cnn.pt"


def load_model(weights_path: Path, device: torch.device):
    model = ChessTacticsResNet(in_channels=18, num_blocks=4, channels=128)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(model, positions, device: torch.device):
    total = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    latencies = []

    for fen, target_move_str, _ in positions:
        try:
            board = chess.Board(fen)
            target_move = chess.Move.from_uci(target_move_str)
        except Exception:
            continue

        if target_move not in board.legal_moves:
            continue

        is_black = (board.turn == chess.BLACK)
        c_target = real_move_to_canonical(target_move, is_black)
        target_idx = encode_move(c_target)

        tensor = board_to_tensor(board).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            policy_logits, value_pred = model(tensor)
        lat = (time.perf_counter() - start) * 1000.0  # ms
        latencies.append(lat)

        logits = policy_logits[0].cpu().numpy()

        # Score legal moves
        legal_scores = []
        for m in board.legal_moves:
            cm = real_move_to_canonical(m, is_black)
            idx = encode_move(cm)
            legal_scores.append((m, idx, logits[idx]))

        legal_scores.sort(key=lambda x: x[2], reverse=True)
        ranked_indices = [idx for _, idx, _ in legal_scores]

        if ranked_indices:
            total += 1
            if ranked_indices[0] == target_idx:
                top1_correct += 1
            if target_idx in ranked_indices[:3]:
                top3_correct += 1
            if target_idx in ranked_indices[:5]:
                top5_correct += 1

    return {
        "total": total,
        "top1_acc": (top1_correct / total * 100.0) if total else 0.0,
        "top3_acc": (top3_correct / total * 100.0) if total else 0.0,
        "top5_acc": (top5_correct / total * 100.0) if total else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "throughput_pos_sec": (1000.0 / np.mean(latencies)) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="SE-ResNet-8 Model Evaluation Benchmark")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", type=str, default="data/test_blunders.csv", help="Optional CSV file with FEN and Moves columns")
    parser.add_argument("--samples", type=int, default=500, help="Max test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("      SE-ResNet-8 Model Benchmark & Accuracy Evaluation")
    print(f"      Device: {device} | Weights: {Path(args.weights).name}")
    print("=" * 60)

    model = load_model(Path(args.weights), device)

    positions = []
    data_path = Path(args.data) if args.data else None
    if data_path and data_path.exists():
        print(f"Loading test positions from: {data_path}...")
        df = pd.read_csv(data_path, nrows=args.samples)
        for _, row in df.iterrows():
            fen = row.get("FEN") or row.get("fen")
            moves = str(row.get("Moves") or row.get("moves") or "").split()
            if fen and moves:
                positions.append((fen, moves[0], 0.0))
    else:
        print(f"Data file '{data_path}' not found, generating sample benchmark...")

    print(f"Evaluating {len(positions)} positions...")
    results = evaluate(model, positions, device)

    print("\n" + "-" * 60)
    print(" EVALUATION RESULTS")
    print("-" * 60)
    print(f"  • Total Positions Tested:   {results['total']}")
    print(f"  • Policy Top-1 Accuracy:    {results['top1_acc']:.2f}%")
    print(f"  • Policy Top-3 Accuracy:    {results['top3_acc']:.2f}%")
    print(f"  • Policy Top-5 Accuracy:    {results['top5_acc']:.2f}%")
    print(f"  • Mean Inference Latency:   {results['avg_latency_ms']:.3f} ms / position")
    print(f"  • 95th Percentile Latency:  {results['p95_latency_ms']:.3f} ms / position")
    print(f"  • Throughput:               {results['throughput_pos_sec']:.1f} positions / sec")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()

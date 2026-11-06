#!/usr/bin/env python3
"""
SE-ResNet-8 Model Evaluation & Benchmark Script.

Evaluates the trained dual-head neural network on tactical chess positions.
Supports both PyTorch and ONNX Runtime backends.
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import board_to_numpy, board_to_tensor, encode_move, real_move_to_canonical

DEFAULT_WEIGHTS_PT = PROJECT_ROOT / "weights" / "chess_mate_cnn.pt"
DEFAULT_WEIGHTS_ONNX = PROJECT_ROOT / "weights" / "chess_mate_cnn.onnx"

# Curated fallback benchmark of classic tactical motifs
BUILTIN_BENCHMARK = [
    # Scholar's mate attack: Qxf7#
    ("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4", "f3f7", 1.0),
    # Back-rank rook mate: Rd8#
    ("6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", "d1d8", 1.0),
    # Opera Box mate: Rd8#
    ("4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 0", "b3b8", 1.0),
    # Smothered mate setup: Qg8+
    ("6k1/5Npp/8/8/8/8/8/4Q1K1 w - - 0 1", "e1e8", 1.0),
    # King Hunt / Greek Gift capture: Bxh7+
    ("r1bq1rk1/pp1nbppp/4p3/3pP3/3P4/3BBN2/PPQ2PPP/R4RK1 w - - 1 12", "d3h7", 1.0),
    # Back-rank queen infiltration: Qe8#
    ("3r2k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "e1e8", 1.0),
    # Italian game bishop sacrifice on f7: Bxf7+
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "c4f7", 1.0),
]


def load_pytorch_model(weights_path: Path, device: "torch.device"):
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed. Install via: pip install torch")
    from ml.train import ChessTacticsResNet
    model = ChessTacticsResNet(in_channels=18, num_blocks=4, channels=128)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_onnx_model(weights_path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for ONNX inference. Install via: pip install onnxruntime")
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX model weights not found at: {weights_path}")
    session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
    return session


def evaluate(predictor_fn, positions):
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

        start = time.perf_counter()
        logits, _ = predictor_fn(board)
        lat = (time.perf_counter() - start) * 1000.0  # ms
        latencies.append(lat)

        # Score legal moves
        legal_scores = []
        for m in board.legal_moves:
            cm = real_move_to_canonical(m, is_black)
            idx = encode_move(cm)
            score = logits[idx] if idx < len(logits) else -1e9
            legal_scores.append((m, idx, score))

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
        "throughput_pos_sec": (1000.0 / np.mean(latencies)) if (latencies and np.mean(latencies) > 0) else 0.0,
    }


def find_data_file(custom_path: str | None) -> Path | None:
    """Locate dataset file resolving relative to PROJECT_ROOT or current dir."""
    if custom_path:
        p1 = Path(custom_path)
        if p1.is_file():
            return p1
        p2 = PROJECT_ROOT / custom_path
        if p2.is_file():
            return p2
        return None

    # Fallback cascade
    candidates = [
        PROJECT_ROOT / "data" / "mined_blunders.csv",
        PROJECT_ROOT / "data" / "test_blunders.csv",
        PROJECT_ROOT / "data" / "sound_sacrifices.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="SE-ResNet-8 Model Evaluation Benchmark")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt or .onnx)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX Runtime for evaluation instead of PyTorch")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset (FEN and Moves columns)")
    parser.add_argument("--samples", type=int, default=500, help="Max test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")

    args = parser.parse_args()

    # Determine backend
    use_onnx = args.onnx or not HAS_TORCH
    if use_onnx:
        weights_path = Path(args.weights) if args.weights else DEFAULT_WEIGHTS_ONNX
        if not weights_path.exists() and DEFAULT_WEIGHTS_ONNX.exists():
            weights_path = DEFAULT_WEIGHTS_ONNX
        print("=" * 60)
        print("      SE-ResNet-8 Model Benchmark & Accuracy Evaluation")
        print(f"      Backend: ONNX Runtime (CPU) | Weights: {weights_path.name}")
        print("=" * 60)
        session = load_onnx_model(weights_path)
        def predictor_fn(board: chess.Board):
            tensor = board_to_numpy(board)[np.newaxis, ...].astype(np.float32)
            inputs = {session.get_inputs()[0].name: tensor}
            outputs = session.run(None, inputs)
            return outputs[0][0], float(outputs[1][0][0])
    else:
        weights_path = Path(args.weights) if args.weights else DEFAULT_WEIGHTS_PT
        device = torch.device(args.device)
        print("=" * 60)
        print("      SE-ResNet-8 Model Benchmark & Accuracy Evaluation")
        print(f"      Backend: PyTorch ({device}) | Weights: {weights_path.name}")
        print("=" * 60)
        model = load_pytorch_model(weights_path, device)
        def predictor_fn(board: chess.Board):
            tensor = board_to_tensor(board).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, val = model(tensor)
            return policy[0].cpu().numpy(), float(val.item())

    positions = []
    data_path = find_data_file(args.data)
    if data_path and data_path.exists():
        print(f"Loading test positions from: {data_path}...")
        df = pd.read_csv(data_path, nrows=args.samples)
        for _, row in df.iterrows():
            fen = row.get("FEN") or row.get("fen")
            moves = str(row.get("Moves") or row.get("moves") or "").split()
            if fen and moves:
                if len(moves) == 1:
                    positions.append((fen, moves[0], 0.0))
                else:
                    # Lichess puzzle format: moves[0] is setup move, moves[1] is target response
                    try:
                        b = chess.Board(fen)
                        b.push_uci(moves[0])
                        positions.append((b.fen(), moves[1], 0.0))
                    except Exception:
                        positions.append((fen, moves[0], 0.0))
    else:
        print("Dataset CSV not found; using built-in tactical benchmark suite...")
        positions = BUILTIN_BENCHMARK.copy()

    print(f"Evaluating {len(positions)} positions...")
    results = evaluate(predictor_fn, positions)

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


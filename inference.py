#!/usr/bin/env python3
"""
SE-ResNet-8 Standalone Neural Chess Inference Script.

Evaluates chess board positions using the trained dual-head SE-ResNet-8 deep learning model.
Outputs:
- ASCII Board Display & Active Player Perspective
- Value Head: Scalar Position Evaluation V(s) in [-1.0, +1.0] and Win Probability
- Policy Head: Top-K Legal Move Candidates ranked by Softmax Probability
"""

import argparse
import os
from pathlib import Path
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import (
    board_to_tensor,
    canonical_move_to_real,
    decode_move,
    encode_move,
    real_move_to_canonical,
)
from ml.train import ChessTacticsResNet

DEFAULT_WEIGHTS_PT = PROJECT_ROOT / "weights" / "chess_mate_cnn.pt"
DEFAULT_WEIGHTS_ONNX = PROJECT_ROOT / "weights" / "chess_mate_cnn.onnx"


def load_pytorch_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load PyTorch checkpoint into ChessTacticsResNet model."""
    if not weights_path.exists():
        raise FileNotFoundError(f"PyTorch weights not found at: {weights_path}")
    
    model = ChessTacticsResNet(in_channels=18, num_blocks=4, channels=128)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_onnx_model(weights_path: Path):
    """Load exported ONNX model with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for ONNX inference. Install via: pip install onnxruntime")
    
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX model not found at: {weights_path}")
    
    session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
    return session


def predict_pytorch(model: torch.nn.Module, board: chess.Board, device: torch.device):
    """Run forward pass using PyTorch."""
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value_pred = model(tensor)
    return policy_logits[0].cpu().numpy(), float(value_pred.item())


def predict_onnx(session, board: chess.Board):
    """Run forward pass using ONNX Runtime."""
    tensor = board_to_tensor(board).unsqueeze(0).numpy().astype(np.float32)
    inputs = {session.get_inputs()[0].name: tensor}
    outputs = session.run(None, inputs)
    policy_logits, value_pred = outputs[0][0], float(outputs[1][0][0])
    return policy_logits, value_pred


def rank_legal_moves(board: chess.Board, policy_logits: np.ndarray, top_k: int = 5):
    """Filter and rank legal moves from active perspective using Softmax probabilities."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    is_black = (board.turn == chess.BLACK)
    move_scores = []
    for move in legal_moves:
        c_move = real_move_to_canonical(move, is_black)
        idx = encode_move(c_move)
        score = policy_logits[idx] if idx < len(policy_logits) else -1e9
        move_scores.append((move, score))

    # Compute softmax over legal moves only
    logits_array = np.array([s for _, s in move_scores], dtype=np.float32)
    exp_logits = np.exp(logits_array - np.max(logits_array))
    probs = exp_logits / np.sum(exp_logits)

    ranked = []
    for i, (move, _) in enumerate(move_scores):
        ranked.append({
            "move": move,
            "uci": move.uci(),
            "san": board.san(move),
            "probability": float(probs[i]),
            "raw_logit": float(logits_array[i]),
        })

    ranked.sort(key=lambda x: x["probability"], reverse=True)
    return ranked[:top_k]


def display_ascii_board(board: chess.Board):
    """Render an ASCII chess board with coordinates."""
    lines = str(board).split("\n")
    print("      a   b   c   d   e   f   g   h")
    print("    +-------------------------------+ - - - - - - - - -")
    for r, line in zip(range(8, 0, -1), lines):
        cells = line.split(" ")
        formatted = " | ".join(cells)
        print(f"  {r} | {formatted} | {r}")
    print("    +-------------------------------+ - - - - - - - - -")
    print("      a   b   c   d   e   f   g   h\n")


def main():
    parser = argparse.ArgumentParser(
        description="SE-ResNet-8 Neural Chess Model Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fen",
        type=str,
        default="r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5",
        help="Target board position in FEN (Forsyth-Edwards Notation)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pt or .onnx). Defaults to weights/chess_mate_cnn.pt",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX Runtime for inference instead of PyTorch",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top legal candidate moves to display",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for PyTorch inference ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    # Validate FEN
    try:
        board = chess.Board(args.fen)
    except ValueError as e:
        print(f"❌ Error: Invalid FEN string '{args.fen}': {e}")
        sys.exit(1)

    print("=" * 65)
    print("     SE-ResNet-8 Dual-Head Neural Chess Engine Inference")
    print("     Architecture: 4× SE-ResBlock (128 ch) | 18-Plane Input")
    print("=" * 65)
    print(f"\nFEN: {board.fen()}")
    turn_str = "White" if board.turn == chess.WHITE else "Black"
    print(f"Active Player to Move: {turn_str}\n")

    display_ascii_board(board)

    # Determine backend
    if args.onnx:
        weights_path = Path(args.weights) if args.weights else DEFAULT_WEIGHTS_ONNX
        print(f"Loading ONNX Model: {weights_path}...")
        session = load_onnx_model(weights_path)
        policy_logits, value_score = predict_onnx(session, board)
    else:
        weights_path = Path(args.weights) if args.weights else DEFAULT_WEIGHTS_PT
        device = torch.device(args.device)
        print(f"Loading PyTorch Model: {weights_path} on {device}...")
        model = load_pytorch_model(weights_path, device)
        policy_logits, value_score = predict_pytorch(model, board, device)

    # Win probability mapping from Tanh [-1, 1] -> [0%, 100%]
    win_prob = (value_score + 1.0) / 2.0 * 100.0

    print("\n" + "-" * 65)
    print(f" POSITION EVALUATION (Value Head)")
    print("-" * 65)
    print(f"  • Scalar Value V(s):       {value_score:+.4f}  (in range [-1.0, +1.0])")
    print(f"  • Active Win Probability:  {win_prob:.2f}%")
    
    # Qualitative interpretation
    if value_score > 0.4:
        assessment = f"Decisive advantage for {turn_str}"
    elif value_score > 0.15:
        assessment = f"Moderate advantage for {turn_str}"
    elif value_score > -0.15:
        assessment = "Approximately equal / balanced position"
    elif value_score > -0.4:
        assessment = f"Moderate advantage for {'Black' if turn_str == 'White' else 'White'}"
    else:
        assessment = f"Decisive advantage for {'Black' if turn_str == 'White' else 'White'}"
    print(f"  • Positional Assessment:   {assessment}")

    print("\n" + "-" * 65)
    print(f" CANDIDATE MOVE PREDICTIONS (Policy Head - Top {args.top_k})")
    print("-" * 65)
    print(f"  {'Rank':<5} {'SAN':<8} {'UCI':<8} {'Probability':<14} {'Confidence Bar'}")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*13} {'-'*20}")

    ranked_moves = rank_legal_moves(board, policy_logits, top_k=args.top_k)
    for rank, entry in enumerate(ranked_moves, start=1):
        bar_len = int(entry["probability"] * 30)
        bar = "█" * bar_len
        print(f"  #{rank:<4} {entry['san']:<8} {entry['uci']:<8} {entry['probability']*100:>6.2f}%       {bar}")

    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Automated Blunder Mining & Distillation Pipeline ("The Grandmaster Coach")
Adversarially evaluates SE-ResNet-8 moves against Stockfish to harvest tactical blind spots.
"""

import argparse
import os
from pathlib import Path
import random
import shutil
import sys
import time

import chess
import chess.engine
import chess.polyglot
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Insert project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

try:
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_stockfish(custom_path: str | None = None) -> str:
    """Locate local Stockfish executable."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    
    which_sf = shutil.which("stockfish")
    if which_sf:
        candidates.append(which_sf)
        
    candidates.extend([
        str(PROJECT_ROOT / "bin" / "stockfish"),
        os.path.expanduser("~/.local/bin/stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ])
    
    for path in candidates:
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return path
            
    raise FileNotFoundError(
        "Could not find Stockfish executable. Please install Stockfish or specify --stockfish_path."
    )


class ModelInference:
    """Unified inference wrapper for ONNX and PyTorch SE-ResNet-8."""

    def __init__(self, weights_dir: Path = PROJECT_ROOT / "weights"):
        self.onnx_path = weights_dir / "chess_mate_cnn.onnx"
        self.pt_path = weights_dir / "chess_mate_cnn.pt"
        self.mode = "none"
        self.session = None
        self.model = None

        if HAS_ONNX and self.onnx_path.exists():
            print(f"Loading ONNX model from {self.onnx_path}...")
            self.session = onnxruntime.InferenceSession(str(self.onnx_path))
            self.input_name = self.session.get_inputs()[0].name
            self.mode = "onnx"
        elif self.pt_path.exists():
            print(f"Loading PyTorch model from {self.pt_path}...")
            self.model = ChessTacticsResNet(in_channels=18).to(DEVICE)
            self.model.load_state_dict(
                torch.load(self.pt_path, map_location=DEVICE, weights_only=True)
            )
            self.model.eval()
            self.mode = "pytorch"
        else:
            raise FileNotFoundError(
                f"No model weights found in {weights_dir} (expected .onnx or .pt)."
            )

    def predict_move(self, board: chess.Board) -> tuple[chess.Move | None, float]:
        """
        Predict model's best move with canonical perspective flipping and illegal move masking.
        Returns (best_real_move, confidence).
        """
        if not list(board.legal_moves):
            return None, 0.0

        is_black = board.turn == chess.BLACK
        tensor_input = board_to_tensor(board).unsqueeze(0).to(DEVICE)

        if self.mode == "onnx":
            numpy_input = tensor_input.cpu().numpy().astype(np.float32)
            outputs = self.session.run(None, {self.input_name: numpy_input})
            logits = torch.tensor(outputs[0][0])
        else:
            with torch.no_grad():
                policy_logits, _ = self.model(tensor_input)
                logits = policy_logits[0].cpu()

        # Mask illegal moves in canonical coordinate space
        mask = torch.full((4096,), float("-inf"))
        for m in board.legal_moves:
            cm = real_move_to_canonical(m, is_black)
            mask[encode_move(cm)] = 0.0

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=0)

        best_idx = torch.argmax(probs).item()
        best_prob = probs[best_idx].item()

        best_canonical_move = decode_move(best_idx)
        best_real_move = canonical_move_to_real(best_canonical_move, is_black)

        # Restore Queen promotion if a pawn moves to the 8th or 1st rank
        piece = board.piece_at(best_real_move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(best_real_move.to_square)
            if (board.turn == chess.WHITE and to_rank == 7) or (board.turn == chess.BLACK and to_rank == 0):
                best_real_move = chess.Move(
                    best_real_move.from_square,
                    best_real_move.to_square,
                    promotion=chess.QUEEN,
                )

        # Sanity check legality
        if best_real_move not in board.legal_moves:
            for lm in board.legal_moves:
                if lm.from_square == best_real_move.from_square and lm.to_square == best_real_move.to_square:
                    best_real_move = lm
                    break

        if best_real_move not in board.legal_moves:
            best_real_move = list(board.legal_moves)[0]

        return best_real_move, best_prob


def score_to_cp(score_obj: chess.engine.Score, mate_cp: int = 10000) -> int:
    """Convert chess.engine.Score (centipawns or mate) to a bounded integer."""
    cp = score_obj.score(mate_score=mate_cp)
    return cp if cp is not None else 0


def seed_from_polyglot_book(
    board: chess.Board,
    max_plies: int = 6,
    temperature: float = 1.2,
) -> None:
    """Advance board using authentic Polyglot opening lines sampled with temperature."""
    book_candidates = [
        PROJECT_ROOT / "data" / "books" / "titans.bin",
        PROJECT_ROOT / "data" / "books" / "performance.bin",
    ]
    valid_books = [b for b in book_candidates if b.exists()]
    if not valid_books:
        return

    chosen_book = random.choice(valid_books)
    try:
        with chess.polyglot.open_reader(str(chosen_book)) as reader:
            for _ in range(max_plies):
                entries = list(reader.find_all(board))
                if not entries:
                    break
                if temperature <= 0.05:
                    weights = [max(1, int(e.weight)) for e in entries]
                    chosen = entries[np.argmax(weights)]
                else:
                    raw_weights = np.array(
                        [max(1, int(e.weight)) for e in entries], dtype=float
                    )
                    scaled = np.power(raw_weights, 1.0 / max(0.1, temperature))
                    probs = scaled / scaled.sum()
                    chosen = np.random.choice(entries, p=probs)
                board.push(chosen.move)
    except Exception:
        pass


def mine_blunders(
    games: int = 50,
    depth: int = 10,
    threshold: int = 100,
    min_ply: int = 1,
    max_plies: int = 40,
    out_csv: str = "data/mined_blunders.csv",
    stockfish_path: str | None = None,
    seed_puzzles_csv: str | None = None,
    opening_temp: float = 1.2,
):
    sf_executable = find_stockfish(stockfish_path)
    print(f"Grandmaster Coach (Stockfish) initialized at: {sf_executable}")

    inference_engine = ModelInference()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    puzzle_fens = []
    if seed_puzzles_csv and Path(seed_puzzles_csv).exists():
        df_p = pd.read_csv(seed_puzzles_csv)
        puzzle_fens = df_p["FEN"].tolist()
        print(f"Loaded {len(puzzle_fens)} seed positions from {seed_puzzles_csv}")

    blunders = []
    total_positions_evaluated = 0
    start_time = time.time()

    with chess.engine.SimpleEngine.popen_uci(sf_executable) as engine:
        engine.configure({"Threads": max(1, os.cpu_count() // 2), "Hash": 64})

        for g in range(games):
            try:
                board = chess.Board()

                # Generate diverse opening positions to prevent dataset echo chambers:
                # 20% tactical puzzles, 55% temperature-sampled Polyglot GM lines (tau=1.2), 25% off-beat random sidelines
                roll = random.random()
                if puzzle_fens and roll < 0.20:
                    board = chess.Board(random.choice(puzzle_fens))
                elif roll < 0.75:
                    seed_from_polyglot_book(
                        board, max_plies=random.randint(4, 8), temperature=opening_temp
                    )
                    if len(board.move_stack) < 2:
                        for _ in range(random.randint(2, 4)):
                            if board.is_game_over() or not list(board.legal_moves):
                                break
                            board.push(random.choice(list(board.legal_moves)))
                else:
                    # Off-beat opening perturbation: inject random legal moves
                    for _ in range(random.randint(3, 6)):
                        if board.is_game_over() or not list(board.legal_moves):
                            break
                        board.push(random.choice(list(board.legal_moves)))

                game_blunders = 0
                for ply in range(1, max_plies + 1):
                    if board.is_game_over() or not list(board.legal_moves):
                        break

                    active_turn = board.turn
                    fen_before = board.fen()

                    # 1. Stockfish Pre-Evaluation & Best Move
                    limit = chess.engine.Limit(depth=depth)
                    info_before = engine.analyse(board, limit)
                    score_before_obj = info_before["score"].pov(active_turn)
                    score_before_cp = score_to_cp(score_before_obj)

                    best_move = info_before.get("pv", [None])[0]
                    if not best_move:
                        result = engine.play(board, limit)
                        best_move = result.move

                    # 2. SE-ResNet Model Decision
                    model_move, confidence = inference_engine.predict_move(board)
                    if not model_move:
                        break

                    total_positions_evaluated += 1

                    # 3. Blunder Trap Verification
                    if model_move == best_move:
                        board.push(model_move)
                        continue

                    # Evaluate position after model's non-optimal move
                    test_board = board.copy()
                    test_board.push(model_move)
                    info_after = engine.analyse(test_board, limit)
                    score_after_obj = info_after["score"].pov(active_turn)
                    score_after_cp = score_to_cp(score_after_obj)

                    eval_drop = score_before_cp - score_after_cp

                    if eval_drop >= threshold and ply >= min_ply:
                        game_blunders += 1
                        blunder_record = {
                            "PuzzleId": f"blunder_{len(blunders)+1:06d}",
                            "FEN": fen_before,
                            "Moves": best_move.uci(),
                            "DropCp": eval_drop,
                            "ModelMove": model_move.uci(),
                            "Confidence": round(confidence, 4),
                            "Game": g + 1,
                            "Ply": ply,
                        }
                        blunders.append(blunder_record)

                        print(
                            f"[BLUNDER #{len(blunders)}] Game {g+1:02d} Ply {ply:02d} | "
                            f"Drop: -{eval_drop:4d} cp | Model: {model_move.uci()} ({confidence*100:.1f}%) | "
                            f"Stockfish: {best_move.uci()} | FEN: {fen_before[:45]}..."
                        )

                    # Advance game board with model_move
                    board.push(model_move)

                print(
                    f"--> Completed Game {g+1}/{games} | Blunders Mined: {game_blunders} "
                    f"| Total Harvested: {len(blunders)}"
                )
            except Exception as ex:
                print(f"[WARN] Skipping game {g+1} due to engine encounter: {ex}")
                continue

    elapsed = time.time() - start_time
    print(
        f"\nMining Finished in {elapsed:.1f}s | "
        f"Evaluated: {total_positions_evaluated} positions | "
        f"Total Blunders Mined: {len(blunders)}"
    )

    if blunders:
        df_blunders = pd.DataFrame(blunders)
        if out_path.exists():
            df_existing = pd.read_csv(out_path)
            df_combined = pd.concat([df_existing, df_blunders], ignore_index=True)
            df_combined.drop_duplicates(subset=["FEN"], inplace=True)
            df_combined.to_csv(out_path, index=False)
            print(f"Appended to {out_path} (Total Unique Blunders: {len(df_combined)})")
        else:
            df_blunders.to_csv(out_path, index=False)
            print(f"Saved {len(df_blunders)} blunder records to {out_path}")
    else:
        print("No blunders met the threshold criteria.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Blunder Mining using Stockfish Distillation"
    )
    parser.add_argument(
        "--games", type=int, default=50, help="Number of simulated games"
    )
    parser.add_argument(
        "--depth", type=int, default=10, help="Stockfish search depth"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Centipawn evaluation drop threshold for blunder detection",
    )
    parser.add_argument(
        "--min_ply",
        type=int,
        default=1,
        help="Minimum half-moves (plies) before recording blunders (e.g. 15 for middlegame)",
    )
    parser.add_argument(
        "--max_plies",
        "--max_ply",
        dest="max_plies",
        type=int,
        default=40,
        help="Maximum half-moves (plies) per simulated game",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "mined_blunders.csv"),
        help="Output CSV file path for mined blunders",
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default=None,
        help="Custom path to Stockfish binary",
    )
    parser.add_argument(
        "--seed_puzzles",
        type=str,
        default=str(PROJECT_ROOT / "data" / "mate_puzzles.csv"),
        help="Path to puzzles CSV for seeding varied tactical states",
    )
    parser.add_argument(
        "--opening_temp",
        type=float,
        default=1.2,
        help="Temperature for opening book sampling to maximize opening diversity (default: 1.2)",
    )

    args = parser.parse_args()
    mine_blunders(
        games=args.games,
        depth=args.depth,
        threshold=args.threshold,
        min_ply=args.min_ply,
        max_plies=args.max_plies,
        out_csv=args.out,
        stockfish_path=args.stockfish_path,
        seed_puzzles_csv=args.seed_puzzles,
        opening_temp=args.opening_temp,
    )

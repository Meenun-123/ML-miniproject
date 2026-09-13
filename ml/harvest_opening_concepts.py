#!/usr/bin/env python3
"""
Chess Mate Vision - Opening Mastery Dataset Generator
Extracts strategic opening positions between plies 6 and 24 from high-Elo master games / Polyglot books.
Teaches the SE-ResNet policy and value heads quiet, harmonious opening principles
(center tension, minor piece development, King safety, castling timing)
to eliminate "Book Amnesia".
"""

import argparse
import random
import sys
from pathlib import Path
import chess
import chess.pgn
import chess.polyglot
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOOK_PATH = PROJECT_ROOT / "data" / "books" / "titans.bin"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "opening_concepts.csv"


def harvest_from_pgn(pgn_file_path: str, max_games: int = 1500) -> pd.DataFrame:
    """
    Extracts positions specifically between plies 6 and 24 from high-Elo master PGN games.
    """
    records = []
    seen_fens = set()

    print(f"📖 Parsing PGN file: {pgn_file_path} (target: up to {max_games} games)...")
    with open(pgn_file_path, "r", encoding="utf-8", errors="ignore") as pgn:
        games_read = 0
        while games_read < max_games:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            games_read += 1

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                ply += 1
                if 6 <= ply <= 24:
                    fen = board.fen()
                    if fen not in seen_fens:
                        seen_fens.add(fen)
                        records.append({
                            "FEN": fen,
                            "Moves": move.uci(),
                            "Value": 0.0,
                            "Ply": ply,
                            "Theme": "opening_development",
                        })
                board.push(move)

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} strategic opening positions from {games_read} PGN games.")
    return df


def harvest_from_polyglot_books(
    book_path: Path,
    num_trajectories: int = 1200,
    min_ply: int = 6,
    max_ply: int = 24,
) -> pd.DataFrame:
    """
    Harvests opening positions between min_ply and max_ply by sampling diverse grandmaster
    paths across the Polyglot tournament book (e.g. Titans / Performance) with fallback to
    classical strategic continuations.
    """
    print(f"📚 Harvesting opening trees from Polyglot book: {book_path}...")
    reader = chess.polyglot.open_reader(str(book_path))

    perf_path = book_path.parent / "performance.bin"
    perf_reader = chess.polyglot.open_reader(str(perf_path)) if perf_path.exists() else None

    records = []
    seen_fens = set()

    for traj_id in range(num_trajectories):
        board = chess.Board()
        ply = 0

        while not board.is_game_over() and ply < max_ply:
            ply += 1

            entries = list(reader.find_all(board))
            if not entries and perf_reader:
                entries = list(perf_reader.find_all(board))

            if entries:
                weights = [max(1, e.weight) for e in entries]
                chosen_entry = random.choices(entries, weights=weights, k=1)[0]
                chosen_move = chosen_entry.move
            else:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                def score_opening_move(m: chess.Move) -> float:
                    s = random.uniform(0.0, 1.0)
                    if board.is_castling(m):
                        s += 5.0
                    if m.to_square in [chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.C5]:
                        s += 3.0
                    p = board.piece_at(m.from_square)
                    if p and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                        s += 2.0
                    if board.is_capture(m):
                        s += 1.5
                    return s

                chosen_move = max(legal_moves, key=score_opening_move)

            if min_ply <= ply <= max_ply:
                fen = board.fen()
                if fen not in seen_fens:
                    seen_fens.add(fen)
                    records.append({
                        "FEN": fen,
                        "Moves": chosen_move.uci(),
                        "Value": 0.0,
                        "Ply": ply,
                        "Theme": "opening_development",
                    })

            board.push(chosen_move)

    reader.close()
    if perf_reader:
        perf_reader.close()

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} strategic opening positions across {num_trajectories} trajectories.")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Harvest strategic opening concept dataset (plies 6-24) to eliminate Book Amnesia"
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="Optional path to master PGN file for extraction",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=str(DEFAULT_BOOK_PATH),
        help=f"Path to polyglot opening book (default: {DEFAULT_BOOK_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=1200,
        help="Number of diverse grandmaster trajectories to sample if using Polyglot (default: 1200)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pgn and Path(args.pgn).exists():
        df = harvest_from_pgn(args.pgn)
    else:
        book_file = Path(args.book)
        if not book_file.exists():
            print(f"Error: Polyglot book not found at {book_file}", file=sys.stderr)
            sys.exit(1)
        df = harvest_from_polyglot_books(
            book_file, num_trajectories=args.trajectories
        )

    df.to_csv(out_path, index=False)
    print(f"✅ Saved opening concepts dataset to {out_path} ({len(df)} samples).")


if __name__ == "__main__":
    main()

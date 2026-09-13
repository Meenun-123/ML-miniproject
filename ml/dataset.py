import random
from typing import Tuple

import chess
import pandas as pd
import torch


def encode_move(move: chess.Move) -> int:
    """Encode a chess.Move into an integer in range [0, 4095]."""
    return move.from_square * 64 + move.to_square


def decode_move(idx: int) -> chess.Move:
    """Decode an integer in range [0, 4095] into a chess.Move."""
    return chess.Move(idx // 64, idx % 64)


def flip_square_horizontal(sq: int) -> int:
    """Reflect square horizontally across files: file x -> 7 - x."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    return chess.square(7 - file, rank)


def flip_move_horizontal(move: chess.Move) -> chess.Move:
    """Reflect a move horizontally (file inversion)."""
    return chess.Move(
        flip_square_horizontal(move.from_square),
        flip_square_horizontal(move.to_square),
        promotion=move.promotion,
    )


def real_move_to_canonical(move: chess.Move, is_black: bool) -> chess.Move:
    """
    Map move from real board coordinates to canonical perspective coordinates.
    If Black to move, ranks are mirrored vertically via chess.square_mirror.
    """
    if is_black:
        return chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move


def canonical_move_to_real(move: chess.Move, is_black: bool) -> chess.Move:
    """
    Map move from canonical perspective back to real board coordinates.
    chess.square_mirror is an involution (f(f(x)) == x).
    """
    if is_black:
        return chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to an enriched 18-plane (18, 8, 8) float32 tensor.

    Canonical Perspective:
    - If Black to move, vertically flip board geometry and swap colors so the
      network ALWAYS observes the position from the active player's perspective ("White moving forward").

    Channels:
    - 0-5: Active player's pieces [P, N, B, R, Q, K]
    - 6-11: Opponent's pieces [p, n, b, r, q, k]
    - 12: Active player's attack map (binary 1 if attacked by active player)
    - 13: Opponent's attack map (binary 1 if attacked by opponent)
    - 14: Pinned pieces plane (binary 1 if active piece is pinned to king)
    - 15: In-check indicator plane (full 1s if active player's king in check, else 0s)
    - 16: Active player castling availability (kingside: cols 4-7, queenside: cols 0-3)
    - 17: Opponent castling availability (kingside: cols 4-7, queenside: cols 0-3)
    """
    # Canonical perspective: active player is always White in b
    if board.turn == chess.BLACK:
        b = board.mirror()
    else:
        b = board

    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)

    # Channels 0-5: Active player pieces [P, N, B, R, Q, K]
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    for ch, pt in enumerate(piece_types):
        for sq in b.pieces(pt, chess.WHITE):
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            tensor[ch, row, col] = 1.0

    # Channels 6-11: Opponent pieces [p, n, b, r, q, k]
    for ch, pt in enumerate(piece_types, start=6):
        for sq in b.pieces(pt, chess.BLACK):
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            tensor[ch, row, col] = 1.0

    # Channels 12, 13, 14: Attack maps & pins
    for sq in chess.SQUARES:
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        if b.is_attacked_by(chess.WHITE, sq):
            tensor[12, row, col] = 1.0
        if b.is_attacked_by(chess.BLACK, sq):
            tensor[13, row, col] = 1.0
        if b.is_pinned(chess.WHITE, sq):
            tensor[14, row, col] = 1.0

    # Channel 15: In-check indicator plane
    if b.is_check():
        tensor[15, :, :] = 1.0

    # Channel 16: Active player castling availability
    if b.has_kingside_castling_rights(chess.WHITE):
        tensor[16, :, 4:] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        tensor[16, :, :4] = 1.0

    # Channel 17: Opponent castling availability
    if b.has_kingside_castling_rights(chess.BLACK):
        tensor[17, :, 4:] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        tensor[17, :, :4] = 1.0

    return tensor


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string into an 18-plane canonical float32 tensor."""
    board = chess.Board(fen)
    return board_to_tensor(board)


class ChessMateDataset(torch.utils.data.Dataset):
    """
    Dataset for chess tactical mate puzzles.
    Loads filtered Lichess CSV and produces (tensor, label, value_target).
    """

    def __init__(
        self,
        csv_path: str | pd.DataFrame,
        split: str = "train",
        augment: bool = True,
    ):
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()
        else:
            self.df = pd.read_csv(csv_path)
        self.split = split
        self.augment = augment if split == "train" else False

        n = len(self.df)
        train_end = int(0.8 * n)
        val_end = train_end + int(0.1 * n)

        if split == "train":
            self.df = self.df.iloc[:train_end]
        elif split == "val":
            self.df = self.df.iloc[train_end:val_end]
        elif split == "test":
            self.df = self.df.iloc[val_end:]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        row = self.df.iloc[idx]
        fen = row["FEN"]
        moves_str = row["Moves"]

        moves = moves_str.split()
        if len(moves) == 0:
            return self.__getitem__((idx + 1) % len(self))

        board = chess.Board(fen)
        try:
            if len(moves) == 1:
                solution_move = chess.Move.from_uci(moves[0])
            else:
                # Standard Lichess puzzle: moves[0] is setup move, moves[1] is solution
                setup_move = chess.Move.from_uci(moves[0])
                board.push(setup_move)
                solution_move = chess.Move.from_uci(moves[1])
        except (ValueError, chess.IllegalMoveError):
            return self.__getitem__((idx + 1) % len(self))

        is_black = board.turn == chess.BLACK

        # Convert solution move to canonical perspective
        canonical_move = real_move_to_canonical(solution_move, is_black)

        # Convert board to canonical 18-plane tensor
        tensor = board_to_tensor(board)

        # Horizontal mirror augmentation (50% probability during training)
        if self.augment and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[-1])
            canonical_move = flip_move_horizontal(canonical_move)

        label = encode_move(canonical_move)
        # Value target: Read custom float if present (e.g. 0.0 for quiet openings), default to 1.0 for checkmates
        if "Value" in row and not pd.isna(row["Value"]):
            value_target = float(row["Value"])
        else:
            value_target = 1.0

        return tensor, label, value_target


def filter_mate_puzzles(
    input_csv: str, output_csv: str, chunk_size: int = 50000
) -> int:
    """Stream Lichess puzzle CSV and extract mateIn1 and mateIn2 puzzles."""
    count = 0
    first_chunk = True

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        filtered = chunk[
            chunk["Themes"].str.contains("mateIn1|mateIn2", na=False)
        ]
        count += len(filtered)

        mode = "w" if first_chunk else "a"
        header = first_chunk
        filtered.to_csv(output_csv, mode=mode, header=header, index=False)
        first_chunk = False

    return count

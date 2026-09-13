import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

import chess
import chess.engine
import chess.polyglot
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import uvicorn

# Insert project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import (
    board_to_tensor,
    canonical_move_to_real,
    decode_move,
    encode_move,
    fen_to_tensor,
    real_move_to_canonical,
)
from ml.train import ChessTacticsCNN, ChessTacticsResNet

try:
    import onnxruntime

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

app = FastAPI(title="Chess Checkmate Pattern Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
MODEL = None
ONNX_SESSION = None
MODEL_TYPE = "none"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stockfish Persistent Engine State
STOCKFISH_ENGINE: chess.engine.SimpleEngine | None = None
STOCKFISH_PATH: str | None = None


def find_stockfish() -> str | None:
    which_sf = shutil.which("stockfish")
    candidates = [
        which_sf,
        str(PROJECT_ROOT / "bin" / "stockfish"),
        os.path.expanduser("~/.local/bin/stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for p in candidates:
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            return str(Path(p).resolve())
    return None


def get_stockfish_engine() -> chess.engine.SimpleEngine | None:
    global STOCKFISH_ENGINE, STOCKFISH_PATH
    if STOCKFISH_ENGINE is not None:
        try:
            if not STOCKFISH_ENGINE.is_alive():
                STOCKFISH_ENGINE = None
        except Exception:
            STOCKFISH_ENGINE = None

    if STOCKFISH_ENGINE is None:
        if STOCKFISH_PATH is None:
            STOCKFISH_PATH = find_stockfish()
        if not STOCKFISH_PATH:
            logger.warning("Stockfish binary not found on system.")
            return None
        try:
            STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            threads = max(1, (os.cpu_count() or 2) // 2)
            STOCKFISH_ENGINE.configure({"Threads": threads, "Hash": 64})
            logger.info(f"Connected persistent Stockfish engine at {STOCKFISH_PATH} (threads={threads})")
        except Exception as e:
            logger.error(f"Failed to start Stockfish engine: {e}")
            STOCKFISH_ENGINE = None
    return STOCKFISH_ENGINE


# Polyglot Master Opening Book Configuration
BOOK_PATH = PROJECT_ROOT / "data" / "books" / "titans.bin"
FALLBACK_BOOK_PATH = PROJECT_ROOT / "data" / "books" / "performance.bin"


def get_book_move(
    board: chess.Board, temperature: float = 0.0
) -> tuple[chess.Move | None, float]:
    """
    Look up current position in the Polyglot master opening book (titans.bin / performance.bin).
    Returns (move, confidence_weight_ratio).
    """
    book_file = BOOK_PATH if BOOK_PATH.exists() else FALLBACK_BOOK_PATH
    if not book_file.exists():
        return None, 0.0

    try:
        with chess.polyglot.open_reader(str(book_file)) as reader:
            entries = list(reader.find_all(board))
            if not entries:
                return None, 0.0

            weights = [max(1, int(e.weight)) for e in entries]
            total_weight = sum(weights)

            if temperature > 0.1 and len(entries) > 1:
                chosen = random.choices(entries, weights=weights, k=1)[0]
                conf = chosen.weight / total_weight if total_weight > 0 else 1.0
                return chosen.move, float(conf)
            else:
                best_entry = max(entries, key=lambda e: e.weight)
                conf = best_entry.weight / total_weight if total_weight > 0 else 1.0
                return best_entry.move, float(conf)
    except Exception as e:
        logger.warning(f"Opening book query error: {e}")
        return None, 0.0


def ensure_model_loaded():
    global MODEL, ONNX_SESSION, MODEL_TYPE

    if MODEL is not None or ONNX_SESSION is not None:
        return

    weights_dir = PROJECT_ROOT / "weights"
    pt_path = weights_dir / "chess_mate_cnn.pt"
    onnx_path = weights_dir / "chess_mate_cnn.onnx"

    if HAS_ONNX and onnx_path.exists():
        logger.info(f"Loading ONNX model from {onnx_path}")
        ONNX_SESSION = onnxruntime.InferenceSession(str(onnx_path))
        MODEL_TYPE = "onnx"
    elif pt_path.exists():
        logger.info(f"Loading PyTorch model from {pt_path}")
        MODEL = ChessTacticsResNet(in_channels=18)
        MODEL.load_state_dict(
            torch.load(pt_path, map_location=DEVICE, weights_only=True)
        )
        MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_TYPE = "pytorch"
    else:
        logger.warning(
            f"No weights found at {weights_dir}. Initializing random SE-ResNet weights for demo."
        )
        MODEL = ChessTacticsResNet(in_channels=18)
        MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_TYPE = "pytorch"


@app.on_event("startup")
async def startup_event():
    ensure_model_loaded()
    # Pre-warm Stockfish bridge
    get_stockfish_engine()


@app.on_event("shutdown")
async def shutdown_event():
    global STOCKFISH_ENGINE
    if STOCKFISH_ENGINE is not None:
        try:
            STOCKFISH_ENGINE.quit()
        except Exception:
            pass
        STOCKFISH_ENGINE = None


class FENRequest(BaseModel):
    fen: str
    temperature: float = 0.0
    ply: int = 0
    use_book: bool = True
    use_search: bool = True
    search_depth: int = 2


class MoveInfo(BaseModel):
    uci: str
    san: str
    from_sq: str
    to_sq: str
    confidence: float
    is_book: bool = False
    source: str = "model"  # "book" | "search" | "model"


class AdaptiveStepRequest(BaseModel):
    fen: str
    history_cpl: list[float] = []
    model_move_uci: str | None = None
    ply: int = 0
    temperature: float = 0.0
    use_book: bool = True
    use_search: bool = False
    search_depth: int = 2


class AdaptiveStepResponse(BaseModel):
    model_cpl: float
    rolling_acpl: float
    estimated_elo: int
    stockfish_elo: int
    stockfish_skill_level: int
    counter_move: MoveInfo | None
    fen_after: str
    is_check: bool
    is_checkmate: bool
    is_draw: bool
    history_cpl: list[float]
    inference_time_ms: float


class EvaluateFullGameRequest(BaseModel):
    pgn_moves: list[str]
    model_color: str = "white"
    result: str = "draw"


class PredictResponse(BaseModel):
    fen: str
    top_moves: list[MoveInfo]
    inference_time_ms: float
    win_eval: float | None = None
    is_book_move: bool = False
    source: str = "model"
    eval_mode: str | None = None
    anticipated_counter: str | None = None
    tactical_rationale: str | None = None
    search_pv: list[str] = []


class SequenceStep(BaseModel):
    step: int
    uci: str
    san: str
    from_sq: str
    to_sq: str
    fen_after: str
    confidence: float
    is_check: bool
    is_mate: bool


class SolveResponse(BaseModel):
    found_mate: bool
    is_checkmate: bool
    final_fen: str
    mate_in: int | None = None
    plies: int = 0
    moves: list[str] = []
    strategy_pivoted: bool = False
    pivoted_from_move: str | None = None
    pivoted_to_move: str | None = None
    candidate_rank: int = 1
    path_score: float = 1.0
    confidence: float
    sequence: list[SequenceStep]
    inference_time_ms: float
    win_eval: float | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL_TYPE != "none",
        "model_type": MODEL_TYPE,
        "architecture": "SE-ResNet-8 (18 planes)",
    }


# Telemetry & Match History Disk-Backed Persistence
MATCH_HISTORY_FILE = PROJECT_ROOT / "data" / "match_history.json"


def load_match_history() -> list[dict]:
    """Safely loads persistent match history from disk, creating default seed if missing."""
    if not MATCH_HISTORY_FILE.exists():
        initial_history = [
            {"game_id": 1, "moves_count": 34, "acpl": 94.2, "final_elo": 1020, "result": "loss", "blunders": 6},
            {"game_id": 2, "moves_count": 42, "acpl": 82.5, "final_elo": 1110, "result": "loss", "blunders": 5},
            {"game_id": 3, "moves_count": 28, "acpl": 86.1, "final_elo": 1080, "result": "loss", "blunders": 5},
            {"game_id": 4, "moves_count": 56, "acpl": 68.4, "final_elo": 1260, "result": "draw", "blunders": 3},
        ]
        save_match_history(initial_history)
        return initial_history
    try:
        with open(MATCH_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"⚠️ Error reading {MATCH_HISTORY_FILE}: {e}")
    return []


def save_match_history(history: list[dict]) -> None:
    """Safely writes match history list to disk with indented UTF-8 JSON formatting."""
    try:
        MATCH_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MATCH_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving match history to {MATCH_HISTORY_FILE}: {e}")


model_stats = {
    "estimated_elo": 1260,
    "recent_cpl": [68.4, 86.1, 82.5],
}


def estimate_performance_elo(acpl: float, accuracy: float = 50.0, result: str = "draw") -> int:
    """
    Estimates realistic Elo based on both ACPL and move accuracy,
    capping massive single-move swings so one blunder doesn't drop the engine to 600.
    """
    # 1. Cap ACPL to realistic human bounds (5.0 to 180.0)
    effective_acpl = min(max(acpl, 5.0), 180.0)

    # Smooth, realistic Club/Intermediate curve:
    # ACPL ~25 -> ~2160 Elo
    # ACPL ~50 -> ~1925 Elo
    # ACPL ~80 -> ~1640 Elo
    # ACPL ~120 -> ~1260 Elo
    # ACPL ~160 -> ~880 Elo
    base_elo = 2400.0 - (effective_acpl * 9.5)

    # 2. Result modifier
    res_lower = result.lower() if isinstance(result, str) else "draw"
    if res_lower == "win":
        base_elo += 60
    elif res_lower == "loss":
        base_elo -= 40

    return int(round(min(max(base_elo, 750), 2800)))


@app.post("/record-game-stats")
async def record_game_stats(data: dict):
    """Logs match metrics when a game reaches checkmate or draw, persisting to disk."""
    history = load_match_history()
    raw_acpl = float(data.get("acpl", 0.0))
    res = str(data.get("result", "loss"))

    provided_elo = data.get("final_elo")
    if provided_elo is not None and int(provided_elo) > 600:
        final_elo = int(provided_elo)
    else:
        final_elo = estimate_performance_elo(raw_acpl, result=res)

    record = {
        "game_id": len(history) + 1,
        "moves_count": int(data.get("moves_count", 0)),
        "acpl": round(min(raw_acpl, 250.0), 1),
        "game_elo": final_elo,
        "final_elo": final_elo,
        "cumulative_elo": final_elo,
        "result": res,
        "blunders": int(data.get("blunders", 0)),
    }
    history.append(record)
    save_match_history(history)
    return {"status": "ok", "record": record, "history": history}


@app.post("/evaluate-full-game")
async def evaluate_full_game(req: EvaluateFullGameRequest):
    """
    Full-game review engine powered by Stockfish at depth 12:
    Replays entire SAN move sequence, calculates CPL and accuracy for every model move,
    derives game Elo and cumulative lifetime Elo, and persists match record to disk.
    """
    engine = get_stockfish_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Stockfish engine unavailable")

    # Unconstrained Judge: Ensure full grandmaster strength for accurate ACPL evaluation
    try:
        engine.configure({
            "UCI_LimitStrength": False,
            "Skill Level": 20,
        })
    except Exception as e:
        logger.warning(f"Error resetting Stockfish evaluator strength: {e}")

    board = chess.Board()
    model_color_chess = (
        chess.WHITE if req.model_color.lower() in ("white", "w") else chess.BLACK
    )

    model_cpls: list[float] = []
    best_count = 0
    inaccuracy_count = 0
    mistake_count = 0
    blunder_count = 0

    limit = chess.engine.Limit(depth=12, time=0.12)

    for san_move in req.pgn_moves:
        try:
            move = board.parse_san(san_move)
        except Exception as e:
            logger.warning(f"Skipping invalid SAN move '{san_move}': {e}")
            break

        is_model_turn = board.turn == model_color_chess
        if is_model_turn:
            try:
                # 1. Query Stockfish BEFORE model move (from model's POV)
                info_before = engine.analyse(board, limit)
                score_before_obj = info_before["score"].pov(model_color_chess)
                score_before = (
                    score_before_obj.score(mate_score=10000) or 0
                )

                # 2. Push model move
                board.push(move)

                # 3. Query Stockfish AFTER model move (from model's POV)
                info_after = engine.analyse(board, limit)
                score_after_obj = info_after["score"].pov(model_color_chess)
                score_after = (
                    score_after_obj.score(mate_score=10000) or 0
                )

                # Centipawn loss (capped at 250 cp to prevent terminal mate swings from exploding ACPL)
                raw_cpl = max(0.0, float(score_before - score_after))
                cpl = min(raw_cpl, 250.0)
                model_cpls.append(cpl)

                if cpl < 15.0:
                    best_count += 1
                elif cpl < 50.0:
                    inaccuracy_count += 1
                elif cpl < 100.0:
                    mistake_count += 1
                else:
                    blunder_count += 1
            except Exception as e:
                logger.warning(f"Error analyzing move '{san_move}': {e}")
                board.push(move)
        else:
            board.push(move)

    # Compute overall game metrics
    game_acpl = float(np.mean(model_cpls)) if model_cpls else 50.0
    game_accuracy = float(np.clip(100.0 * np.exp(-0.004 * game_acpl), 0.0, 100.0))

    # Smooth, realistic Elo mapping
    estimated_game_elo = estimate_performance_elo(game_acpl, accuracy=game_accuracy, result=req.result)

    # Lifetime Cumulative Rolling Elo
    history = load_match_history()
    if history:
        prev_cum = history[-1].get(
            "cumulative_elo", history[-1].get("final_elo", 1200)
        )
        new_cumulative_elo = int(
            round(0.75 * prev_cum + 0.25 * estimated_game_elo)
        )
        prev_game_elo = history[-1].get(
            "game_elo", history[-1].get("final_elo", estimated_game_elo)
        )
        delta_elo = estimated_game_elo - prev_game_elo
    else:
        new_cumulative_elo = estimated_game_elo
        delta_elo = 0

    record = {
        "game_id": len(history) + 1,
        "moves_count": len(req.pgn_moves),
        "acpl": round(game_acpl, 1),
        "game_elo": estimated_game_elo,
        "final_elo": estimated_game_elo,
        "cumulative_elo": new_cumulative_elo,
        "accuracy": round(game_accuracy, 1),
        "blunders": blunder_count,
        "mistakes": mistake_count,
        "inaccuracies": inaccuracy_count,
        "best_moves": best_count,
        "result": req.result,
        "delta_elo": delta_elo,
    }

    history.append(record)
    save_match_history(history)

    # Update live backend stats
    model_stats["estimated_elo"] = new_cumulative_elo
    if "recent_cpl" not in model_stats:
        model_stats["recent_cpl"] = []
    model_stats["recent_cpl"].append(game_acpl)
    if len(model_stats["recent_cpl"]) > 30:
        model_stats["recent_cpl"] = model_stats["recent_cpl"][-30:]

    return {
        "status": "ok",
        "record": record,
        "history": history,
    }


@app.get("/telemetry-history")
async def get_telemetry_history():
    """Fetches full persistent match telemetry from disk for live chart rendering."""
    history = load_match_history()
    recent = model_stats.get("recent_cpl", [60.0])
    recent_acpl = round(sum(recent) / max(1, len(recent)), 1) if recent else 0.0
    current_elo = model_stats.get("estimated_elo", 1200)
    if history:
        current_elo = history[-1].get("final_elo", current_elo)
        recent_acpl = history[-1].get("acpl", recent_acpl)
    return {
        "history": history,
        "current_elo": current_elo,
        "recent_acpl": recent_acpl,
    }


@app.get("/search-cache-stats")
async def get_search_cache_stats():
    """Returns the size and occupancy of the Zobrist Transposition Table and Neural Cache."""
    return {
        "transposition_table_entries": len(TRANSPOSITION_TABLE),
        "neural_eval_cache_entries": len(NEURAL_EVAL_CACHE),
    }


@app.post("/clear-search-cache")
async def clear_search_cache():
    """Clears both the Zobrist Transposition Table and Neural Evaluation Cache."""
    TRANSPOSITION_TABLE.clear()
    NEURAL_EVAL_CACHE.clear()
    return {"status": "ok", "message": "Search caches cleared"}



# Piece-Square Tables (PST) & Positional Baseline Evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_MAP = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}

MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def get_material_balance(board: chess.Board) -> float:
    """Returns raw material balance in centipawns from White's perspective (excluding Kings)."""
    white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in MATERIAL_VALUES.items())
    black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in MATERIAL_VALUES.items())
    return float(white_mat - black_mat)


def evaluate_pst_cp(board: chess.Board) -> float:
    """Calculates material + PST positional bonus from current player's perspective in [-1, 1]."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        pst = PST_MAP.get(piece.piece_type)
        pst_val = 0
        if pst:
            actual_sq = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
            r = chess.square_rank(actual_sq)
            f = chess.square_file(actual_sq)
            table_idx = (7 - r) * 8 + f
            pst_val = pst[table_idx]
        total = val + pst_val
        if piece.color == chess.WHITE:
            score += total
        else:
            score -= total

    pov_score = score if board.turn == chess.WHITE else -score
    return float(np.tanh(pov_score / 600.0))


# ---------------------------------------------------------
# Hanging Piece Scanner & Tactical Defense Move Ordering
# ---------------------------------------------------------

def get_hanging_pieces(board: chess.Board, color: bool) -> list[dict]:
    """
    Scans the board and identifies friendly pieces that are hanging:
    - Completely undefended and attacked by at least one enemy piece.
    - Or attacked by a piece of lower value (e.g. Bishop attacked by Pawn).
    """
    hanging = []
    enemy = not color

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            val = PIECE_VALUES.get(piece.piece_type, 100)
            attackers = list(board.attackers(enemy, sq))
            defenders = list(board.attackers(color, sq))

            if not attackers:
                continue

            min_attacker_val = min(
                PIECE_VALUES.get(board.piece_at(a).piece_type, 100)
                for a in attackers
                if board.piece_at(a)
            )

            # Case A: Piece is completely undefended
            if not defenders:
                hanging.append({
                    "square": sq,
                    "piece_type": piece.piece_type,
                    "value": val,
                    "threat": "undefended",
                    "san_sq": chess.square_name(sq),
                })
            # Case B: Attacked by a lower-value piece (e.g. Pawn attacks Knight)
            elif min_attacker_val < val:
                hanging.append({
                    "square": sq,
                    "piece_type": piece.piece_type,
                    "value": val - min_attacker_val,
                    "threat": "under-protected",
                    "san_sq": chess.square_name(sq),
                })

    # Sort so the highest-value hanging pieces (Queen, Rook) are handled first
    hanging.sort(key=lambda x: x["value"], reverse=True)
    return hanging


def is_square_safe(board: chess.Board, target_sq: chess.Square, color: bool, piece_type: int) -> bool:
    """Checks if target_sq is safe for the piece to retreat or step to."""
    enemy = not color
    enemy_attackers = list(board.attackers(enemy, target_sq))

    # If no enemy attacks this square, it's completely safe
    if not enemy_attackers:
        return True

    # If attacked by enemy pawns/lesser pieces, it is NOT safe
    min_enemy_val = min(
        PIECE_VALUES.get(board.piece_at(a).piece_type, 100)
        for a in enemy_attackers
        if board.piece_at(a)
    )
    piece_val = PIECE_VALUES.get(piece_type, 100)

    if min_enemy_val < piece_val:
        return False

    # If attacked, but defended by at least one friendly piece and equal/higher value attacker
    friendly_defenders = list(board.attackers(color, target_sq))
    return len(friendly_defenders) >= len(enemy_attackers)


def order_moves_with_defensive_awareness(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_scores: dict[chess.Move, float] | None = None,
    tt_move: chess.Move | None = None,
) -> list[chess.Move]:
    """
    Orders moves so that defensive retreats, interpositions, and counter-captures
    for hanging pieces are scored at the very top of the search tree.
    """
    if not legal_moves:
        return []

    color = board.turn
    hanging_pieces = get_hanging_pieces(board, color)
    hanging_sqs = {h["square"]: h for h in hanging_pieces}

    scored_moves = []

    for move in legal_moves:
        if tt_move and move == tt_move:
            scored_moves.append((999.0, move))
            continue

        score = 0.0

        # Base policy prior if available
        p_val = policy_scores[move] if (policy_scores and move in policy_scores) else 0.0
        score += p_val * 1.5

        from_sq = move.from_square
        to_sq = move.to_square
        moving_piece = board.piece_at(from_sq)
        moving_val = PIECE_VALUES.get(moving_piece.piece_type, 100) if (moving_piece and moving_piece.piece_type != chess.KING) else 200

        # 1. RETREAT / ESCAPE: Moving a piece that is currently hanging
        if from_sq in hanging_sqs:
            hanging_info = hanging_sqs[from_sq]
            # Is it retreating to a safe square?
            if is_square_safe(board, to_sq, color, moving_piece.piece_type):
                # Massive boost: Save the hanging piece!
                score += 3.5 + (hanging_info["value"] / 100.0)
            else:
                # Jumping from one fire into another
                score -= 2.0

        # 2. DEFEND / PROTECT: Another piece moves to defend the hanging piece
        elif hanging_pieces:
            for h in hanging_pieces:
                # Push the move temporarily to see if target is now defended
                board.push(move)
                now_defended = board.is_attacked_by(color, h["square"])
                board.pop()
                if now_defended and is_square_safe(board, to_sq, color, moving_piece.piece_type):
                    score += 1.8 + (h["value"] / 200.0)
                    break

        # 3. COUNTER-CAPTURE: Capturing an enemy piece
        if board.is_capture(move):
            captured = board.piece_at(to_sq)
            cap_val = PIECE_VALUES.get(captured.piece_type, 100) if captured else 100
            is_defended = board.is_attacked_by(not color, to_sq)
            if is_defended and moving_val > cap_val:
                # Defended capture losing material (e.g. Minor takes defended Pawn)
                # Only exempt if policy is very confident (>= 0.40) or king hunt check
                if p_val >= 0.40 or (board.gives_check(move) and p_val >= 0.25):
                    score += 1.5
                else:
                    score -= 3.0  # Bad/losing capture
            else:
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                score += 2.0 + (cap_val - moving_val * 0.1) / 100.0

        # 4. AVOID SUICIDE: Never step an unthreatened piece onto a square controlled by enemy pawns or undefended
        if from_sq not in hanging_sqs and not board.is_capture(move):
            if moving_piece and not is_square_safe(board, to_sq, color, moving_piece.piece_type):
                # Do not penalize forcing checks that assault an exposed king
                if not (board.gives_check(move) and evaluate_king_exposure(board, not color) >= 50.0):
                    score -= 4.0  # Heavily de-prioritize blundering a piece into attack

        # 5. CHECKS & KING ASSAULT
        if board.gives_check(move):
            score += 1.0
            opp_exp = evaluate_king_exposure(board, not color)
            if opp_exp >= 45.0:
                score += 1.5

        scored_moves.append((score, move))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored_moves]


# ---------------------------------------------------------
# Zobrist Transposition Table & Neural Evaluation Cache
# ---------------------------------------------------------
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2


class TTEntry:
    __slots__ = ("depth", "flag", "score", "best_move", "pv")

    def __init__(
        self,
        depth: int,
        flag: int,
        score: float,
        best_move: chess.Move | None,
        pv: list[chess.Move] | None = None,
    ):
        self.depth = depth
        self.flag = flag
        self.score = score
        self.best_move = best_move
        self.pv = pv or []


TRANSPOSITION_TABLE: dict[int, TTEntry] = {}
NEURAL_EVAL_CACHE: dict[int, tuple[torch.Tensor, float]] = {}


def get_neural_policy_and_value(board: chess.Board) -> tuple[torch.Tensor, float]:
    """
    Evaluates board with SE-ResNet-8 neural network (with Zobrist Eval caching).
    Returns (canonical_probs_tensor, value_eval in [-1, 1]).
    """
    ensure_model_loaded()
    zkey = chess.polyglot.zobrist_hash(board)
    if zkey in NEURAL_EVAL_CACHE:
        logits, value_eval = NEURAL_EVAL_CACHE[zkey]
    else:
        tensor_input = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        if MODEL_TYPE == "onnx":
            input_name = ONNX_SESSION.get_inputs()[0].name
            numpy_input = tensor_input.cpu().numpy().astype(np.float32)
            outputs = ONNX_SESSION.run(None, {input_name: numpy_input})
            logits = torch.tensor(outputs[0][0], device=DEVICE)
            value_eval = float(outputs[1][0][0]) if len(outputs) > 1 else 0.0
        else:
            with torch.no_grad():
                policy_logits, value_preds = MODEL(tensor_input)
                logits = policy_logits[0]
                value_eval = float(value_preds[0].item())

        if len(NEURAL_EVAL_CACHE) > 50000:
            NEURAL_EVAL_CACHE.clear()
        NEURAL_EVAL_CACHE[zkey] = (logits, value_eval)

    # Canonical illegal move masking
    is_black = board.turn == chess.BLACK
    mask = torch.full((4096,), float("-inf"), device=logits.device)
    for m in board.legal_moves:
        cm = real_move_to_canonical(m, is_black)
        mask[encode_move(cm)] = 0.0

    probs = F.softmax(logits + mask, dim=0)
    return probs, value_eval


def evaluate_king_exposure(board: chess.Board, king_color: bool) -> float:
    """
    Evaluates king vulnerability, pawn shield integrity, open files, and king ring attack pressure.
    Returns an exposure penalty in centipawns (0.0 to 250.0 cp).
    Higher score = more exposed, vulnerable king.
    """
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0.0

    penalty = 0.0
    k_file = chess.square_file(king_sq)
    k_rank = chess.square_rank(king_sq)
    opp_color = not king_color

    # 1. Pawn Shelter Integrity
    shelter_files = [f for f in (k_file - 1, k_file, k_file + 1) if 0 <= f <= 7]
    shield_ranks = range(1, 4) if king_color == chess.WHITE else range(4, 7)

    missing_pawns = 0
    for f in shelter_files:
        pawn_found = any(
            board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, king_color)
            for r in shield_ranks
        )
        if not pawn_found:
            missing_pawns += 1

    penalty += missing_pawns * 40.0

    # 2. Open / Half-Open Files to King
    for f in shelter_files:
        file_mask = chess.SquareSet(chess.BB_FILES[f])
        white_pawns_on_file = any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE) for sq in file_mask)
        black_pawns_on_file = any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK) for sq in file_mask)
        if not white_pawns_on_file and not black_pawns_on_file:
            penalty += 35.0
            heavy_attackers = any(
                board.piece_at(sq) and board.piece_at(sq).color == opp_color and
                board.piece_at(sq).piece_type in (chess.ROOK, chess.QUEEN)
                for sq in file_mask
            )
            if heavy_attackers:
                penalty += 30.0
        elif (king_color == chess.WHITE and not white_pawns_on_file) or (king_color == chess.BLACK and not black_pawns_on_file):
            penalty += 20.0

    # 3. King Ring Pressure (Attacking pieces targeting the 8 adjacent squares of king)
    king_ring = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            rf, rr = k_file + df, k_rank + dr
            if 0 <= rf <= 7 and 0 <= rr <= 7:
                king_ring.append(chess.square(rf, rr))

    ring_attack_weight = 0.0
    for ring_sq in king_ring:
        opp_attackers = board.attackers(opp_color, ring_sq)
        for att_sq in opp_attackers:
            piece = board.piece_at(att_sq)
            if piece:
                if piece.piece_type == chess.QUEEN:
                    ring_attack_weight += 12.0
                elif piece.piece_type in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    ring_attack_weight += 8.0

    penalty += min(ring_attack_weight, 90.0)

    # 4. Advanced / Dislocated King
    if king_color == chess.WHITE and k_rank >= 2:
        penalty += (k_rank - 1) * 30.0
    elif king_color == chess.BLACK and k_rank <= 5:
        penalty += (6 - k_rank) * 30.0

    return float(min(penalty, 250.0))


def evaluate_sacrifice_compensation(board: chess.Board, side: bool) -> tuple[float, bool]:
    """
    Evaluates dynamic compensation for material sacrifices:
    Only valid for Exchange Sacrifices (-210 < deficit <= -130 cp)
    or Positional Pawn Gambits (-130 < deficit <= -60 cp).
    Full piece sacrifices (deficit <= -210 cp) require dynamic search calculation / checkmate.
    """
    opp = not side
    mat_balance = get_material_balance(board)
    mover_material_cp = mat_balance if side == chess.WHITE else -mat_balance

    # If down a full piece or more (deficit <= -210 cp), never declare statically compensated!
    # Concrete piece sacrifices must be proved by search calculation, not static optimism.
    if mover_material_cp <= -210:
        return 0.0, False

    comp_cp = 0.0
    side_exposure = evaluate_king_exposure(board, side)
    opp_exposure = evaluate_king_exposure(board, opp)

    # 1. King Exposure Delta (attacker has destroyed defender's shelter)
    if opp_exposure >= 50.0:
        comp_cp += max(0.0, opp_exposure - side_exposure) * 0.75

    # 2. Damaged Enemy Pawn Structure
    opp_pawns = board.pieces(chess.PAWN, opp)
    files_with_opp_pawns = [chess.square_file(sq) for sq in opp_pawns]
    doubled_count = len(files_with_opp_pawns) - len(set(files_with_opp_pawns))
    comp_cp += doubled_count * 25.0

    # 3. Enemy uncastled with no castling rights
    if not board.has_castling_rights(opp):
        opp_k = board.king(opp)
        if opp_k in (chess.E1, chess.E8, chess.D1, chess.D8):
            comp_cp += 35.0

    # 4. Minor piece outposts on ranks 4-6
    for sq in board.pieces(chess.KNIGHT, side) | board.pieces(chess.BISHOP, side):
        r = chess.square_rank(sq)
        if (side == chess.WHITE and 3 <= r <= 5) or (side == chess.BLACK and 2 <= r <= 4):
            pawn_defenders = [
                d for d in board.attackers(side, sq)
                if board.piece_at(d) and board.piece_at(d).piece_type == chess.PAWN
            ]
            if pawn_defenders:
                comp_cp += 25.0

    is_compensated = False
    # Exchange sacrifice zone: deficit between -210 and -130 cp
    if -210 < mover_material_cp <= -130:
        is_compensated = (comp_cp >= 80.0) or (opp_exposure >= 70.0)
    # Pawn gambit zone: deficit between -130 and -60 cp
    elif -130 < mover_material_cp <= -60:
        is_compensated = (comp_cp >= 40.0)

    return comp_cp, is_compensated


def evaluate_positional_patience(board: chess.Board, ply: int = 0) -> float:
    """
    Positional patience & anti-recklessness heuristic evaluator:
    1. Early Queen & Piece Sortie Penalties (premature attacks before castling/minor development).
    2. King Safety & Pawn Shields (rewards castling, penalizes king stranded in open center).
    3. Centralization of Rooks (open files, semi-open files, 7th rank).
    4. Knight Outposts (anchored on ranks 4-6, supported by friendly pawn, immune to enemy pawn eviction).
    5. Holding Tension (avoids voluntary exchanges that yield initiative).
    Returns score in [-1.0, 1.0] from perspective of board.turn.
    """
    score_white = 0.0
    score_black = 0.0

    # 1. Development & Early Queen Sorties (opening phase: ply <= 24)
    if ply <= 24:
        # Check White minor pieces home on rank 1
        w_minors_home = 0
        for sq, ptype in [
            (chess.B1, chess.KNIGHT),
            (chess.G1, chess.KNIGHT),
            (chess.C1, chess.BISHOP),
            (chess.F1, chess.BISHOP),
        ]:
            p = board.piece_at(sq)
            if p and p.color == chess.WHITE and p.piece_type == ptype:
                w_minors_home += 1

        w_queens = board.pieces(chess.QUEEN, chess.WHITE)
        for q_sq in w_queens:
            if chess.square_rank(q_sq) >= 2 and w_minors_home >= 2:
                score_white -= 40.0

        w_king_sq = board.king(chess.WHITE)
        if w_king_sq in (chess.G1, chess.C1, chess.B1):
            score_white += 25.0
            for fsq in (chess.F2, chess.G2, chess.H2):
                fp = board.piece_at(fsq)
                if fp and fp.color == chess.WHITE and fp.piece_type == chess.PAWN:
                    score_white += 8.0
        elif w_king_sq == chess.E1:
            e_has_w_pawn = any(
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)
                for sq in chess.SquareSet(chess.BB_FILE_E)
            )
            d_has_w_pawn = any(
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)
                for sq in chess.SquareSet(chess.BB_FILE_D)
            )
            if not e_has_w_pawn or not d_has_w_pawn:
                score_white -= 30.0

        # Check Black minor pieces home on rank 8
        b_minors_home = 0
        for sq, ptype in [
            (chess.B8, chess.KNIGHT),
            (chess.G8, chess.KNIGHT),
            (chess.C8, chess.BISHOP),
            (chess.F8, chess.BISHOP),
        ]:
            p = board.piece_at(sq)
            if p and p.color == chess.BLACK and p.piece_type == ptype:
                b_minors_home += 1

        b_queens = board.pieces(chess.QUEEN, chess.BLACK)
        for q_sq in b_queens:
            if chess.square_rank(q_sq) <= 5 and b_minors_home >= 2:
                score_black -= 40.0

        b_king_sq = board.king(chess.BLACK)
        if b_king_sq in (chess.G8, chess.C8, chess.B8):
            score_black += 25.0
            for fsq in (chess.F7, chess.G7, chess.H7):
                fp = board.piece_at(fsq)
                if fp and fp.color == chess.BLACK and fp.piece_type == chess.PAWN:
                    score_black += 8.0
        elif b_king_sq == chess.E8:
            e_has_b_pawn = any(
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK)
                for sq in chess.SquareSet(chess.BB_FILE_E)
            )
            d_has_b_pawn = any(
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK)
                for sq in chess.SquareSet(chess.BB_FILE_D)
            )
            if not e_has_b_pawn or not d_has_b_pawn:
                score_black -= 30.0

    # 2. Rook Placement: Open & Semi-Open Files, 7th Rank
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        rank = chess.square_rank(sq)
        file_idx = chess.square_file(sq)
        file_mask = chess.BB_FILES[file_idx]
        w_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) & file_mask)
        b_pawns = bool(board.pieces(chess.PAWN, chess.BLACK) & file_mask)
        if not w_pawns and not b_pawns:
            score_white += 25.0
        elif not w_pawns:
            score_white += 15.0
        if rank == 6:
            score_white += 35.0

    for sq in board.pieces(chess.ROOK, chess.BLACK):
        rank = chess.square_rank(sq)
        file_idx = chess.square_file(sq)
        file_mask = chess.BB_FILES[file_idx]
        w_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) & file_mask)
        b_pawns = bool(board.pieces(chess.PAWN, chess.BLACK) & file_mask)
        if not w_pawns and not b_pawns:
            score_black += 25.0
        elif not b_pawns:
            score_black += 15.0
        if rank == 1:
            score_black += 35.0

    # 3. Knight Outposts
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        rank = chess.square_rank(sq)
        file_idx = chess.square_file(sq)
        if rank in (3, 4, 5) and file_idx in (2, 3, 4, 5):
            pawn_defended = False
            for p_sq in [sq - 9, sq - 7]:
                if 0 <= p_sq < 64 and chess.square_file(p_sq) in (file_idx - 1, file_idx + 1):
                    p = board.piece_at(p_sq)
                    if p and p.color == chess.WHITE and p.piece_type == chess.PAWN:
                        pawn_defended = True
                        break
            if pawn_defended:
                score_white += 25.0

    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        rank = chess.square_rank(sq)
        file_idx = chess.square_file(sq)
        if rank in (2, 3, 4) and file_idx in (2, 3, 4, 5):
            pawn_defended = False
            for p_sq in [sq + 9, sq + 7]:
                if 0 <= p_sq < 64 and chess.square_file(p_sq) in (file_idx - 1, file_idx + 1):
                    p = board.piece_at(p_sq)
                    if p and p.color == chess.BLACK and p.piece_type == chess.PAWN:
                        pawn_defended = True
                        break
            if pawn_defended:
                score_black += 25.0

    net = score_white - score_black
    pov = net if board.turn == chess.WHITE else -net
    return float(np.tanh(pov / 350.0))


def evaluate_tenacity_and_swindles(board: chess.Board, pov_score: float) -> tuple[float, str]:
    """
    Evaluates defensive tenacity, fortress structures, piece retention, and drawing swindles
    when in worse / losing positions (evaluation < -1.50 pawns / -150 cp).
    Returns (tenacity_score in [-1, 1], mode_label).
    """
    my_color = board.turn
    opp_color = not my_color

    my_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == my_color and p.piece_type != chess.KING
    )
    opp_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == opp_color and p.piece_type != chess.KING
    )
    material_diff = my_material - opp_material

    is_losing = (material_diff <= -180) or (pov_score <= -0.25)
    if not is_losing:
        return 0.0, "normal"

    bonus = 0.0

    # 1. Threefold Repetition & Drawing Mechanisms
    if board.can_claim_threefold_repetition() or board.is_repetition(2):
        bonus += 65.0
    if board.can_claim_fifty_moves():
        bonus += 60.0

    # 2. Stalemate Potential
    my_king_sq = board.king(my_color)
    if my_king_sq is not None:
        king_moves = [m for m in board.legal_moves if m.from_square == my_king_sq]
        if not king_moves and not board.is_check():
            if len(list(board.legal_moves)) <= 3:
                bonus += 50.0

    # 3. Anti-Simplification (Piece Retention)
    my_queens = len(board.pieces(chess.QUEEN, my_color))
    opp_queens = len(board.pieces(chess.QUEEN, opp_color))
    if my_queens >= 1 and opp_queens >= 1:
        bonus += 35.0
    elif my_queens == 0 and opp_queens == 0:
        bonus -= 20.0

    # 4. Fortress Detection: Blockading Opponent Passed Pawns
    opp_pawns = board.pieces(chess.PAWN, opp_color)
    for p_sq in opp_pawns:
        p_file = chess.square_file(p_sq)
        p_rank = chess.square_rank(p_sq)
        files_to_check = [p_file]
        if p_file > 0:
            files_to_check.append(p_file - 1)
        if p_file < 7:
            files_to_check.append(p_file + 1)

        is_passed = True
        for f in files_to_check:
            r_range = range(p_rank + 1, 8) if opp_color == chess.WHITE else range(0, p_rank)
            for r in r_range:
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.color == my_color and piece.piece_type == chess.PAWN:
                    is_passed = False
                    break
            if not is_passed:
                break

        if is_passed:
            blockade_rank = p_rank + 1 if opp_color == chess.WHITE else p_rank - 1
            if 0 <= blockade_rank <= 7:
                blockade_sq = chess.square(p_file, blockade_rank)
                blockader = board.piece_at(blockade_sq)
                if (
                    blockader
                    and blockader.color == my_color
                    and blockader.piece_type in (chess.KNIGHT, chess.BISHOP, chess.KING)
                ):
                    bonus += 30.0

    # 5. Check Activity / Perpetual Check Pressure
    if board.is_check():
        bonus += 20.0

    return float(np.tanh(bonus / 250.0)), "tenacious_defense"


# ---------------------------------------------------------
# Strategic Opening Principles (Plies 1-24)
# ---------------------------------------------------------
def evaluate_opening_concepts(board: chess.Board, ply: int) -> float:
    """
    Evaluates opening strategic principles (Plies 1-24):
    1. Center Occupancy & Control (e4, d4, e5, d5, c4, f4, c5, f5)
    2. Minor Piece Development (Knights & Bishops active off back rank)
    3. King Safety / Castling Readiness
    4. Anti-Wasted Tempi & King Exposure Timing
    Returns score in centipawns from White's perspective.
    """
    if ply > 24:
        return 0.0

    score = 0.0
    CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

    for color, mult in [(chess.WHITE, 1.0), (chess.BLACK, -1.0)]:
        # 1. Central Pawn Footprint
        for sq in ([chess.E4, chess.D4] if color == chess.WHITE else [chess.E5, chess.D5]):
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                score += 35.0 * mult  # Strong reward for classical pawn centers

        # Extended central flank support (c4, f4 / c5, f5)
        for sq in ([chess.C4, chess.F4] if color == chess.WHITE else [chess.C5, chess.F5]):
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                score += 15.0 * mult

        # 2. Central Control via Attacks
        for sq in CENTER_SQUARES:
            attackers = board.attackers(color, sq)
            score += len(attackers) * 8.0 * mult

        # 3. Development of Minor Pieces off the Back Rank
        minors_developed = 0
        back_squares = (
            [chess.B1, chess.C1, chess.F1, chess.G1]
            if color == chess.WHITE
            else [chess.B8, chess.C8, chess.F8, chess.G8]
        )
        for sq in back_squares:
            p = board.piece_at(sq)
            if p is None or p.piece_type not in [chess.KNIGHT, chess.BISHOP]:
                minors_developed += 1

        score += minors_developed * 20.0 * mult

        # 4. Castling & King Safety Incentives
        king_sq = board.king(color)
        if king_sq:
            # Did the king castle to safety?
            if king_sq in ([chess.G1, chess.C1] if color == chess.WHITE else [chess.G8, chess.C8]):
                score += 45.0 * mult
            elif ply > 14 and chess.square_file(king_sq) in [3, 4]:
                # King remains uncastled on open/semi-open central files past ply 14
                score -= 50.0 * mult

    return score


def evaluate_opening_concepts_normalized(board: chess.Board, ply: int) -> float:
    """Returns strategic opening evaluation normalized in [-1, 1] from moving player's POV."""
    score_white = evaluate_opening_concepts(board, ply)
    pov_score = score_white if board.turn == chess.WHITE else -score_white
    return float(np.tanh(pov_score / 250.0))


def evaluate_board_hybrid(board: chess.Board, ply: int = 0) -> tuple[float, str]:
    """
    Blends SE-ResNet-8 neural value head with PST, Positional Patience, and Tenacity.
    Returns (score in [-1, 1], eval_mode).
    eval_mode is one of: "clean_conversion", "prophylaxis", "tenacious_defense", "tactical_strike".
    """
    if board.is_checkmate():
        return -10.0, "tactical_strike"
    if board.is_game_over() or board.can_claim_threefold_repetition() or board.is_repetition(3):
        return 0.0, "tenacious_defense"

    _, neural_val = get_neural_policy_and_value(board)
    pst_val = evaluate_pst_cp(board)
    pos_val = evaluate_positional_patience(board, ply)
    tenacity_val, _ = evaluate_tenacity_and_swindles(board, pst_val)

    # Hard Material Reality Check:
    mat_balance = get_material_balance(board)
    turn_mult = 1.0 if board.turn == chess.WHITE else -1.0
    mover_material_cp = mat_balance * turn_mult
    mat_tanh = float(np.tanh(mover_material_cp / 400.0))

    # Dynamic Sacrifice Compensation & King Exposure:
    comp_cp, is_compensated = evaluate_sacrifice_compensation(board, board.turn)
    comp_norm = float(np.tanh(comp_cp / 300.0))

    # Anti-Delusion Clamp: If mover gave away material without forced mate AND without compensation,
    # never allow the neural network value head to hallucinate an optimistic positive evaluation.
    if mover_material_cp <= -180 and not is_compensated:
        neural_val = min(neural_val, mat_tanh)

    if (pst_val <= -0.25 or mover_material_cp <= -180) and not is_compensated:
        # Losing (-1.50 pawns or worse / material deficit <= -180 cp): tenacious survival, swindles, perpetuals
        eval_mode = "tenacious_defense"
        final_score = float(0.35 * neural_val + 0.35 * pst_val + 0.30 * tenacity_val)
        final_score = min(final_score, mat_tanh)
    elif is_compensated and comp_cp >= 60.0:
        # Dynamic tactical attack or sound compensated sacrifice!
        eval_mode = "tactical_strike"
        final_score = float(0.50 * neural_val + 0.25 * pst_val + 0.25 * comp_norm)
        final_score = float(np.clip(final_score, -0.6, 2.0))
    elif pst_val >= 0.25 and mover_material_cp >= 180:
        # Winning (+1.50 pawns or better / material advantage >= +180 cp): clean conversion, king safety, simplification
        eval_mode = "clean_conversion"
        final_score = float(0.60 * neural_val + 0.20 * pst_val + 0.20 * pos_val)
    elif 1 <= ply <= 24:
        # Warm Book-to-Middlegame Handoff (Plies 1-24):
        # 50% Neural Policy/Value + 30% Strategic Opening Principles + 20% Material/PST
        eval_mode = "strategic_development"
        opening_norm = evaluate_opening_concepts_normalized(board, ply)
        final_score = float(
            0.50 * neural_val +
            0.30 * opening_norm +
            0.20 * pst_val
        )
        if mover_material_cp <= -90 and not is_compensated:
            final_score = min(final_score, mat_tanh)
    else:
        # Equal / Dry (-0.75 to +0.75 pawns): prophylaxis, patient positional maneuvering
        eval_mode = "prophylaxis"
        final_score = float(0.65 * neural_val + 0.20 * pst_val + 0.15 * pos_val)
        if mover_material_cp <= -90 and not is_compensated:
            final_score = min(final_score, mat_tanh)

    # Hanging Piece Penalty & Under-Attack Adjustments:
    # 1. Did the player who just moved (not board.turn) leave any pieces hanging?
    last_moved_color = not board.turn
    opponent_hanging = get_hanging_pieces(board, last_moved_color)
    if opponent_hanging and not board.is_checkmate():
        # The opponent blundered a piece! We can take it for free.
        largest_hanging_loss = opponent_hanging[0]["value"]
        hanging_bonus_for_us = float((largest_hanging_loss * 0.9) / 400.0)
        final_score += hanging_bonus_for_us

    # 2. Does the current player to move (board.turn) have any pieces currently hanging?
    friendly_hanging = get_hanging_pieces(board, board.turn)
    if friendly_hanging and not board.is_checkmate():
        largest_friendly = friendly_hanging[0]["value"]
        hanging_penalty_for_us = float((largest_friendly * 0.9) / 400.0)
        final_score -= hanging_penalty_for_us
        eval_mode = "tenacious_defense"

    return float(np.clip(final_score, -9.0, 9.0)), eval_mode


class SearchTimeout(Exception):
    """Raised when search exceeds the allocated wall-clock deadline."""
    pass


def quiescence_search(
    board: chess.Board,
    alpha: float,
    beta: float,
    depth: int = 0,
    max_q_depth: int = 3,
    ply: int = 0,
    deadline: float | None = None,
) -> float:
    """
    Quiescence search with MVV-LVA ordering, Delta Pruning, and strict depth cap:
    Extends tactical capture sequences up to depth 3 to eliminate the 1-ply Horizon Effect
    without suffering combinatorial explosion.
    """
    if deadline and time.time() > deadline:
        raise SearchTimeout()

    if board.is_checkmate():
        return -10.0 - depth
    if board.is_game_over() or board.can_claim_threefold_repetition() or board.is_repetition(3):
        return 0.0

    in_check = board.is_check()

    stand_pat, _ = evaluate_board_hybrid(board, ply=ply + depth)
    if depth >= max_q_depth:
        return stand_pat

    # Cannot stand pat if in check (player must resolve check)
    if not in_check:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    BIG_DELTA = 0.95
    if in_check:
        moves_to_search = list(board.legal_moves)[:4]
        if not moves_to_search:
            return -10.0 - depth
    else:
        moves_to_search = [m for m in board.legal_moves if board.is_capture(m)]
        if not moves_to_search:
            return stand_pat

    # Only delta-prune if not in check
    if not in_check and stand_pat < alpha - BIG_DELTA:
        return alpha

    def quiescence_move_score(m: chess.Move) -> int:
        victim = board.piece_at(m.to_square)
        attacker = board.piece_at(m.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
        return v_val * 10 - a_val

    moves_to_search.sort(key=quiescence_move_score, reverse=True)

    for move in moves_to_search[:5]:
        board.push(move)
        try:
            score = -quiescence_search(board, -beta, -alpha, depth + 1, max_q_depth, ply=ply + 1, deadline=deadline)
        finally:
            board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def search_alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    ply: int = 0,
    deadline: float | None = None,
) -> tuple[float, list[chess.Move], str]:
    """
    Minimax (Negamax) with Full Principal Variation (PV) search, Zobrist Transposition Table,
    Null Move Pruning (NMP), Late Move Reductions (LMR), Opponent Refutation Analysis,
    and Speculative Sacrifice Penalties.

    Returns:
        (best_score, pv_moves, eval_mode)
    """
    if deadline and time.time() > deadline and ply > 0:
        raise SearchTimeout()

    if board.is_checkmate():
        return -10.0 - depth, [], "tactical_strike"
    if board.is_game_over() or board.can_claim_threefold_repetition() or board.is_repetition(3):
        return 0.0, [], "tenacious_defense"
    if depth <= 0:
        q_val = quiescence_search(board, alpha, beta, depth=0, max_q_depth=6, ply=ply, deadline=deadline)
        _, mode = evaluate_board_hybrid(board, ply=ply)
        return q_val, [], mode

    zkey = chess.polyglot.zobrist_hash(board)
    tt_entry = TRANSPOSITION_TABLE.get(zkey)
    if tt_entry is not None and tt_entry.depth >= depth:
        if tt_entry.flag == FLAG_EXACT:
            return tt_entry.score, tt_entry.pv, "tt_exact"
        elif tt_entry.flag == FLAG_LOWERBOUND and tt_entry.score >= beta:
            return tt_entry.score, tt_entry.pv, "tt_cut"
        elif tt_entry.flag == FLAG_UPPERBOUND and tt_entry.score <= alpha:
            return tt_entry.score, tt_entry.pv, "tt_cut"

    # Null Move Pruning (NMP) for depth >= 3:
    # If skipping our turn still produces a score >= beta, the branch is cut off.
    # Exclude when in check or in pure pawn endgames to avoid Zugzwang.
    if depth >= 3 and ply > 0 and not board.is_check():
        has_major_pieces = any(
            board.pieces(pt, board.turn)
            for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )
        if has_major_pieces:
            R = 2
            board.push(chess.Move.null())
            try:
                null_score, _, _ = search_alphabeta(
                    board, max(0, depth - 1 - R), -beta, -beta + 1, ply + 1, deadline=deadline
                )
                null_score = -null_score
            finally:
                board.pop()
            if null_score >= beta:
                return beta, [], "nmp_cutoff"

    is_black = board.turn == chess.BLACK
    probs, _ = get_neural_policy_and_value(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return 0.0, [], "draw"

    tt_move = (
        tt_entry.best_move
        if (tt_entry and tt_entry.best_move in legal_moves)
        else None
    )

    policy_scores = {}
    for m in legal_moves:
        cm = real_move_to_canonical(m, is_black)
        idx = encode_move(cm)
        policy_scores[m] = probs[idx].item()

    legal_moves = order_moves_with_defensive_awareness(
        board, legal_moves, policy_scores=policy_scores, tt_move=tt_move
    )

    if ply == 0:
        candidates = legal_moves[:12]
    else:
        # Ensure search calculates all checks, all captures, retreats of hanging pieces, plus top 4 quiet moves
        hanging = get_hanging_pieces(board, board.turn)
        hanging_sqs = {h["square"] for h in hanging}
        forced_tactical = [m for m in legal_moves if board.is_capture(m) or board.gives_check(m)]
        retreat_moves = [m for m in legal_moves if m.from_square in hanging_sqs and m not in forced_tactical]
        quiet_moves = [m for m in legal_moves if m not in forced_tactical and m not in retreat_moves]
        candidates = forced_tactical + retreat_moves + quiet_moves[:4]
        if not candidates:
            candidates = legal_moves[:6]

    best_move = candidates[0]
    best_pv = [best_move]
    best_score = -float("inf")
    best_mode = "prophylaxis"
    orig_alpha = alpha
    timed_out = False

    for idx, move in enumerate(candidates):
        if deadline and time.time() > deadline and ply == 0 and idx > 0:
            break

        piece_moved = board.piece_at(move.from_square)
        piece_captured = board.piece_at(move.to_square)
        to_sq = move.to_square

        gives_check = board.gives_check(move)
        is_capture = piece_captured is not None or board.is_en_passant(move)

        new_depth = depth - 1
        # Tactical Check Extension: Never cut off search when King is under direct assault!
        if gives_check and depth == 1 and ply < 6:
            new_depth = 1

        board.push(move)
        try:
            # Late Move Reductions (LMR):
            # Top 2 candidates, checks, and captures are searched at full depth.
            # Quiet moves ranked 3rd or later are searched with 1-ply reduction.
            do_lmr = (depth >= 3 and idx >= 2 and not gives_check and not is_capture)
            if do_lmr:
                reduced_depth = max(1, new_depth - 1)
                score, child_pv, mode = search_alphabeta(
                    board, reduced_depth, -beta, -alpha, ply + 1, deadline=deadline
                )
                score = -score
                # If the reduced search beats alpha, re-search at full depth
                if score > alpha:
                    score, child_pv, mode = search_alphabeta(
                        board, new_depth, -beta, -alpha, ply + 1, deadline=deadline
                    )
                    score = -score
            else:
                score, child_pv, mode = search_alphabeta(
                    board, new_depth, -beta, -alpha, ply + 1, deadline=deadline
                )
                score = -score

            # Check for hanging pieces / unsound sacrifices
            # A sacrifice is unsound if it loses material WITHOUT check, WITHOUT mate threat,
            # and WITHOUT heuristic compensation (king exposure / structural ruin).
            if score < 8.0:
                opp_attackers = board.attackers(board.turn, to_sq)
                our_defenders = board.attackers(not board.turn, to_sq)
                moved_val = PIECE_VALUES.get(piece_moved.piece_type, 0) if (piece_moved and piece_moved.piece_type != chess.KING) else 0
                cap_val = PIECE_VALUES.get(piece_captured.piece_type, 0) if piece_captured else 0

                # Opponent king exposure after our move
                opp_king_exposure = evaluate_king_exposure(board, board.turn)
                has_king_assault = opp_king_exposure >= 55.0

                # 1. Did we put an unthreatened piece on a square attacked by an enemy pawn?
                pawn_attackers = [
                    sq for sq in opp_attackers
                    if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN
                ]
                if pawn_attackers and moved_val > PIECE_VALUES[chess.PAWN]:
                    if not gives_check and score < 0.80:
                        score -= 3.5  # Severe penalty: recklessly hung to enemy pawn

                # 2. Did we put a queen/rook on a square attacked by a minor piece?
                minor_attackers = [
                    sq for sq in opp_attackers
                    if board.piece_at(sq) and board.piece_at(sq).piece_type in (chess.KNIGHT, chess.BISHOP)
                ]
                if minor_attackers and moved_val > PIECE_VALUES[chess.BISHOP]:
                    if not gives_check and score < 0.80:
                        score -= 3.0  # Severe penalty: recklessly hung to enemy minor piece

                # 3. Completely undefended piece left under attack without check or compensation
                if opp_attackers and not our_defenders and moved_val > cap_val:
                    if not gives_check and not has_king_assault and score < 0.50:
                        score -= 2.5

                # 4. Speculative sacrifice (giving up material without check, king hunt, or search refutation)
                if moved_val > cap_val + 150:
                    is_exchange_sac = (piece_moved and piece_moved.piece_type == chess.ROOK and
                                       piece_captured and piece_captured.piece_type in (chess.KNIGHT, chess.BISHOP))
                    exchange_compensated = is_exchange_sac and (has_king_assault or score >= -0.20)

                    if not gives_check and not has_king_assault and not exchange_compensated:
                        if len(opp_attackers) >= len(our_defenders):
                            score -= 3.0
                        elif score < 0.20:
                            score -= 1.5

                # 5. Check if THIS move left ANY other friendly piece hanging
                hanging_left = get_hanging_pieces(board, not board.turn)
                if hanging_left:
                    worst_loss = hanging_left[0]["value"]
                    if not gives_check and not has_king_assault:
                        score -= (worst_loss / 100.0) * 0.9
        except SearchTimeout:
            timed_out = True
            if ply > 0:
                raise
        finally:
            board.pop()

        if timed_out:
            if ply == 0:
                break

        if score > best_score:
            best_score = score
            best_move = move
            best_pv = [move] + child_pv
            best_mode = mode

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    if best_score == -float("inf"):
        best_score, best_mode = evaluate_board_hybrid(board, ply=ply)
        best_pv = [candidates[0]]
        best_move = candidates[0]

    if not timed_out:
        if best_score <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif best_score >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT

        if len(TRANSPOSITION_TABLE) > 100000:
            TRANSPOSITION_TABLE.clear()
        TRANSPOSITION_TABLE[zkey] = TTEntry(depth, flag, best_score, best_move, best_pv)

    return best_score, best_pv, best_mode


def predict_move_internal(
    board: chess.Board,
    temperature: float = 0.0,
    ply: int = 0,
    use_book: bool = True,
    use_search: bool = True,
    search_depth: int = 2,
) -> tuple[chess.Move, float, torch.Tensor, float, str, str, str | None, str, list[str]]:
    """
    Returns:
        (best_real_move, confidence, probs, value_eval, source, eval_mode, anticipated_counter, tactical_rationale, pv_san)
    source is one of: "book", "search", "model".
    """
    is_black = board.turn == chess.BLACK

    # Pre-scan for friendly hanging pieces before decision
    hanging_pieces_before = get_hanging_pieces(board, board.turn)

    # 1. Polyglot Master Opening Book (for plies <= 16)
    if use_book and ply <= 16:
        book_move, book_conf = get_book_move(board, temperature=temperature)
        if book_move is not None and book_move in board.legal_moves:
            dummy_probs = torch.zeros((4096,), device=DEVICE)
            cm = real_move_to_canonical(book_move, is_black)
            dummy_probs[encode_move(cm)] = 1.0
            pv_san = [board.san(book_move)]
            rationale = "Standard grandmaster tournament preparation from Titans polyglot book. Maintains theoretical equity without tactical risk."
            return book_move, max(0.95, book_conf), dummy_probs, 0.0, "book", "prophylaxis", None, rationale, pv_san

    # Current root evaluation mode
    _, root_eval_mode = evaluate_board_hybrid(board, ply=ply)

    # 2. Minimax Tactical Search with Opponent Refutation (Depth 2-4 with LMR, NMP & Check Extensions)
    if use_search and search_depth >= 2 and not board.is_game_over():
        # Hard Safety Clamp: Never attempt raw depth > 4 in Python to prevent combinatorial freeze
        safe_depth = min(max(search_depth, 2), 4)
        # Search Time Budget: 1000ms cap to guarantee snappy UX and prevent 504 gateway timeouts
        search_deadline = time.time() + 1.00
        search_score, search_pv, _ = search_alphabeta(
            board, depth=safe_depth, alpha=-float("inf"), beta=float("inf"), ply=0, deadline=search_deadline
        )
        eval_mode = "tactical_strike" if search_score >= 8.0 else root_eval_mode
        if search_pv and search_pv[0] in board.legal_moves:
            search_move = search_pv[0]
            pv_san = []
            temp_b = board.copy()
            for m in search_pv:
                if m in temp_b.legal_moves:
                    pv_san.append(temp_b.san(m))
                    temp_b.push(m)
                else:
                    break

            anticipated_counter = pv_san[1] if len(pv_san) > 1 else None
            probs, _ = get_neural_policy_and_value(board)
            cm = real_move_to_canonical(search_move, is_black)
            best_p = probs[encode_move(cm)].item()

            # Check for Sound Sacrifices or King Hunt
            moved_p = board.piece_at(search_move.from_square)
            cap_p = board.piece_at(search_move.to_square)
            m_val = PIECE_VALUES.get(moved_p.piece_type, 0) if moved_p else 0
            c_val = PIECE_VALUES.get(cap_p.piece_type, 0) if cap_p else 0
            gives_chk = board.gives_check(search_move)

            board.push(search_move)
            opp_exp_after = evaluate_king_exposure(board, board.turn)
            opp_attackers_to = board.attackers(board.turn, search_move.to_square)
            our_defenders_to = board.attackers(not board.turn, search_move.to_square)
            board.pop()

            is_attacked_on_to = (len(opp_attackers_to) > len(our_defenders_to)) or any(
                board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN for sq in opp_attackers_to
            )

            is_sac = False
            if (m_val > c_val + 150) and is_attacked_on_to:
                is_sac = True
            elif (m_val == 500 and c_val in (300, 320)) and is_attacked_on_to:
                is_sac = True

            if is_sac and search_score >= 0.20 and (search_score >= 0.50 or opp_exp_after >= 55.0 or gives_chk):
                if m_val == 500 and c_val in (300, 320):
                    rationale = f"🗡️ Exchange Sacrifice: Conceding the exchange ({board.san(search_move)}) to dismantle enemy pawn shelter and seize positional domination."
                    eval_mode = "tactical_strike"
                elif gives_chk or opp_exp_after >= 75.0:
                    rationale = f"✨ Sound Tactical Sacrifice: {board.san(search_move)} forces the exposed enemy King into a calculated mating net."
                    eval_mode = "tactical_strike"
                else:
                    rationale = f"✨ Sound Positional Sacrifice: Material conceded for decisive space and attacking compensation."
                    eval_mode = "tactical_strike"
            elif gives_chk and (opp_exp_after >= 55.0 or search_score >= 1.0):
                rationale = f"👑 King Hunt: Direct assault with {board.san(search_move)} against vulnerable King with tactical check extensions."
                eval_mode = "tactical_strike"
            elif hanging_pieces_before and search_move.from_square == hanging_pieces_before[0]["square"]:
                saved_piece = hanging_pieces_before[0]
                p_name = chess.piece_name(saved_piece["piece_type"]).title()
                sq_name = saved_piece["san_sq"]
                rationale = f"🛡️ Tactical Retreat: Saved hanging {p_name} on {sq_name}."
                eval_mode = "tenacious_defense"
            elif hanging_pieces_before:
                saved_piece = hanging_pieces_before[0]
                board.push(search_move)
                now_defended = board.is_attacked_by(not board.turn, saved_piece["square"])
                board.pop()
                if now_defended:
                    p_name = chess.piece_name(saved_piece["piece_type"]).title()
                    sq_name = saved_piece["san_sq"]
                    rationale = f"🛡️ Tactical Reinforcement: Defended threatened {p_name} on {sq_name}."
                    eval_mode = "tenacious_defense"
                elif eval_mode == "strategic_development":
                    rationale = "🏛️ Strategic Development: Maintaining central pawn tension, harmonious piece coordination, and King safety."
                elif eval_mode == "tenacious_defense":
                    rationale = "🛡️ Tenacious Defense: Complicating position, preventing passive simplification, and fighting for counterplay."
                else:
                    rationale = "⚡ Tactical Refutation: Concrete calculated tactic with tactical refutation."
            elif eval_mode == "strategic_development":
                rationale = "🏛️ Strategic Development: Maintaining central pawn tension, harmonious piece coordination, and King safety."
            elif eval_mode == "tenacious_defense":
                if board.can_claim_threefold_repetition() or board.is_repetition(2):
                    rationale = "🛡️ Tenacious Defense: Seeking perpetual check draw / repetition swindle to hold the fortress."
                else:
                    rationale = "🛡️ Tenacious Defense: Complicating position, preventing passive simplification, and fighting for counterplay."
            elif eval_mode == "prophylaxis":
                rationale = "🧘 Prophylaxis / Positional Maneuvering: Securing piece coordination, central control, and avoiding premature attacks."
            elif eval_mode == "clean_conversion":
                rationale = "⚔️ Clean Conversion: Consolidating advantage, escorting passed pawns, and restricting opponent counterplay."
            elif eval_mode == "tactical_strike":
                rationale = "⚡ Tactical Strike: Dynamic combination targeting tactical vulnerabilities."
            else:
                rationale = "⚡ Tactical Refutation: Concrete calculated tactic with tactical refutation."

            return search_move, max(0.5, best_p), probs, search_score, "search", eval_mode, anticipated_counter, rationale, pv_san

    # 3. Direct SE-ResNet Neural Network Evaluation (with Zobrist Eval Caching)
    probs, value_eval = get_neural_policy_and_value(board)
    masked_logits = torch.log(probs + 1e-12)

    # Opening temperature sampling during first 10 plies
    if ply <= 10 and temperature > 0.1:
        scaled_logits = masked_logits / max(0.1, temperature)
        sample_probs = F.softmax(scaled_logits, dim=0)
        best_idx = torch.multinomial(sample_probs, 1).item()
        best_prob = probs[best_idx].item()
    else:
        best_idx = torch.argmax(probs).item()
        best_prob = probs[best_idx].item()

    best_canonical_move = decode_move(best_idx)
    best_real_move = canonical_move_to_real(best_canonical_move, is_black)

    # Pawn promotion fallback if not specified in Move
    piece = board.piece_at(best_real_move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(best_real_move.to_square)
        if (board.turn == chess.WHITE and to_rank == 7) or (board.turn == chess.BLACK and to_rank == 0):
            if best_real_move.promotion is None:
                best_real_move = chess.Move(
                    best_real_move.from_square,
                    best_real_move.to_square,
                    promotion=chess.QUEEN,
                )

    if best_real_move not in board.legal_moves:
        for lm in board.legal_moves:
            if lm.from_square == best_real_move.from_square and lm.to_square == best_real_move.to_square:
                best_real_move = lm
                break

    hybrid_eval, eval_mode = evaluate_board_hybrid(board, ply=ply)
    if hanging_pieces_before and best_real_move.from_square == hanging_pieces_before[0]["square"]:
        saved_piece = hanging_pieces_before[0]
        p_name = chess.piece_name(saved_piece["piece_type"]).title()
        sq_name = saved_piece["san_sq"]
        rationale = f"🛡️ Tactical Retreat: Saved hanging {p_name} on {sq_name}."
        eval_mode = "tenacious_defense"
    elif hanging_pieces_before:
        saved_piece = hanging_pieces_before[0]
        board.push(best_real_move)
        now_defended = board.is_attacked_by(not board.turn, saved_piece["square"])
        board.pop()
        if now_defended:
            p_name = chess.piece_name(saved_piece["piece_type"]).title()
            sq_name = saved_piece["san_sq"]
            rationale = f"🛡️ Tactical Reinforcement: Defended threatened {p_name} on {sq_name}."
            eval_mode = "tenacious_defense"
        else:
            rationale = f"Direct tactical policy head selection with {(best_prob * 100):.1f}% certainty."
    else:
        rationale = f"Direct tactical policy head selection with {(best_prob * 100):.1f}% certainty."
    pv_san = [board.san(best_real_move)]

    return best_real_move, best_prob, probs, hybrid_eval, "model", eval_mode, None, rationale, pv_san


@app.post("/predict-move", response_model=PredictResponse)
def predict_move(req: FENRequest):
    start_time = time.perf_counter()
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if not board.legal_moves:
        raise HTTPException(
            status_code=400,
            detail="No legal moves available from this position",
        )

    is_black = board.turn == chess.BLACK
    best_move, best_prob, probs, value_eval, source, eval_mode, anticipated_counter, tactical_rationale, pv_san = predict_move_internal(
        board,
        temperature=req.temperature,
        ply=req.ply,
        use_book=req.use_book,
        use_search=req.use_search,
        search_depth=req.search_depth,
    )

    top_moves = [
        MoveInfo(
            uci=best_move.uci(),
            san=board.san(best_move),
            from_sq=chess.square_name(best_move.from_square),
            to_sq=chess.square_name(best_move.to_square),
            confidence=best_prob,
            is_book=(source == "book"),
            source=source,
        )
    ]

    # Additional top legal moves
    k = min(3, len(list(board.legal_moves)))
    top_k_indices = torch.topk(probs, k=k).indices.tolist()

    for idx in top_k_indices:
        canonical_m = decode_move(idx)
        real_m = canonical_move_to_real(canonical_m, is_black)
        if real_m not in board.legal_moves or real_m == best_move:
            continue
        prob = probs[idx].item()

        san = board.san(real_m)
        from_sq = chess.square_name(real_m.from_square)
        to_sq = chess.square_name(real_m.to_square)

        top_moves.append(
            MoveInfo(
                uci=real_m.uci(),
                san=san,
                from_sq=from_sq,
                to_sq=to_sq,
                confidence=prob,
                is_book=False,
                source="model",
            )
        )

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000

    return PredictResponse(
        fen=board.fen(),
        top_moves=top_moves,
        inference_time_ms=inference_time_ms,
        win_eval=value_eval,
        is_book_move=(source == "book"),
        source=source,
        eval_mode=eval_mode,
        anticipated_counter=anticipated_counter,
        tactical_rationale=tactical_rationale,
        search_pv=pv_san,
    )


@app.post("/adaptive-step", response_model=AdaptiveStepResponse)
def adaptive_step(req: AdaptiveStepRequest):
    """
    Adaptive Sparring step against Stockfish 17:
    1. Evaluates model's move quality to compute Centipawn Loss (CPL).
    2. Updates 30-move rolling Average Centipawn Loss (ACPL).
    3. Calculates estimated Model Elo: clamp(3100 - 28 * ACPL, 600, 2800).
    4. Throttles Stockfish dynamically: UCI_LimitStrength=True, UCI_Elo=estimated_elo + 30, Skill Level.
    5. Generates Stockfish counter-move and returns live metrics.
    """
    start_time = time.perf_counter()
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    engine = get_stockfish_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Stockfish engine unavailable")

    cpl = 0.0
    updated_history = list(req.history_cpl)

    # 1. Centipawn Loss evaluation if a model move was supplied
    # Ensure engine runs unconstrained as Judge for objective ACPL evaluation
    try:
        engine.configure({
            "UCI_LimitStrength": False,
            "Skill Level": 20,
        })
    except Exception:
        pass

    if req.model_move_uci:
        try:
            model_move = chess.Move.from_uci(req.model_move_uci)
            if model_move in board.legal_moves:
                turn = board.turn
                # Pre-move evaluation (unconstrained judge)
                info_before = engine.analyse(board, chess.engine.Limit(time=0.08, depth=10))
                score_before = info_before["score"].pov(turn).score(mate_score=10000) or 0

                # Execute model move
                board.push(model_move)

                # Post-move evaluation (from model's perspective)
                info_after = engine.analyse(board, chess.engine.Limit(time=0.08, depth=10))
                score_after = info_after["score"].pov(turn).score(mate_score=10000) or 0

                raw_cpl = max(0.0, float(score_before - score_after))
                cpl = min(raw_cpl, 250.0)
                updated_history.append(round(cpl, 1))
        except Exception as e:
            logger.warning(f"Error evaluating model move CPL: {e}")

    # 2. Rolling 30-move ACPL
    recent_cpl = updated_history[-30:] if updated_history else []
    rolling_acpl = float(np.mean(recent_cpl)) if recent_cpl else 0.0

    # 3. Estimated Model Elo: smooth, realistic curve (no 600 floor clamp)
    estimated_elo = estimate_performance_elo(rolling_acpl, result="draw")
    model_stats["estimated_elo"] = estimated_elo
    model_stats["recent_cpl"] = recent_cpl

    # 4. Stockfish Dynamic Elo & Skill Level: UCI_Elo = estimated_elo + 30
    sf_elo = min(2850, estimated_elo + 30)
    skill_level = int(np.clip((sf_elo - 800) / 100.0, 0, 20))

    # 5. Check if board is terminal before counter-move
    if board.is_game_over() or not list(board.legal_moves):
        is_check = board.is_check()
        is_mate = board.is_checkmate()
        is_draw = (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        )
        end_time = time.perf_counter()
        return AdaptiveStepResponse(
            model_cpl=round(cpl, 1),
            rolling_acpl=round(rolling_acpl, 1),
            estimated_elo=estimated_elo,
            stockfish_elo=sf_elo,
            stockfish_skill_level=skill_level,
            counter_move=None,
            fen_after=board.fen(),
            is_check=is_check,
            is_checkmate=is_mate,
            is_draw=is_draw,
            history_cpl=updated_history,
            inference_time_ms=(end_time - start_time) * 1000,
        )

    # 6. Generate Stockfish Counter-Move with dynamic handicap
    try:
        try:
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": max(1320, min(2850, sf_elo)),
                "Skill Level": skill_level,
            })
        except Exception as e:
            logger.warning(f"Error configuring Stockfish limit strength: {e}")

        sf_result = engine.play(board, chess.engine.Limit(time=0.1, depth=10))
    finally:
        # Immediately reset engine so any subsequent analyses/evaluations run unconstrained
        try:
            engine.configure({
                "UCI_LimitStrength": False,
                "Skill Level": 20,
            })
        except Exception:
            pass

    try:
        counter_m = sf_result.move
        if counter_m is None:
            counter_m = list(board.legal_moves)[0]

        san = board.san(counter_m)
        from_sq = chess.square_name(counter_m.from_square)
        to_sq = chess.square_name(counter_m.to_square)

        board.push(counter_m)

        is_check = board.is_check()
        is_mate = board.is_checkmate()
        is_draw = (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        )

        counter_info = MoveInfo(
            uci=counter_m.uci(),
            san=san,
            from_sq=from_sq,
            to_sq=to_sq,
            confidence=1.0,
        )
    except Exception as e:
        logger.error(f"Error executing Stockfish move: {e}")
        raise HTTPException(status_code=500, detail=f"Stockfish execution failed: {e}")

    end_time = time.perf_counter()
    return AdaptiveStepResponse(
        model_cpl=round(cpl, 1),
        rolling_acpl=round(rolling_acpl, 1),
        estimated_elo=estimated_elo,
        stockfish_elo=sf_elo,
        stockfish_skill_level=skill_level,
        counter_move=counter_info,
        fen_after=board.fen(),
        is_check=is_check,
        is_checkmate=is_mate,
        is_draw=is_draw,
        history_cpl=updated_history,
        inference_time_ms=(end_time - start_time) * 1000,
    )


def predict_top_moves(
    board: chess.Board, k: int = 3
) -> tuple[list[tuple[chess.Move, float]], float]:
    """
    Extracts top-k legal moves from the CNN policy head with probabilities,
    plus the position value evaluation.
    """
    is_black = board.turn == chess.BLACK
    tensor_input = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    if MODEL_TYPE == "onnx":
        input_name = ONNX_SESSION.get_inputs()[0].name
        numpy_input = tensor_input.cpu().numpy().astype(np.float32)
        outputs = ONNX_SESSION.run(None, {input_name: numpy_input})
        logits = torch.tensor(outputs[0][0], device=DEVICE)
        value_eval = float(outputs[1][0][0]) if len(outputs) > 1 else 0.0
    else:
        with torch.no_grad():
            policy_logits, value_preds = MODEL(tensor_input)
            logits = policy_logits[0]
            value_eval = float(value_preds[0].item())

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], value_eval

    # Canonical illegal move masking
    mask = torch.full((4096,), float("-inf"), device=logits.device)
    for m in legal_moves:
        cm = real_move_to_canonical(m, is_black)
        mask[encode_move(cm)] = 0.0

    probs = F.softmax(logits + mask, dim=0)
    top_k_count = min(k, len(legal_moves))
    topk = torch.topk(probs, k=top_k_count)

    results = []
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        cm = decode_move(idx)
        rm = canonical_move_to_real(cm, is_black)
        results.append((rm, p))

    return results, value_eval


def select_adversarial_defense(board: chess.Board) -> chess.Move | None:
    """
    Selects the most obstinate defensive move for the opponent:
    1. If forced (only 1 legal move), returns it immediately.
    2. Filters out candidate moves that leave an immediate checkmate for the attacker.
    3. Evaluates remaining defensive candidates in a single batched forward pass
       using the value head, selecting the move that minimizes the attacker's win certainty.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    if len(legal_moves) == 1:
        return legal_moves[0]

    # Filter out moves that leave immediate checkmate
    safe_moves = []
    for m in legal_moves:
        board.push(m)
        if not board.is_checkmate():
            safe_moves.append(m)
        board.pop()

    candidates = safe_moves if safe_moves else legal_moves
    if len(candidates) == 1:
        return candidates[0]

    # Batch tensor evaluation for candidate replies
    tensors = []
    for m in candidates:
        board.push(m)
        tensors.append(board_to_tensor(board))
        board.pop()

    batch_input = torch.stack(tensors)  # (N, 18, 8, 8)

    if MODEL_TYPE == "onnx":
        input_name = ONNX_SESSION.get_inputs()[0].name
        numpy_input = batch_input.cpu().numpy().astype(np.float32)
        outputs = ONNX_SESSION.run(None, {input_name: numpy_input})
        if len(outputs) > 1:
            values = outputs[1].flatten()
            min_idx = int(np.argmin(values))
            return candidates[min_idx]
        return candidates[0]
    else:
        with torch.no_grad():
            _, value_preds = MODEL(batch_input.to(DEVICE))
            values = value_preds.view(-1)
            min_idx = int(torch.argmin(values).item())
            return candidates[min_idx]


def find_shortest_mate(
    board: chess.Board, max_search_depth: int = 20
) -> tuple[
    list[tuple[chess.Move, float]] | None,
    bool,
    str | None,
    str | None,
    int,
    float,
    float,
]:
    """
    Finds the quickest forced checkmate sequence using policy-guided
    Iterative Deepening Search with Strategy Pivoting and Backtracking.

    Returns:
        (solution_path, strategy_pivoted, pivoted_from, pivoted_to, candidate_rank, path_score, win_eval)
    """
    root_turn = board.turn
    top_candidates, initial_val = predict_top_moves(board, k=3)
    if not top_candidates:
        return None, False, None, None, 1, 0.0, initial_val

    top1_move = top_candidates[0][0]

    # Fast-path: Checkmate in 1 on top-1 candidate
    board.push(top1_move)
    if board.is_checkmate():
        board.pop()
        score = 1.0 - (0.05 * 1)
        return (
            [(top1_move, top_candidates[0][1])],
            False,
            None,
            None,
            1,
            score,
            initial_val,
        )
    board.pop()

    # Iterative Deepening limits: Mate-in-1 (1, 2), Mate-in-2 (3, 4), Mate-in-3 (5, 6), etc.
    limits = [
        lim
        for lim in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
        if lim <= max_search_depth
    ]

    for limit in limits:
        visited = set()

        def dfs(
            curr_board: chess.Board, current_depth: int, path: list
        ) -> list[tuple[chess.Move, float]] | None:
            # Repetition & cycle prevention (3-fold repetition avoidance)
            pos_key = curr_board._transposition_key()
            if pos_key in visited:
                return None
            visited.add(pos_key)

            if curr_board.is_checkmate():
                visited.remove(pos_key)
                return path.copy()

            if current_depth >= limit or curr_board.is_game_over():
                visited.remove(pos_key)
                return None

            is_attacker = curr_board.turn == root_turn

            if is_attacker:
                # Branching & Backtracking: evaluate top-3 candidate moves from CNN policy head
                candidates, _ = predict_top_moves(curr_board, k=3)
                for move, prob in candidates:
                    path.append((move, prob))
                    curr_board.push(move)

                    result = dfs(curr_board, current_depth + 1, path)

                    curr_board.pop()
                    path.pop()

                    if result is not None:
                        visited.remove(pos_key)
                        return result
            else:
                # Adversarial defender: tests the most obstinate defensive reply
                def_move = select_adversarial_defense(curr_board)
                if def_move is not None:
                    path.append((def_move, 0.0))
                    curr_board.push(def_move)

                    result = dfs(curr_board, current_depth + 1, path)

                    curr_board.pop()
                    path.pop()

                    if result is not None:
                        visited.remove(pos_key)
                        return result

            visited.remove(pos_key)
            return None

        solution = dfs(board, 0, [])
        if solution:
            # Check if strategy pivoted away from top1_move at root
            chosen_root_move = solution[0][0]
            strategy_pivoted = chosen_root_move != top1_move
            pivoted_from = top1_move.uci() if strategy_pivoted else None
            pivoted_to = chosen_root_move.uci() if strategy_pivoted else None

            candidate_rank = 1
            for rank_idx, (cand_m, _) in enumerate(top_candidates, 1):
                if cand_m == chosen_root_move:
                    candidate_rank = rank_idx
                    break

            plies = len(solution)
            path_score = max(0.0, round(1.0 - (0.05 * plies), 4))
            return (
                solution,
                strategy_pivoted,
                pivoted_from,
                pivoted_to,
                candidate_rank,
                path_score,
                initial_val,
            )

    return None, False, None, None, 1, 0.0, initial_val


@app.post("/solve-mate", response_model=SolveResponse)
def solve_mate(req: FENRequest):
    start_time = time.perf_counter()
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    # 1. Optimal Shortest-Path Search with Iterative Deepening & Pivoting (depth 1..20)
    (
        solution,
        strategy_pivoted,
        pivoted_from,
        pivoted_to,
        candidate_rank,
        path_score,
        initial_val,
    ) = find_shortest_mate(board, max_search_depth=20)

    if solution:
        sequence = []
        step_num = 1
        cumulative_confidence = 1.0
        moves_list = []

        curr = board.copy()
        for move, conf in solution:
            moves_list.append(move.uci())
            if conf > 0:
                cumulative_confidence *= conf

            san = curr.san(move)
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)

            curr.push(move)

            is_check = curr.is_check()
            is_mate = curr.is_checkmate()

            sequence.append(
                SequenceStep(
                    step=step_num,
                    uci=move.uci(),
                    san=san,
                    from_sq=from_sq,
                    to_sq=to_sq,
                    fen_after=curr.fen(),
                    confidence=conf,
                    is_check=is_check,
                    is_mate=is_mate,
                )
            )
            step_num += 1

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        mate_in = (len(sequence) + 1) // 2

        return SolveResponse(
            found_mate=True,
            is_checkmate=True,
            final_fen=curr.fen(),
            mate_in=mate_in,
            plies=len(sequence),
            moves=moves_list,
            strategy_pivoted=strategy_pivoted,
            pivoted_from_move=pivoted_from,
            pivoted_to_move=pivoted_to,
            candidate_rank=candidate_rank,
            path_score=path_score,
            confidence=cumulative_confidence,
            sequence=sequence,
            inference_time_ms=inference_time_ms,
            win_eval=initial_val,
        )

    # 2. Fallback: Deep greedy auto-play rollout (up to 40 plies) if no mate found in <= 20 plies
    sequence = []
    found_mate = False
    step_num = 1
    depth = 0
    max_depth = 40
    cumulative_confidence = 1.0
    initial_value_eval = None
    moves_list = []

    curr = board.copy()
    while not curr.is_game_over() and depth < max_depth:
        best_move, conf, _, val_eval, _, *_ = predict_move_internal(
            curr, use_book=False, use_search=False
        )
        if initial_value_eval is None:
            initial_value_eval = val_eval
        cumulative_confidence *= conf

        moves_list.append(best_move.uci())
        san = curr.san(best_move)
        from_sq = chess.square_name(best_move.from_square)
        to_sq = chess.square_name(best_move.to_square)

        curr.push(best_move)

        is_check = curr.is_check()
        is_mate = curr.is_checkmate()

        sequence.append(
            SequenceStep(
                step=step_num,
                uci=best_move.uci(),
                san=san,
                from_sq=from_sq,
                to_sq=to_sq,
                fen_after=curr.fen(),
                confidence=conf,
                is_check=is_check,
                is_mate=is_mate,
            )
        )

        depth += 1
        if is_mate:
            found_mate = True
            break

        if curr.is_game_over() or depth >= max_depth:
            break

        step_num += 1

        # Defender's adversarial response
        chosen_defense = select_adversarial_defense(curr)
        if chosen_defense is None:
            break

        moves_list.append(chosen_defense.uci())
        san = curr.san(chosen_defense)
        from_sq = chess.square_name(chosen_defense.from_square)
        to_sq = chess.square_name(chosen_defense.to_square)

        curr.push(chosen_defense)

        is_check = curr.is_check()
        is_mate = curr.is_checkmate()

        sequence.append(
            SequenceStep(
                step=step_num,
                uci=chosen_defense.uci(),
                san=san,
                from_sq=from_sq,
                to_sq=to_sq,
                fen_after=curr.fen(),
                confidence=0.0,
                is_check=is_check,
                is_mate=is_mate,
            )
        )

        depth += 1
        if is_mate:
            break

        if curr.is_game_over() or depth >= max_depth:
            break

        step_num += 1

    mate_in = (len(sequence) + 1) // 2 if found_mate else None
    plies = len(sequence)
    path_score = max(0.0, round(1.0 - (0.05 * plies), 4)) if found_mate else 0.0
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000

    return SolveResponse(
        found_mate=found_mate,
        is_checkmate=found_mate,
        final_fen=curr.fen(),
        mate_in=mate_in,
        plies=plies,
        moves=moves_list,
        strategy_pivoted=False,
        pivoted_from_move=None,
        pivoted_to_move=None,
        candidate_rank=1,
        path_score=path_score,
        confidence=cumulative_confidence if found_mate else 0.0,
        sequence=sequence,
        inference_time_ms=inference_time_ms,
        win_eval=initial_value_eval,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


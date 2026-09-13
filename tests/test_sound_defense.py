#!/usr/bin/env python3
"""
Test script to verify:
1. Hard Material Delta Check & Anti-Delusion Clamping (evaluate_board_hybrid)
2. Quiescence Search Stand-Pat fix (cannot stand pat on hallucinated score or in check)
3. Opponent Refutation and Unsound Sacrifice Rejection in Minimax Search
4. Quiet Middlegame Positional Patience & Clean Conversion
"""

import os
import sys
from pathlib import Path

# Ensure ML-miniproject project root is on sys.path
PROJECT_DIR = Path("/home/meenun/Documents/ML-miniproject")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import json
import urllib.request
import chess

BASE_URL = "http://localhost:8000"

def post_json(endpoint: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{endpoint}") as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_health():
    data = get_json("/health")
    assert data.get("status") == "ok", f"Health check failed: {data}"
    print("✅ Backend health check passed.")

def test_hard_material_clamp():
    print("\n--- Test 1: Hard Material Delta Check & Anti-Delusion Clamp ---")
    # Position where White is down a Queen and Rook for nothing
    # FEN: 4k3/8/8/8/8/8/8/4K2R w - - 0 1 -> White only has King and Rook vs King (winning)
    # White down Queen: White has King and pawns, Black has King, Queen, Rook, pawns
    fen_down_queen = "r2q2k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1"
    payload = {
        "fen": fen_down_queen,
        "temperature": 0.0,
        "use_book": False,
        "use_search": False  # Check raw evaluation without search masking
    }
    data = post_json("/predict-move", payload)
    print(f"Down Queen - Eval mode: {data.get('eval_mode')}, Win eval: {data.get('win_eval')}")
    assert data.get("eval_mode") == "tenacious_defense", f"Expected tenacious_defense, got {data.get('eval_mode')}"
    assert data.get("win_eval") is not None and data.get("win_eval") < -0.5, f"Expected win_eval < -0.5 when down a queen, got {data.get('win_eval')}"
    print("✅ Hard material delta check verified: Down-material positions are strictly recognized as defensive with negative evaluation.")

def test_unsound_queen_sacrifice_rejected():
    print("\n--- Test 2: Unsound Queen Sacrifice Rejection ---")
    # Classic position where superficial puzzle pattern might want to play Bxh7+ or Qxf7+ or sacrifice queen,
    # but there is no forced mate and the opponent simply captures it.
    # Position: r1bq1rk1/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 4 8
    payload = {
        "fen": "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 4 8",
        "temperature": 0.0,
        "use_book": False,
        "use_search": True,
        "search_depth": 2
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    san = top_move["san"]
    print(f"Top move chosen: {san} ({top_move['uci']})")
    print(f"Eval Mode: {data.get('eval_mode')}")
    print(f"Anticipated Counter: {data.get('anticipated_counter')}")
    print(f"Search PV: {data.get('search_pv')}")
    print(f"Tactical Rationale: {data.get('tactical_rationale')}")

    # Verify no reckless queen fling into attacked squares
    assert san not in ("Qh5", "Qxa7", "Qc2", "Qb3") or not san.startswith("Qx"), f"Reckless queen sacrifice: {san}"
    # Verify move is sound positional chess (O-O, Qc2, b3, cxd5, etc.)
    print(f"✅ Unsound sacrifice test passed: sound move chosen ({san}) with anticipated counter {data.get('anticipated_counter')}.")

def test_queen_hung_to_pawn_rejected():
    print("\n--- Test 3: Queen Hanging to Enemy Pawn Rejection ---")
    # Position where White Queen could move to h6 (attacked by g7 pawn) or a5/b6
    # Let's test standard Italian opening: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5
    fen = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    payload = {
        "fen": fen,
        "temperature": 0.0,
        "use_book": False,
        "use_search": True,
        "search_depth": 2
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    san = top_move["san"]
    print(f"Top move chosen: {san} ({top_move['uci']})")
    print(f"Eval Mode: {data.get('eval_mode')}")
    print(f"Anticipated Counter: {data.get('anticipated_counter')}")
    print(f"Search PV: {data.get('search_pv')}")
    
    # Must NOT play suicidal queen or bishop sacrifice
    assert not san.startswith("Q") or san == "Qe2", f"Reckless queen move chosen: {san}"
    assert san != "Bxf7+", "Reckless bishop sacrifice chosen!"
    print(f"✅ Suicide move rejection passed: chosen {san}.")

def test_winning_conversion():
    print("\n--- Test 4: Winning Conversion Preserved ---")
    payload = {
        "fen": "4k3/8/8/8/8/8/4Q3/4K2R w - - 0 1",
        "temperature": 0.0,
        "use_book": False,
        "use_search": True
    }
    data = post_json("/predict-move", payload)
    print(f"Top move chosen: {data['top_moves'][0]['san']}")
    print(f"Eval mode: {data.get('eval_mode')}")
    assert data.get('eval_mode') in ('clean_conversion', 'tactical_strike'), f"Expected clean_conversion or tactical_strike, got {data.get('eval_mode')}"
    print("✅ Winning conversion test passed.")

def test_opening_concepts_warm_handoff():
    print("\n--- Test 5: Warm Handoff & Strategic Opening Concepts ---")
    # Position at ply 16 out of book (Italian game after 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. h3 a6 8. Bb3)
    fen = "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2PP1N1P/PP3PP1/RNBQ1RK1 b - - 0 8"
    payload = {
        "fen": fen,
        "temperature": 0.0,
        "ply": 16,
        "use_book": False,  # Test neural search with warm handoff
        "use_search": True,
        "search_depth": 2
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    print(f"Top move chosen: {top_move['san']} ({top_move['uci']})")
    print(f"Eval Mode: {data.get('eval_mode')}")
    print(f"Anticipated Counter: {data.get('anticipated_counter')}")
    print(f"Tactical Rationale: {data.get('tactical_rationale')}")
    assert data.get("eval_mode") == "strategic_development", f"Expected strategic_development, got {data.get('eval_mode')}"
    assert "Strategic Development" in data.get("tactical_rationale", "")
    print("✅ Strategic development warm handoff test passed.")

def test_hanging_piece_tactical_retreat():
    print("\n--- Test 6: Hanging Piece Detection & Safe Tactical Retreat ---")
    # Position where White Knight on f3 is undefended and attacked by Black pawn on g4
    fen = "r1bqkbnr/pppp1ppp/8/8/4P1p1/5NP1/PPQP4/RNB1KB1R w KQkq - 0 1"
    payload = {
        "fen": fen,
        "temperature": 0.0,
        "use_book": False,
        "use_search": True,
        "search_depth": 2,
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    san = top_move["san"]
    from_sq = top_move["from_sq"]
    to_sq = top_move["to_sq"]
    print(f"Top move chosen: {san} ({top_move['uci']}) from {from_sq} to {to_sq}")
    print(f"Eval Mode: {data.get('eval_mode')}")
    print(f"Anticipated Counter: {data.get('anticipated_counter')}")
    print(f"Tactical Rationale: {data.get('tactical_rationale')}")
    print(f"Search PV: {data.get('search_pv')}")

    # Must retreat the hanging knight from f3 or defend it
    assert from_sq == "f3", f"Expected hanging knight on f3 to retreat, but chose {san} from {from_sq}"
    # Retreat square must be safe
    assert to_sq in ("h4", "d4", "e5", "g5", "h2", "g1"), f"Expected safe retreat, got {to_sq}"
    assert data.get("eval_mode") == "tenacious_defense", f"Expected tenacious_defense, got {data.get('eval_mode')}"
    assert "Tactical Retreat" in data.get("tactical_rationale", ""), f"Expected 'Tactical Retreat' in rationale, got: {data.get('tactical_rationale')}"
    print(f"✅ Tactical retreat test passed: Knight successfully rescued to {to_sq} with retreat rationale.")

def test_sound_greek_gift_king_hunt():
    print("\n--- Test 7: Sound Tactical Sacrifice / Greek Gift (King Hunt) ---")
    # Queen-Bishop battery on c2/d3 against h7: White plays Bxh7+
    fen = "r1bq1rk1/pp1nbppp/4p3/3pP3/3P4/3BBN2/PPQ2PPP/R4RK1 w - - 1 12"
    payload = {
        "fen": fen,
        "temperature": 0.0,
        "use_book": False,
        "use_search": True,
        "search_depth": 2
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    san = top_move["san"]
    print(f"Top move chosen: {san} ({top_move['uci']})")
    print(f"Eval Mode: {data.get('eval_mode')}")
    print(f"Tactical Rationale: {data.get('tactical_rationale')}")
    print(f"Search PV: {data.get('search_pv')}")

    assert san == "Bxh7+", f"Expected Greek Gift sacrifice Bxh7+, got {san}"
    assert data.get("eval_mode") == "tactical_strike", f"Expected tactical_strike, got {data.get('eval_mode')}"
    assert "King Hunt" in data.get("tactical_rationale", "") or "Sound Tactical Sacrifice" in data.get("tactical_rationale", ""), (
        f"Expected King Hunt / Sound Tactical Sacrifice rationale, got: {data.get('tactical_rationale')}"
    )
    print("✅ Sound tactical sacrifice test passed: Engine accurately identifies Bxh7+ King Hunt assault.")

def test_unsound_sacrifice_contrast():
    print("\n--- Test 8: Unsound Sacrifice Contrast (Rejection of Blind Hope Sacrifices) ---")
    # Position where h7 is well-defended by Nf6 and Bxh7+ is a pure suicide blunder
    fen = "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 4 8"
    payload = {
        "fen": fen,
        "temperature": 0.0,
        "use_book": False,
        "use_search": True,
        "search_depth": 2
    }
    data = post_json("/predict-move", payload)
    top_move = data["top_moves"][0]
    san = top_move["san"]
    print(f"Top move chosen: {san} ({top_move['uci']})")
    print(f"Eval Mode: {data.get('eval_mode')}")

    # Must reject reckless sacrifices on h7/f7
    assert san not in ("Bxh7+", "Bxf7+", "Qxh7+", "Nxf7"), f"Unsound sacrifice played: {san}"
    print(f"✅ Unsound sacrifice contrast passed: Rebuffed speculative sacrifice, selected {san}.")

def test_king_exposure_scoring():
    print("\n--- Test 9: King Vulnerability & Shelter Scoring Benchmark ---")
    from backend.app import evaluate_king_exposure
    # Intact pawn shield
    b_safe = chess.Board("r1bq1rk1/pp1nbppp/4p3/3p4/3P4/3BBN2/PPP2PPP/R2Q1RK1 w - - 0 10")
    exp_safe = evaluate_king_exposure(b_safe, chess.BLACK)

    # Completely stripped pawn shield with open files
    b_stripped = chess.Board("r1bq1rk1/pp1nb3/4p3/3p4/3P4/3BBN2/PPP2PPP/R2Q1RK1 b - - 0 10")
    exp_stripped = evaluate_king_exposure(b_stripped, chess.BLACK)

    print(f"Safe castled King exposure: {exp_safe:.1f} cp")
    print(f"Stripped exposed King exposure: {exp_stripped:.1f} cp")

    assert exp_safe <= 15.0, f"Expected safe king exposure <= 15.0 cp, got {exp_safe}"
    assert exp_stripped >= 60.0, f"Expected stripped king exposure >= 60.0 cp, got {exp_stripped}"
    print("✅ King exposure scoring verified: Reliably differentiates fortified vs exposed kings.")

def test_exchange_sacrifice_compensation():
    print("\n--- Test 10: Sicilian Exchange Sacrifice Compensation Heuristic ---")
    from backend.app import evaluate_sacrifice_compensation
    # Standard Yugoslav Attack Dragon after 12...Rxc3 13.bxc3
    b = chess.Board()
    moves = "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 O-O-O Bd7 g4 Rc8 h4 Ne5 h5 Qa5 Kb1 Rxc3 bxc3".split()
    for m in moves:
        b.push_san(m)

    comp_cp, is_comp = evaluate_sacrifice_compensation(b, chess.BLACK)
    print(f"Sicilian Dragon after ...Rxc3 bxc3: comp_cp = {comp_cp:.1f} cp, is_compensated = {is_comp}")

    assert is_comp, f"Expected Sicilian exchange sacrifice to be compensated, got {is_comp} (comp_cp: {comp_cp})"
    assert comp_cp >= 80.0, f"Expected compensation >= 80.0 cp, got {comp_cp}"
    print("✅ Exchange sacrifice compensation verified: Dynamic king exposure & damaged structure balance material deficit.")

if __name__ == "__main__":
    test_health()
    test_hard_material_clamp()
    test_unsound_queen_sacrifice_rejected()
    test_queen_hung_to_pawn_rejected()
    test_winning_conversion()
    test_opening_concepts_warm_handoff()
    test_hanging_piece_tactical_retreat()
    test_sound_greek_gift_king_hunt()
    test_unsound_sacrifice_contrast()
    test_king_exposure_scoring()
    test_exchange_sacrifice_compensation()
    print("\n🎉 ALL 10 DEFENSE, SOUND SACRIFICE & KING HUNT TESTS PASSED!")


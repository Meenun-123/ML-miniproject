import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Chess } from 'chess.js';
import ChessboardPanel from './components/ChessboardPanel';
import PuzzleSelector from './components/PuzzleSelector';
import SolveControls from './components/SolveControls';
import DefenderPanel from './components/DefenderPanel';
import SpectatorPanel from './components/SpectatorPanel';
import TelemetryPanel from './components/TelemetryPanel';
import { Bot, Shield, Swords, RotateCcw } from 'lucide-react';
import { 
  playMoveSound, 
  playCheckSound, 
  playMateSound, 
  playVictorySound 
} from './utils/soundEffects';
import { computeCustomSquareStyles } from './utils/chessHighlights';

const PRESET_PUZZLES = [
  {
    name: 'Back-Rank Mate',
    type: 'Mate in 1',
    fen: '6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1',
    description: 'White rook delivers back-rank checkmate',
    matePlies: 1
  },
  {
    name: "Anastasia's Mate",
    type: 'Mate in 2',
    fen: '4r1k1/1p3ppp/8/8/8/8/2B2PPP/2R1N1K1 w - - 0 1',
    description: 'Knight and rook coordinate for a devastating mate',
    matePlies: 3
  },
  {
    name: 'Smothered Mate',
    type: 'Mate in 2', 
    fen: '6rk/6pp/8/6N1/8/8/8/4K2R w K - 0 1',
    description: 'Knight delivers mate on a suffocated king',
    matePlies: 3
  },
  {
    name: 'Opera House Mate',
    type: 'Mate in 2',
    fen: '4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 0',
    description: 'Queen sacrifice leads to back-rank pin and mate',
    matePlies: 3
  },
  {
    name: "Black Back-Rank Mate",
    type: 'Mate in 1',
    fen: '4k3/8/8/8/8/8/5PPP/r3K3 b - - 0 1',
    description: 'Black rook delivers back-rank checkmate against White King',
    matePlies: 1
  }
];

const API_BASE = 'http://localhost:8000';

function App() {
  // Active Puzzle & Board State
  const [startingFen, setStartingFen] = useState(PRESET_PUZZLES[0].fen);
  const [currentPuzzlePlies, setCurrentPuzzlePlies] = useState(PRESET_PUZZLES[0].matePlies);
  const [game, setGame] = useState(() => new Chess(PRESET_PUZZLES[0].fen));
  const [fen, setFen] = useState(PRESET_PUZZLES[0].fen);
  const [customFen, setCustomFen] = useState('');
  
  // Game Mode: 'auto' | 'defender'
  const [mode, setMode] = useState('auto');
  const [boardOrientation, setBoardOrientation] = useState('white');

  // Auto-Rollout Solver State
  const [solution, setSolution] = useState(null);
  const [currentStep, setCurrentStep] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSolving, setIsSolving] = useState(false);

  // Defender Mode State
  const [defenderColor, setDefenderColor] = useState('b');
  const [defenderStatus, setDefenderStatus] = useState('user_turn'); // 'user_turn' | 'machine_thinking' | 'checkmate' | 'refuted'
  const [pliesSurvived, setPliesSurvived] = useState(0);
  const [targetPlies, setTargetPlies] = useState(1);
  const [matchHistory, setMatchHistory] = useState([]);
  const [latestMachineMove, setLatestMachineMove] = useState(null);
  const [banner, setBanner] = useState(null);

  // Spectator Battle Arena State
  const [spectatorStatus, setSpectatorStatus] = useState('idle'); // 'idle' | 'playing' | 'paused' | 'game_over'
  const [spectatorElo, setSpectatorElo] = useState(2400);
  const [spectatorAcpl, setSpectatorAcpl] = useState(0.0);
  const [stockfishLevel, setStockfishLevel] = useState(16);
  const [stockfishElo, setStockfishElo] = useState(2430);
  const [spectatorPlies, setSpectatorPlies] = useState(0);
  const [lastModelCpl, setLastModelCpl] = useState(null);
  const [spectatorHistoryCpl, setSpectatorHistoryCpl] = useState([]);
  const [spectatorMatchHistory, setSpectatorMatchHistory] = useState([]);
  const [openingTemperature, setOpeningTemperature] = useState(1.0);
  const [useBook, setUseBook] = useState(true);
  const [useMinimaxSearch, setUseMinimaxSearch] = useState(true);
  const [activeSpectatorRole, setActiveSpectatorRole] = useState('model'); // 'model' | 'stockfish'
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [isStepLoading, setIsStepLoading] = useState(false);
  const [moveExplanation, setMoveExplanation] = useState(null);
  const [modelColor, setModelColor] = useState('w'); // 'w' | 'b'
  const [telemetryHistory, setTelemetryHistory] = useState([]);
  const [isEvaluatingFullGame, setIsEvaluatingFullGame] = useState(false);
  const [latestReview, setLatestReview] = useState(null);
  const spectatorLoopRef = useRef(null);

  // Tactical Arrows & Telemetry
  const [arrows, setArrows] = useState([]);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);

  const playbackRef = useRef(null);

  // Dynamic Square Highlights: Checked King (crimson), Source (amber), Tactical Destination (emerald)
  const activeHighlight = useMemo(() => {
    if (arrows && arrows.length > 0) {
      return { from_sq: arrows[0][0], to_sq: arrows[0][1] };
    }
    if (latestMachineMove) {
      return { from_sq: latestMachineMove.from_sq, to_sq: latestMachineMove.to_sq };
    }
    return null;
  }, [arrows, latestMachineMove]);

  const customSquareStyles = useMemo(() => {
    return computeCustomSquareStyles({
      game,
      activeMove: activeHighlight
    });
  }, [game, activeHighlight]);

  // Backend Health Ping
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setBackendStatus(data);
    } catch (err) {
      setBackendStatus({ status: 'error', error: err.message });
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Arrow Helper
  const updateArrows = useCallback((stepIdx, seq) => {
    if (!seq || seq.length === 0) {
      setArrows([]);
      return;
    }
    
    const newArrows = [];
    if (stepIdx >= 0 && stepIdx < seq.length) {
      const move = seq[stepIdx];
      newArrows.push([move.from_sq, move.to_sq, 'rgba(34, 197, 94, 0.85)']);
    }
    for (let i = Math.max(0, stepIdx + 1); i < seq.length; i++) {
      const move = seq[i];
      newArrows.push([move.from_sq, move.to_sq, 'rgba(255, 255, 255, 0.35)']);
    }
    setArrows(newArrows);
  }, []);

  // Trigger Machine's Turn in Defender Mode
  const triggerMachineTurn = useCallback(async (currentGame, pliesCount, historyList) => {
    setDefenderStatus('machine_thinking');
    
    // Realistic 260ms thinking cadence
    const delayPromise = new Promise(resolve => setTimeout(resolve, 260));

    try {
      const currentFen = currentGame.fen();
      const fetchPromise = fetch(`${API_BASE}/predict-move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: currentFen })
      }).then(res => res.json());

      const [data] = await Promise.all([fetchPromise, delayPromise]);
      setInferenceTime(data.inference_time_ms);

      if (!data.top_moves || data.top_moves.length === 0) {
        // Machine has no legal moves -> Fortress Held / Refuted!
        playVictorySound();
        setDefenderStatus('refuted');
        setBanner({
          type: 'victory',
          emoji: '🛡️',
          title: 'Fortress Held!',
          subtitle: 'The machine ran out of attacking resources. You successfully refuted the mate sequence!'
        });
        return;
      }

      const bestMove = data.top_moves[0];
      const newGame = new Chess(currentGame.fen());
      const moveResult = newGame.move({
        from: bestMove.from_sq,
        to: bestMove.to_sq,
        promotion: bestMove.uci.length > 4 ? bestMove.uci[4] : 'q'
      });

      if (!moveResult) {
        console.error("Machine predicted illegal move:", bestMove);
        return;
      }

      const isCheck = newGame.inCheck();
      const isMate = newGame.isCheckmate();
      const isDraw = newGame.isDraw() || newGame.isStalemate();

      // Sound feedback
      if (isMate) {
        playMateSound();
      } else if (isCheck) {
        playCheckSound();
      } else {
        playMoveSound();
      }

      setGame(newGame);
      setFen(newGame.fen());
      setArrows([[bestMove.from_sq, bestMove.to_sq, 'rgba(255, 170, 0, 0.85)']]);
      setLatestMachineMove(bestMove);

      const nextPlies = pliesCount + 1;
      setPliesSurvived(nextPlies);

      const newHistory = [
        ...historyList,
        {
          role: 'machine',
          san: bestMove.san,
          uci: bestMove.uci,
          from_sq: bestMove.from_sq,
          to_sq: bestMove.to_sq,
          confidence: bestMove.confidence
        }
      ];
      setMatchHistory(newHistory);

      if (isMate) {
        setDefenderStatus('checkmate');
        setBanner({
          type: 'checkmate',
          emoji: '💥',
          title: 'Checkmate Delivered!',
          subtitle: `SE-ResNet found the lethal blow in ${nextPlies} plies.`
        });
      } else if (isDraw) {
        playVictorySound();
        setDefenderStatus('refuted');
        setBanner({
          type: 'victory',
          emoji: '🛡️',
          title: 'Fortress Held!',
          subtitle: 'Position drawn or stalemated! You withstood the attack.'
        });
      } else {
        setDefenderStatus('user_turn');
      }
    } catch (err) {
      console.error("Error executing machine turn:", err);
      setDefenderStatus('user_turn');
    }
  }, []);

  // Initialize Defender Session
  const initDefenderSession = useCallback((startFenStr, expectedPlies) => {
    try {
      const cleanGame = new Chess(startFenStr);
      const attackerCol = cleanGame.turn(); // 'w' or 'b'
      const defCol = attackerCol === 'w' ? 'b' : 'w';

      setGame(cleanGame);
      setFen(cleanGame.fen());
      setDefenderColor(defCol);
      setBoardOrientation(defCol === 'w' ? 'white' : 'black');
      setPliesSurvived(0);
      setTargetPlies(expectedPlies || 3);
      setMatchHistory([]);
      setLatestMachineMove(null);
      setBanner(null);
      setArrows([]);
      setSolution(null);
      setCurrentStep(-1);
      setIsPlaying(false);
      if (playbackRef.current) clearInterval(playbackRef.current);

      if (cleanGame.turn() === attackerCol) {
        triggerMachineTurn(cleanGame, 0, []);
      } else {
        setDefenderStatus('user_turn');
      }
    } catch (e) {
      console.error('Invalid FEN initialization:', e);
    }
  }, [triggerMachineTurn]);

  // Fetch initial telemetry match trajectory on mount (persistent rehydration)
  useEffect(() => {
    fetch(`${API_BASE}/telemetry-history`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data.history && Array.isArray(data.history)) {
          setTelemetryHistory(data.history);
        }
      })
      .catch(err => console.warn("Failed to fetch initial telemetry history:", err));
  }, []);

  // Record match end to backend telemetry and update live charts via full-game evaluation
  const recordMatchEnd = useCallback(async (result, movesCount, acpl, elo, currentHistory = [], gameInstance = null) => {
    const targetGame = gameInstance || game;
    let pgnMoves = [];
    try {
      if (Array.isArray(currentHistory) && currentHistory.length > 0) {
        pgnMoves = currentHistory.map(h => h.san).filter(Boolean);
      }
      if ((!pgnMoves || pgnMoves.length === 0) && targetGame && typeof targetGame.history === 'function') {
        pgnMoves = targetGame.history();
      }
    } catch (e) {
      console.warn("Could not extract PGN history:", e);
    }

    if (pgnMoves && pgnMoves.length > 0) {
      setIsEvaluatingFullGame(true);
      try {
        const res = await fetch(`${API_BASE}/evaluate-full-game`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            pgn_moves: pgnMoves,
            model_color: modelColor === 'w' ? 'white' : 'black',
            result: result,
          }),
        });
        const data = await res.json();
        if (data.history && Array.isArray(data.history)) {
          setTelemetryHistory(data.history);
        }
        if (data.record) {
          setLatestReview(data.record);
        }
      } catch (e) {
        console.warn("Failed to evaluate full game with Stockfish:", e);
      } finally {
        setIsEvaluatingFullGame(false);
      }
    } else {
      try {
        const blunders = currentHistory.filter(h => h.role === 'model' && h.cpl !== null && h.cpl >= 100).length;
        const res = await fetch(`${API_BASE}/record-game-stats`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            moves_count: movesCount,
            acpl: acpl,
            final_elo: elo,
            result: result,
            blunders: blunders,
          }),
        });
        const data = await res.json();
        if (data.history && Array.isArray(data.history)) {
          setTelemetryHistory(data.history);
        }
        if (data.record) {
          setLatestReview(data.record);
        }
      } catch (e) {
        console.warn("Failed to record game stats:", e);
      }
    }
  }, [game, modelColor]);

  // Initialize Spectator Session
  const initSpectatorSession = useCallback((startFenStr = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1') => {
    try {
      const cleanGame = new Chess(startFenStr);
      setGame(cleanGame);
      setFen(cleanGame.fen());
      setBoardOrientation(modelColor === 'w' ? 'white' : 'black');
      setSpectatorStatus('idle');
      setIsAutoPlaying(false);
      setLatestReview(null);
      setIsStepLoading(false);
      setMoveExplanation(null);
      setSpectatorPlies(0);
      setSpectatorElo(2400);
      setSpectatorAcpl(0.0);
      setStockfishLevel(16);
      setStockfishElo(2430);
      setLastModelCpl(null);
      setSpectatorHistoryCpl([]);
      setSpectatorMatchHistory([]);
      setActiveSpectatorRole(cleanGame.turn() === modelColor ? 'model' : 'stockfish');
      setArrows([]);
      setBanner(null);
      setSolution(null);
      setCurrentStep(-1);
      setIsPlaying(false);
      if (playbackRef.current) clearInterval(playbackRef.current);
      if (spectatorLoopRef.current) clearTimeout(spectatorLoopRef.current);
    } catch (e) {
      console.error('Invalid FEN initialization for spectator:', e);
    }
  }, [modelColor]);

  // Swap Sides (Model plays White or Black)
  const swapSides = useCallback(() => {
    const nextColor = modelColor === 'w' ? 'b' : 'w';
    setModelColor(nextColor);
    setBoardOrientation(nextColor === 'w' ? 'white' : 'black');
    setMoveExplanation(null);
    setIsAutoPlaying(false);
    setActiveSpectatorRole(game.turn() === nextColor ? 'model' : 'stockfish');
  }, [modelColor, game]);

  // Spectator Step: Executes 1 ply for either Model or Stockfish
  const runSpectatorStep = useCallback(async (currentGame, currentHistory, cplHistory, pliesCount, temp, bookOpt = true, searchOpt = true, currentModelColor = 'w') => {
    if (currentGame.isGameOver()) {
      setSpectatorStatus('game_over');
      setIsAutoPlaying(false);
      if (currentGame.isCheckmate()) {
        playMateSound();
        const losingColor = currentGame.turn();
        const isModelDefeated = losingColor === currentModelColor;
        const winner = isModelDefeated ? 'Stockfish 17' : 'SE-ResNet-8';
        setBanner({
          type: 'checkmate',
          emoji: '💥',
          title: 'Checkmate Delivered!',
          subtitle: `${winner} wins the battle in ${pliesCount} plies.`
        });
        recordMatchEnd(isModelDefeated ? 'loss' : 'win', pliesCount, spectatorAcpl, spectatorElo, currentHistory, currentGame);
      } else {
        playVictorySound();
        setBanner({
          type: 'victory',
          emoji: '⚖️',
          title: 'Game Drawn / Stalemate',
          subtitle: `Battle ended in a draw after ${pliesCount} plies.`
        });
        recordMatchEnd('draw', pliesCount, spectatorAcpl, spectatorElo, currentHistory, currentGame);
      }
      return null;
    }

    const turn = currentGame.turn();
    const isModelTurn = (turn === currentModelColor);

    if (isModelTurn) {
      setActiveSpectatorRole('model');
      try {
        const res = await fetch(`${API_BASE}/predict-move`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            fen: currentGame.fen(),
            temperature: temp,
            ply: pliesCount + 1,
            use_book: bookOpt,
            use_search: searchOpt,
            search_depth: 2,
          })
        });
        const data = await res.json();
        if (!data.top_moves || data.top_moves.length === 0) return null;

        const bestMove = data.top_moves[0];
        const newGame = new Chess(currentGame.fen());
        const move = newGame.move({ 
          from: bestMove.from_sq, 
          to: bestMove.to_sq, 
          promotion: bestMove.uci.length > 4 ? bestMove.uci[4] : 'q' 
        });
        if (!move) return null;

        if (newGame.isCheckmate()) playMateSound();
        else if (newGame.inCheck()) playCheckSound();
        else playMoveSound();

        const isBook = Boolean(data.is_book_move || bestMove.is_book);
        const arrowColor = isBook
          ? 'rgba(168, 85, 247, 0.9)' // Purple for Opening Book
          : 'rgba(34, 197, 94, 0.85)'; // Emerald for Model

        setGame(newGame);
        setFen(newGame.fen());
        setArrows([[bestMove.from_sq, bestMove.to_sq, arrowColor]]);

        const nextPlies = pliesCount + 1;
        setSpectatorPlies(nextPlies);

        // Detailed reasoning commentary
        const sourceLabel = isBook 
          ? '📖 Book Theory (Polyglot)' 
          : (data.source === 'search' ? '🧠 Neural Policy + Minimax D2' : '⚡ Neural Policy Head');

        let detailText = data.tactical_rationale || '';
        if (!detailText) {
          if (isBook) {
            detailText = `Standard grandmaster tournament preparation from Titans polyglot book. Maintains theoretical equity without tactical risk.`;
          } else if (data.source === 'search') {
            detailText = `Tactical lookahead evaluated to depth 2 with alpha-beta pruning, combining neural policy priors with SE-ResNet leaf scoring.`;
          } else {
            detailText = `Direct tactical policy head selection with ${((bestMove.confidence || 0.9) * 100).toFixed(1)}% certainty.`;
          }
        }

        setMoveExplanation({
          role: 'model',
          player: `SE-ResNet-8 (${currentModelColor === 'w' ? 'White' : 'Black'})`,
          san: move.san,
          uci: move.from + move.to + (move.promotion || ''),
          source: sourceLabel,
          is_book: isBook,
          metric: isBook ? 'Book Line' : `${((bestMove.confidence || 0.9) * 100).toFixed(1)}% conf`,
          detail: detailText,
          eval_mode: data.eval_mode,
          anticipated_counter: data.anticipated_counter,
          tactical_rationale: data.tactical_rationale,
          search_pv: data.search_pv || [],
        });

        const newHistory = [
          ...currentHistory,
          {
            role: 'model',
            san: move.san,
            uci: move.from + move.to + (move.promotion || ''),
            cpl: null,
            fen_before: currentGame.fen(),
            is_book: isBook,
            source: data.source || bestMove.source || 'model',
            eval_mode: data.eval_mode,
            tactical_rationale: data.tactical_rationale,
          }
        ];
        setSpectatorMatchHistory(newHistory);

        if (newGame.isGameOver()) {
          setSpectatorStatus('game_over');
          setIsAutoPlaying(false);
          if (newGame.isCheckmate()) {
            playMateSound();
            setBanner({
              type: 'checkmate',
              emoji: '👑',
              title: 'Model Victorious!',
              subtitle: `SE-ResNet-8 delivered checkmate against Stockfish in ${nextPlies} plies!`
            });
            recordMatchEnd('win', nextPlies, spectatorAcpl, spectatorElo, newHistory, newGame);
          } else {
            playVictorySound();
            setBanner({
              type: 'victory',
              emoji: '⚖️',
              title: 'Match Drawn',
              subtitle: `Match ended in a draw after ${nextPlies} plies.`
            });
            recordMatchEnd('draw', nextPlies, spectatorAcpl, spectatorElo, newHistory, newGame);
          }
          return null;
        }

        return { newGame, newHistory, cplHistory, nextPlies };
      } catch (err) {
        console.error("Spectator Model Step Error:", err);
        return null;
      }
    } else {
      setActiveSpectatorRole('stockfish');
      try {
        const lastModelEntry = currentHistory.length > 0 && currentHistory[currentHistory.length - 1].role === 'model'
          ? currentHistory[currentHistory.length - 1]
          : null;

        const evalFen = lastModelEntry ? lastModelEntry.fen_before : currentGame.fen();
        const evalUci = lastModelEntry ? lastModelEntry.uci : null;

        const res = await fetch(`${API_BASE}/adaptive-step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            fen: evalFen,
            model_move_uci: evalUci,
            history_cpl: cplHistory,
            ply: pliesCount,
            temperature: temp
          })
        });
        const data = await res.json();

        setSpectatorElo(data.estimated_elo);
        setSpectatorAcpl(data.rolling_acpl);
        setStockfishLevel(data.stockfish_skill_level);
        setStockfishElo(data.stockfish_elo);
        setLastModelCpl(data.model_cpl);
        setSpectatorHistoryCpl(data.history_cpl);

        const annotatedHistory = currentHistory.map((h, i) => {
          if (i === currentHistory.length - 1 && h.role === 'model') {
            return { ...h, cpl: data.model_cpl };
          }
          return h;
        });

        if (!data.counter_move) {
          if (data.is_checkmate) {
            setSpectatorStatus('game_over');
            setIsAutoPlaying(false);
            playMateSound();
            setBanner({
              type: 'checkmate',
              emoji: '👑',
              title: 'Model Victorious!',
              subtitle: 'SE-ResNet checkmated Stockfish!'
            });
            recordMatchEnd('win', pliesCount, data.rolling_acpl, data.estimated_elo, currentHistory, currentGame);
          }
          return null;
        }

        const counterMove = data.counter_move;
        const newGame = new Chess(data.fen_after);

        if (data.is_checkmate) playMateSound();
        else if (data.is_check) playCheckSound();
        else playMoveSound();

        setGame(newGame);
        setFen(data.fen_after);
        setArrows([[counterMove.from_sq, counterMove.to_sq, 'rgba(6, 182, 212, 0.9)']]); // Electric cyan for Stockfish

        const nextPlies = pliesCount + 1;
        setSpectatorPlies(nextPlies);

        // Detailed reasoning commentary for Stockfish
        const sfColorName = currentModelColor === 'w' ? 'Black' : 'White';
        let sfDetail = `Stockfish calibrated to Level ${data.stockfish_skill_level} (~${data.stockfish_elo} Elo) to match SE-ResNet's rolling ACPL of ${data.rolling_acpl.toFixed(1)} cp.`;
        if (data.model_cpl !== null && data.model_cpl !== undefined && lastModelEntry) {
          if (data.model_cpl <= 15) {
            sfDetail += ` Graded Model's ${lastModelEntry.san} as top engine choice (${data.model_cpl} cp loss).`;
          } else if (data.model_cpl <= 50) {
            sfDetail += ` Graded Model's ${lastModelEntry.san} as solid (+${data.model_cpl} cp loss).`;
          } else if (data.model_cpl <= 100) {
            sfDetail += ` Capitalizing on a slight inaccuracy (+${data.model_cpl} cp loss) by the model.`;
          } else {
            sfDetail += ` Exploiting a sharp tactical blunder (+${data.model_cpl} cp loss) by the model!`;
          }
        }

        setMoveExplanation({
          role: 'stockfish',
          player: `Stockfish 17 (${sfColorName})`,
          san: counterMove.san,
          uci: counterMove.uci,
          source: `⚔️ Stockfish Calibrated (Lvl ${data.stockfish_skill_level})`,
          is_book: false,
          metric: data.model_cpl !== null ? `Model cost: ${data.model_cpl} cp` : `${data.stockfish_elo} Elo`,
          detail: sfDetail
        });

        const finalHistory = [
          ...annotatedHistory,
          {
            role: 'stockfish',
            san: counterMove.san,
            uci: counterMove.uci,
            cpl: null
          }
        ];
        setSpectatorMatchHistory(finalHistory);

        if (data.is_checkmate) {
          setSpectatorStatus('game_over');
          setIsAutoPlaying(false);
          playMateSound();
          setBanner({
            type: 'checkmate',
            emoji: '💥',
            title: 'Stockfish Checkmate!',
            subtitle: `Adaptive Stockfish (Lvl ${data.stockfish_skill_level}) delivered mate in ${nextPlies} plies.`
          });
          recordMatchEnd('loss', nextPlies, data.rolling_acpl, data.estimated_elo, finalHistory, newGame);
          return null;
        }

        if (data.is_draw) {
          setSpectatorStatus('game_over');
          setIsAutoPlaying(false);
          playVictorySound();
          setBanner({
            type: 'victory',
            emoji: '⚖️',
            title: 'Match Drawn',
            subtitle: `Game concluded in a draw after ${nextPlies} plies.`
          });
          recordMatchEnd('draw', nextPlies, data.rolling_acpl, data.estimated_elo, finalHistory, newGame);
          return null;
        }

        return { newGame, newHistory: finalHistory, cplHistory: data.history_cpl, nextPlies };
      } catch (err) {
        console.error("Spectator Stockfish Step Error:", err);
        return null;
      }
    }
  }, []);

  const stepSpectator = useCallback(async () => {
    if (spectatorStatus === 'game_over' || isStepLoading) return;
    setIsStepLoading(true);
    try {
      await runSpectatorStep(
        game,
        spectatorMatchHistory,
        spectatorHistoryCpl,
        spectatorPlies,
        openingTemperature,
        useBook,
        useMinimaxSearch,
        modelColor
      );
    } finally {
      setIsStepLoading(false);
    }
  }, [game, spectatorMatchHistory, spectatorHistoryCpl, spectatorPlies, openingTemperature, useBook, useMinimaxSearch, spectatorStatus, isStepLoading, modelColor, runSpectatorStep]);

  // Spectator 1200ms Auto-Play Loop
  useEffect(() => {
    let timer = null;
    if (mode === 'spectator' && isAutoPlaying && spectatorStatus !== 'game_over' && !isStepLoading) {
      timer = setTimeout(async () => {
        await stepSpectator();
      }, 1200);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [mode, isAutoPlaying, spectatorStatus, isStepLoading, stepSpectator]);

  // Load a puzzle
  const loadPuzzle = useCallback((newFen, matePlies = 3) => {
    try {
      const testGame = new Chess(newFen);
      setStartingFen(newFen);
      setCurrentPuzzlePlies(matePlies);

      if (mode === 'defender') {
        initDefenderSession(newFen, matePlies);
      } else if (mode === 'spectator') {
        initSpectatorSession(newFen);
      } else {
        setGame(testGame);
        setFen(testGame.fen());
        setBoardOrientation('white');
        setSolution(null);
        setCurrentStep(-1);
        setIsPlaying(false);
        setArrows([]);
        setBanner(null);
        setInferenceTime(null);
        if (playbackRef.current) clearInterval(playbackRef.current);
      }
    } catch (e) {
      console.error('Invalid FEN:', newFen);
    }
  }, [mode, initDefenderSession, initSpectatorSession]);

  const handleLoadCustom = () => {
    if (customFen) loadPuzzle(customFen, 3);
  };

  // Reset board
  const resetBoard = useCallback(() => {
    if (mode === 'defender') {
      initDefenderSession(startingFen, currentPuzzlePlies);
    } else if (mode === 'spectator') {
      initSpectatorSession('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    } else {
      const startFen = solution?.initial_fen || startingFen;
      try {
        const cleanGame = new Chess(startFen);
        setGame(cleanGame);
        setFen(cleanGame.fen());
      } catch (e) {}
      setCurrentStep(-1);
      setIsPlaying(false);
      if (playbackRef.current) clearInterval(playbackRef.current);
      updateArrows(-1, solution?.sequence || []);
      setBanner(null);
    }
  }, [mode, startingFen, currentPuzzlePlies, initDefenderSession, initSpectatorSession, solution, updateArrows]);

  // Switch Mode
  const switchMode = (newMode) => {
    if (newMode === mode) return;
    setMode(newMode);
    setBanner(null);
    setArrows([]);
    setIsPlaying(false);
    setIsAutoPlaying(false);
    setMoveExplanation(null);
    if (playbackRef.current) clearInterval(playbackRef.current);
    if (spectatorLoopRef.current) clearTimeout(spectatorLoopRef.current);

    if (newMode === 'defender') {
      initDefenderSession(startingFen, currentPuzzlePlies);
    } else if (newMode === 'spectator') {
      initSpectatorSession(
        startingFen.includes('8/8') || startingFen.length < 50
          ? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
          : startingFen
      );
    } else {
      const cleanGame = new Chess(startingFen);
      setGame(cleanGame);
      setFen(cleanGame.fen());
      setBoardOrientation('white');
      setSolution(null);
      setCurrentStep(-1);
    }
  };

  // Auto-Rollout Solver
  const solveMate = async () => {
    setIsSolving(true);
    try {
      const res = await fetch(`${API_BASE}/solve-mate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: game.fen() })
      });
      
      if (!res.ok) throw new Error('API Error');
      const data = await res.json();
      data.initial_fen = game.fen();
      
      setSolution(data);
      setInferenceTime(data.inference_time_ms);
      setCurrentStep(-1);
      updateArrows(-1, data.sequence);
    } catch (err) {
      console.error('Failed to solve mate:', err);
    } finally {
      setIsSolving(false);
    }
  };

  const stepForward = useCallback(() => {
    if (!solution || !solution.sequence) return;
    
    setCurrentStep(prev => {
      const nextStep = prev + 1;
      if (nextStep >= solution.sequence.length) {
        setIsPlaying(false);
        if (playbackRef.current) clearInterval(playbackRef.current);
        return prev;
      }
      
      const move = solution.sequence[nextStep];
      try {
        const newGame = new Chess(game.fen());
        const promo = move.uci && move.uci.length === 5 ? move.uci[4] : 'q';
        newGame.move({ from: move.from_sq, to: move.to_sq, promotion: promo });
        
        if (move.is_mate) {
          playMateSound();
        } else if (move.is_check) {
          playCheckSound();
        } else {
          playMoveSound();
        }

        setGame(newGame);
        setFen(newGame.fen());
        updateArrows(nextStep, solution.sequence);
      } catch (e) {
        console.error("Invalid move in sequence", move);
      }
      return nextStep;
    });
  }, [game, solution, updateArrows]);

  const playSequence = useCallback(() => {
    if (!solution || !solution.sequence) return;
    if (currentStep >= solution.sequence.length - 1) {
      resetBoard();
      setTimeout(() => {
        setIsPlaying(true);
      }, 50);
      return;
    }
    setIsPlaying(true);
  }, [solution, currentStep, resetBoard]);

  const pausePlayback = useCallback(() => {
    setIsPlaying(false);
    if (playbackRef.current) clearInterval(playbackRef.current);
  }, []);

  useEffect(() => {
    if (isPlaying) {
      playbackRef.current = setInterval(() => {
        stepForward();
      }, 1200);
    } else {
      if (playbackRef.current) clearInterval(playbackRef.current);
    }
    
    return () => {
      if (playbackRef.current) clearInterval(playbackRef.current);
    };
  }, [isPlaying, stepForward]);

  // Handle Piece Drag & Drop
  const onPieceDrop = (sourceSquare, targetSquare, piece) => {
    // 1. Robust Pawn Promotion Detection
    const currentPiece = game.get(sourceSquare);
    const isPawnMove = currentPiece?.type === 'p';
    const toRank = targetSquare[1];
    const isPromotion = isPawnMove && (toRank === '8' || toRank === '1');

    let promotionPiece = undefined;
    if (isPromotion) {
      if (piece && piece.length >= 2 && ['q', 'r', 'b', 'n'].includes(piece[1].toLowerCase())) {
        promotionPiece = piece[1].toLowerCase();
      } else if (piece && ['q', 'r', 'b', 'n'].includes(piece.toLowerCase())) {
        promotionPiece = piece.toLowerCase();
      } else {
        promotionPiece = 'q';
      }
    }

    // 2. Defender Mode Handling
    if (mode === 'defender') {
      if (defenderStatus !== 'user_turn') return false;
      const pieceColor = currentPiece?.color || piece?.[0];
      if (pieceColor !== defenderColor) return false;

      try {
        const newGame = new Chess(game.fen());
        const move = newGame.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: promotionPiece
        });
        if (!move) return false;

        const isCheck = newGame.inCheck();
        const isMate = newGame.isCheckmate();
        const isDraw = newGame.isDraw() || newGame.isStalemate();

        if (isMate) playVictorySound();
        else if (isCheck) playCheckSound();
        else playMoveSound();

        setGame(newGame);
        setFen(newGame.fen());
        setArrows([[sourceSquare, targetSquare, 'rgba(6,182,212,0.9)']]);

        const nextPlies = pliesSurvived + 1;
        setPliesSurvived(nextPlies);

        const newHistory = [
          ...matchHistory,
          {
            role: 'user',
            san: move.san,
            uci: move.from + move.to + (move.promotion || ''),
            from_sq: sourceSquare,
            to_sq: targetSquare
          }
        ];
        setMatchHistory(newHistory);

        if (isMate) {
          setDefenderStatus('refuted');
          setBanner({
            type: 'victory',
            emoji: '👑',
            title: 'Counter-Checkmate!',
            subtitle: 'Brilliant! You counter-attacked and checkmated the machine!'
          });
          return true;
        }

        if (isDraw) {
          playVictorySound();
          setDefenderStatus('refuted');
          setBanner({
            type: 'victory',
            emoji: '🛡️',
            title: 'Fortress Held!',
            subtitle: 'You forced a draw/stalemate, successfully holding off the attack!'
          });
          return true;
        }

        triggerMachineTurn(newGame, nextPlies, newHistory);
        return true;
      } catch (e) {
        console.error("Defender move error:", e);
        return false;
      }
    }

    // 3. Auto Mode Free Play
    try {
      const newGame = new Chess(game.fen());
      const move = newGame.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: promotionPiece,
      });
      if (move === null) return false;

      if (newGame.isCheckmate()) playMateSound();
      else if (newGame.inCheck()) playCheckSound();
      else playMoveSound();

      setGame(newGame);
      setFen(newGame.fen());
      setSolution(null);
      setCurrentStep(-1);
      setArrows([[sourceSquare, targetSquare, 'rgba(34, 197, 94, 0.85)']]);
      return true;
    } catch (e) {
      console.error("Free play move error:", e);
      return false;
    }
  };

  const onPromotionPieceSelect = (piece, promoteFromSquare, promoteToSquare) => {
    return true;
  };

  // Draggable filter: strictly allow dragging only the active side's pieces
  const isDraggablePiece = useCallback(({ piece }) => {
    if (isPlaying) return false;
    if (mode === 'spectator') return false;

    if (mode === 'defender') {
      if (defenderStatus !== 'user_turn') return false;
      return piece[0] === defenderColor;
    }

    // In Auto/Free mode, only allow the side whose turn it is to move
    return piece[0] === game.turn();
  }, [isPlaying, mode, defenderStatus, defenderColor, game]);

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col items-center py-7 px-4 sm:px-6 relative overflow-x-hidden selection:bg-emerald-500/30">
      {/* Ambient Lighting Orbs */}
      <div className="fixed top-0 left-1/4 w-[28rem] h-[28rem] bg-emerald-500/10 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="fixed bottom-0 right-1/4 w-[28rem] h-[28rem] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -z-10" />

      {/* Top Header */}
      <div className="max-w-[1600px] w-full mb-6 md:mb-8 flex flex-col md:flex-row md:items-center justify-between gap-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
              Chess Mate Vision
            </h1>
            <span className="text-[11px] font-mono font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 shadow-sm">
              SE-ResNet-8
            </span>
          </div>
          <p className="text-slate-400 text-sm mt-1">
            Deep CNN Tactical Pattern Recognition & Interactive Battle Arena
          </p>
        </div>

        {/* Mode Selector Toggle */}
        <div className="flex flex-wrap bg-neutral-900/80 backdrop-blur-xl p-1.5 rounded-2xl border border-neutral-800/80 shadow-2xl self-start md:self-auto gap-1.5">
          <button
            onClick={() => switchMode('auto')}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl font-bold text-xs sm:text-sm transition-all ${
              mode === 'auto'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-950/50 border border-emerald-400/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-neutral-800/60'
            }`}
          >
            <Bot className="w-4 h-4" />
            <span>Machine Auto-Rollout</span>
          </button>
          <button
            onClick={() => switchMode('defender')}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl font-bold text-xs sm:text-sm transition-all ${
              mode === 'defender'
                ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-950/50 border border-cyan-400/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-neutral-800/60'
            }`}
          >
            <Shield className="w-4 h-4" />
            <span>Play as Defender</span>
          </button>
          <button
            onClick={() => switchMode('spectator')}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl font-bold text-xs sm:text-sm transition-all ${
              mode === 'spectator'
                ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white shadow-lg shadow-amber-950/50 border border-amber-400/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-neutral-800/60'
            }`}
          >
            <Swords className="w-4 h-4" />
            <span>Spectator Battle</span>
          </button>
        </div>
      </div>
      
      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-6 min-h-screen p-4 md:p-6 lg:p-8 max-w-[1600px] w-full mx-auto items-start">
        {/* Left Column: Chessboard Panel & HUD */}
        <div className="col-span-12 lg:col-span-7 xl:col-span-7 flex flex-col items-center gap-4 w-full">
          <ChessboardPanel 
            position={fen} 
            onPieceDrop={onPieceDrop} 
            onPromotionPieceSelect={onPromotionPieceSelect}
            customArrows={arrows}
            boardOrientation={boardOrientation}
            isDraggablePiece={isDraggablePiece}
            customSquareStyles={customSquareStyles}
            turn={game.turn()}
            isCheck={game.inCheck()}
            isCheckmate={game.isCheckmate()}
            mode={mode}
            defenderColor={defenderColor}
            modelColor={modelColor}
            isMachineThinking={defenderStatus === 'machine_thinking' || isSolving || (mode === 'spectator' && isStepLoading)}
            pliesSurvived={mode === 'defender' ? pliesSurvived : null}
            banner={banner ? {
              ...banner,
              action: (
                <button
                  onClick={resetBoard}
                  className="flex items-center gap-1.5 bg-neutral-800 hover:bg-neutral-700 text-slate-200 px-4 py-2 rounded-xl text-xs font-semibold shadow border border-neutral-600 transition-colors mx-auto"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Try Again
                </button>
              )
            } : null}
          />
        </div>
        
        {/* Right Column: Control & Telemetry Panels */}
        <div className="col-span-12 lg:col-span-5 xl:col-span-5 flex flex-col gap-4 w-full">
          <PuzzleSelector 
            puzzles={PRESET_PUZZLES} 
            onSelect={loadPuzzle} 
            customFen={customFen}
            onCustomFenChange={setCustomFen}
            onLoadCustom={handleLoadCustom}
            selectedFen={startingFen}
          />
          
          {mode === 'spectator' ? (
            <SpectatorPanel 
              spectatorStatus={spectatorStatus}
              activeTurnRole={activeSpectatorRole}
              modelElo={spectatorElo}
              rollingAcpl={spectatorAcpl}
              stockfishLevel={stockfishLevel}
              stockfishElo={stockfishElo}
              plies={spectatorPlies}
              lastModelCpl={lastModelCpl}
              temperature={openingTemperature}
              setTemperature={setOpeningTemperature}
              useBook={useBook}
              setUseBook={setUseBook}
              useSearch={useMinimaxSearch}
              setUseSearch={setUseMinimaxSearch}
              onPlay={() => setIsAutoPlaying(true)}
              onPause={() => setIsAutoPlaying(false)}
              onStep={stepSpectator}
              onReset={() => initSpectatorSession(startingFen)}
              matchHistory={spectatorMatchHistory}
              isAutoPlaying={isAutoPlaying}
              setIsAutoPlaying={setIsAutoPlaying}
              isStepLoading={isStepLoading}
              moveExplanation={moveExplanation}
              modelColor={modelColor}
              onSwapSides={swapSides}
              turn={game.turn()}
              telemetryHistory={telemetryHistory}
              latestReview={latestReview}
              isEvaluatingFullGame={isEvaluatingFullGame}
            />
          ) : mode === 'defender' ? (
            <DefenderPanel 
              defenderColor={defenderColor}
              defenderStatus={defenderStatus}
              pliesSurvived={pliesSurvived}
              targetPlies={targetPlies}
              matchHistory={matchHistory}
              onReset={resetBoard}
              onSwitchToAuto={() => switchMode('auto')}
              latestMachineMove={latestMachineMove}
            />
          ) : (
            <SolveControls 
              onSolve={solveMate}
              onPlay={playSequence}
              onPause={pausePlayback}
              onStep={stepForward}
              onReset={resetBoard}
              isSolving={isSolving}
              isPlaying={isPlaying}
              hasSolution={!!solution}
              currentStep={currentStep}
              totalSteps={solution?.sequence?.length || 0}
            />
          )}
          
          <TelemetryPanel 
            solution={solution}
            currentStep={currentStep}
            inferenceTime={inferenceTime}
            backendStatus={backendStatus}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

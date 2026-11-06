import React, { useState, useRef, useEffect } from 'react';
import { 
  User, 
  BrainCircuit, 
  RotateCcw, 
  Flag, 
  Handshake, 
  Award, 
  Sliders, 
  Sparkles, 
  ShieldAlert, 
  CheckCircle2, 
  Loader2,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

const UserVsEnginePanel = ({
  userColor = 'w',
  onUserColorChange,
  history = [],
  status = 'user_turn',
  onNewGame,
  onResign,
  onOfferDraw,
  onReviewGame,
  isEvaluating = false,
  latestEngineMove = null,
  useBook = true,
  setUseBook,
  useMinimaxSearch = true,
  setUseMinimaxSearch,
  openingTemperature = 0.0,
  setOpeningTemperature,
  hasReviewData = false,
  onOpenReviewModal
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const moveListEndRef = useRef(null);

  // Group moves into pairs for standard chess notation
  const movePairs = [];
  for (let i = 0; i < history.length; i += 2) {
    movePairs.push({
      number: Math.floor(i / 2) + 1,
      white: history[i]?.san || '',
      black: history[i + 1]?.san || ''
    });
  }

  useEffect(() => {
    moveListEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history.length]);

  return (
    <div className="bg-neutral-900/90 border border-neutral-800 rounded-2xl p-5 shadow-2xl space-y-4 font-mono text-xs w-full">
      {/* 1. Player Matchup Card */}
      <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 flex items-center justify-between gap-3 shadow-inner">
        {/* User Card */}
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-300">
            <User className="w-5 h-5" />
          </div>
          <div>
            <div className="font-bold text-white text-sm font-sans flex items-center gap-1.5">
              You
              <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-bold ${
                userColor === 'w' ? 'bg-slate-200 text-slate-900' : 'bg-neutral-800 text-slate-300 border border-neutral-700'
              }`}>
                {userColor === 'w' ? 'White' : 'Black'}
              </span>
            </div>
            <span className="text-[11px] text-cyan-400/90">Human Player</span>
          </div>
        </div>

        {/* VS Badge */}
        <div className="px-2.5 py-1 rounded-full bg-neutral-900 border border-neutral-700 text-slate-400 font-black text-[11px] uppercase tracking-wider">
          VS
        </div>

        {/* Engine Card */}
        <div className="flex items-center gap-2.5 text-right">
          <div>
            <div className="font-bold text-white text-sm font-sans flex items-center justify-end gap-1.5">
              <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-bold ${
                userColor === 'w' ? 'bg-neutral-800 text-slate-300 border border-neutral-700' : 'bg-slate-200 text-slate-900'
              }`}>
                {userColor === 'w' ? 'Black' : 'White'}
              </span>
              SE-ResNet-8
            </div>
            <span className="text-[11px] text-emerald-400/90 font-mono">Neural Engine</span>
          </div>
          <div className="w-9 h-9 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-300">
            <BrainCircuit className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 2. Color Selection (only when game hasn't started or on reset) */}
      {history.length === 0 && (
        <div className="flex items-center justify-between p-3 bg-neutral-950/70 border border-neutral-800 rounded-xl">
          <span className="text-slate-400 font-sans text-xs">Choose Your Side:</span>
          <div className="flex gap-2">
            <button
              onClick={() => onUserColorChange('w')}
              className={`px-3 py-1.5 rounded-lg font-bold text-xs transition flex items-center gap-1.5 ${
                userColor === 'w'
                  ? 'bg-slate-100 text-slate-950 shadow-md font-black'
                  : 'bg-neutral-900 text-slate-400 hover:text-white border border-neutral-800'
              }`}
            >
              <div className="w-3 h-3 rounded-full bg-white border border-slate-300" />
              White (1st Move)
            </button>
            <button
              onClick={() => onUserColorChange('b')}
              className={`px-3 py-1.5 rounded-lg font-bold text-xs transition flex items-center gap-1.5 ${
                userColor === 'b'
                  ? 'bg-neutral-800 text-white shadow-md border border-neutral-600 font-black'
                  : 'bg-neutral-900 text-slate-400 hover:text-white border border-neutral-800'
              }`}
            >
              <div className="w-3 h-3 rounded-full bg-neutral-950 border border-neutral-600" />
              Black (2nd Move)
            </button>
          </div>
        </div>
      )}

      {/* 3. Action Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <button
          onClick={onNewGame}
          className="flex items-center justify-center gap-1.5 p-2.5 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-slate-200 transition font-bold text-xs border border-neutral-700"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>New Game</span>
        </button>

        <button
          onClick={onResign}
          disabled={status === 'game_over' || history.length === 0}
          className="flex items-center justify-center gap-1.5 p-2.5 rounded-xl bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 disabled:opacity-40 disabled:cursor-not-allowed transition font-bold text-xs border border-rose-800/50"
        >
          <Flag className="w-3.5 h-3.5 text-rose-400" />
          <span>Resign</span>
        </button>

        <button
          onClick={onOfferDraw}
          disabled={status === 'game_over' || history.length < 6}
          className="flex items-center justify-center gap-1.5 p-2.5 rounded-xl bg-amber-950/40 hover:bg-amber-900/50 text-amber-300 disabled:opacity-40 disabled:cursor-not-allowed transition font-bold text-xs border border-amber-800/50"
        >
          <Handshake className="w-3.5 h-3.5 text-amber-400" />
          <span>Claim Draw</span>
        </button>

        <button
          onClick={hasReviewData ? onOpenReviewModal : onReviewGame}
          disabled={isEvaluating || history.length === 0}
          className={`flex items-center justify-center gap-1.5 p-2.5 rounded-xl transition font-bold text-xs border shadow-lg ${
            hasReviewData
              ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white border-emerald-400/40 shadow-emerald-950/50'
              : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white border-indigo-400/40 shadow-indigo-950/50 hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed'
          }`}
        >
          {isEvaluating ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : hasReviewData ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-200" />
              <span>View Review</span>
            </>
          ) : (
            <>
              <Award className="w-3.5 h-3.5 text-amber-300" />
              <span>Review Game</span>
            </>
          )}
        </button>
      </div>

      {/* 4. Live Stockfish Evaluation Banner (when in progress) */}
      {isEvaluating && (
        <div className="flex items-center gap-2.5 p-3 rounded-xl bg-indigo-950/60 border border-indigo-500/40 text-indigo-200 animate-pulse">
          <Loader2 className="w-4 h-4 animate-spin text-indigo-400 flex-shrink-0" />
          <div>
            <span className="font-bold text-xs block">Stockfish Depth 12 Evaluation in Progress</span>
            <span className="text-[11px] text-indigo-300/80">Calculating ACPL, blunder tracking, and performance Elos for both sides...</span>
          </div>
        </div>
      )}

      {/* 5. Engine Telemetry & Tactical Insight */}
      {latestEngineMove && (
        <div className="p-3.5 bg-neutral-950 border border-neutral-800 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-sans font-semibold text-xs flex items-center gap-1.5 text-white">
              <BrainCircuit className="w-3.5 h-3.5 text-emerald-400" />
              SE-ResNet-8 Insight
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              {latestEngineMove.inference_ms ? `${latestEngineMove.inference_ms.toFixed(0)} ms` : ''}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="bg-neutral-900 p-2 rounded-lg border border-neutral-800">
              <span className="text-[10px] text-slate-500 uppercase block">Last Played</span>
              <span className="font-bold text-sm text-emerald-400 font-sans">
                {latestEngineMove.san}
                {latestEngineMove.is_book && (
                  <span className="ml-1.5 text-[9px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    Book
                  </span>
                )}
              </span>
            </div>

            <div className="bg-neutral-900 p-2 rounded-lg border border-neutral-800">
              <span className="text-[10px] text-slate-500 uppercase block">Confidence</span>
              <span className="font-bold text-sm text-cyan-400 font-mono">
                {latestEngineMove.confidence ? `${(latestEngineMove.confidence * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
          </div>

          {latestEngineMove.tactical_rationale && (
            <div className="text-[11px] text-slate-300 bg-neutral-900/80 p-2 rounded-lg border border-neutral-800/80 leading-relaxed">
              {latestEngineMove.tactical_rationale}
            </div>
          )}
        </div>
      )}

      {/* 6. Move History Table */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 font-sans text-xs font-semibold">Move Notation</span>
          <span className="text-slate-500 text-[10px] font-mono">{history.length} plies played</span>
        </div>

        <div className="h-44 overflow-y-auto bg-neutral-950 p-2 rounded-xl border border-neutral-800/80 font-mono text-[11px] select-text">
          {movePairs.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-600 italic">
              Make a move on the board to start the match
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="text-slate-600 border-b border-neutral-900 text-[10px]">
                  <th className="py-1 px-2 w-10">#</th>
                  <th className="py-1 px-2">White</th>
                  <th className="py-1 px-2">Black</th>
                </tr>
              </thead>
              <tbody>
                {movePairs.map((pair) => (
                  <tr key={pair.number} className="hover:bg-neutral-900/60 transition">
                    <td className="py-1 px-2 text-slate-600 font-bold">{pair.number}.</td>
                    <td className="py-1 px-2 text-slate-200 font-semibold">{pair.white}</td>
                    <td className="py-1 px-2 text-slate-400 font-semibold">{pair.black}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div ref={moveListEndRef} />
        </div>
      </div>

      {/* 7. Collapsible Engine Configuration */}
      <div className="border-t border-neutral-800/80 pt-3">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center justify-between w-full text-slate-400 hover:text-slate-200 transition font-sans text-xs font-medium"
        >
          <span className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-slate-500" />
            Engine Settings & Search
          </span>
          {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showSettings && (
          <div className="mt-3 p-3 bg-neutral-950 rounded-xl border border-neutral-800/80 space-y-3">
            {/* Opening Book Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-300 font-sans text-xs font-semibold block">Opening Book</span>
                <span className="text-[10px] text-slate-500">Use Polyglot Grandmaster Book</span>
              </div>
              <input
                type="checkbox"
                checked={useBook}
                onChange={(e) => setUseBook(e.target.checked)}
                className="w-4 h-4 rounded bg-neutral-800 border-neutral-700 text-emerald-500 focus:ring-0 cursor-pointer"
              />
            </div>

            {/* Minimax Search Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-300 font-sans text-xs font-semibold block">Minimax Alpha-Beta</span>
                <span className="text-[10px] text-slate-500">Defensive awareness & trade evaluation</span>
              </div>
              <input
                type="checkbox"
                checked={useMinimaxSearch}
                onChange={(e) => setUseMinimaxSearch(e.target.checked)}
                className="w-4 h-4 rounded bg-neutral-800 border-neutral-700 text-emerald-500 focus:ring-0 cursor-pointer"
              />
            </div>

            {/* Temperature Slider */}
            <div>
              <div className="flex justify-between text-slate-400 mb-1">
                <span className="font-sans text-xs">Opening Temperature</span>
                <span className="font-mono">{openingTemperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={openingTemperature}
                onChange={(e) => setOpeningTemperature(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                <span>Deterministic (0.0)</span>
                <span>Varied (1.0)</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserVsEnginePanel;

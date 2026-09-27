import React, { useState } from 'react';
import { 
  X, 
  Trophy, 
  Award, 
  AlertTriangle, 
  Zap, 
  BrainCircuit, 
  User, 
  CheckCircle2, 
  TrendingUp, 
  ShieldAlert,
  RotateCcw
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from 'recharts';

const GameReviewModal = ({
  isOpen = false,
  onClose,
  reviewData = null,
  onNewGame
}) => {
  const [filter, setFilter] = useState('all'); // 'all' | 'user' | 'engine'

  if (!isOpen || !reviewData) return null;

  const userStats = reviewData.user_stats || {};
  const engineStats = reviewData.engine_stats || {};
  const evalHistory = reviewData.eval_history || [];
  const missedTactics = reviewData.missed_tactics || [];

  // Filter tactics based on tab
  const filteredTactics = missedTactics.filter(t => {
    if (filter === 'user') return t.player === 'user';
    if (filter === 'engine') return t.player === 'engine';
    return true;
  });

  const isInProgress = Boolean(reviewData.is_in_progress || userStats.result === 'in_progress');
  const userWon = !isInProgress && userStats.result === 'win';
  const engineWon = !isInProgress && engineStats.result === 'win';
  const isDraw = !isInProgress && userStats.result === 'draw';

  const userTacticsCount = missedTactics.filter(t => t.player === 'user').length;
  const engineTacticsCount = missedTactics.filter(t => t.player === 'engine').length;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-3 sm:p-5 overflow-y-auto animate-fade-in">
      <div className="bg-[#0e131f] border border-neutral-800 w-full max-w-4xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh] font-mono text-xs">
        
        {/* Header */}
        <div className="p-4 sm:p-5 bg-neutral-950 border-b border-neutral-800 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center text-amber-400 shadow-md">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-white font-sans font-black text-base sm:text-lg tracking-tight">
                  Stockfish 17 Dual Match Review
                </h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-bold">
                  Depth 12 Grandmaster Analysis
                </span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Centipawn loss, move accuracy, and performance rating computed for both players
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-neutral-900 hover:bg-neutral-800 text-slate-400 hover:text-white transition border border-neutral-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Content Body */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-6">
          
          {/* Match Outcome Banner */}
          <div className={`p-4 rounded-xl border flex items-center justify-between gap-3 ${
            isInProgress
              ? 'bg-indigo-950/40 border-indigo-500/40 text-indigo-200'
              : userWon 
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200' 
              : engineWon 
              ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
              : 'bg-amber-950/40 border-amber-500/40 text-amber-200'
          }`}>
            <div className="flex items-center gap-3">
              <span className="text-3xl">
                {isInProgress ? '⚡' : userWon ? '👑' : engineWon ? '🤖' : '⚖️'}
              </span>
              <div>
                <h3 className="font-sans font-black text-sm sm:text-base text-white">
                  {isInProgress
                    ? 'Match In Progress: Live Position Review'
                    : userWon 
                    ? 'Victory! You defeated SE-ResNet-8!' 
                    : engineWon 
                    ? 'SE-ResNet-8 claimed victory in this match!' 
                    : 'Game Drawn by repetition, agreement, or stalemate!'}
                </h3>
                <p className="text-[11px] opacity-80 font-sans">
                  {evalHistory.length} total half-moves evaluated with Stockfish 17 (Depth 12)
                </p>
              </div>
            </div>

            <div className="text-right">
              <span className={`px-2.5 py-1 rounded font-bold uppercase text-[11px] ${
                isInProgress
                  ? 'bg-indigo-900/60 text-indigo-300 border border-indigo-700'
                  : userWon 
                  ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700' 
                  : engineWon 
                  ? 'bg-rose-900/60 text-rose-300 border border-rose-700'
                  : 'bg-amber-900/60 text-amber-300 border border-amber-700'
              }`}>
                {isInProgress ? 'In Progress' : userWon ? 'You Won' : engineWon ? 'Engine Won' : 'Draw'}
              </span>
            </div>
          </div>

          {/* Side-by-Side Head-to-Head Comparison Matrix */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* 👤 User Card */}
            <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 space-y-3 relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-neutral-800/80 pb-2.5">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-cyan-500/20 text-cyan-300 flex items-center justify-center">
                    <User className="w-3.5 h-3.5" />
                  </div>
                  <span className="font-sans font-bold text-white text-sm">You ({userStats.color === 'white' ? 'White' : 'Black'})</span>
                </div>
                <span className="text-cyan-400 font-bold text-xs">{userStats.accuracy}% Accuracy</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="bg-neutral-900/80 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Performance Elo</span>
                  <span className="text-2xl font-black text-cyan-400 font-mono block mt-0.5">
                    {userStats.game_elo || 1200}
                  </span>
                </div>
                <div className="bg-neutral-900/80 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Average CPL</span>
                  <span className="text-2xl font-black text-slate-200 font-mono block mt-0.5">
                    {userStats.acpl !== undefined ? `${userStats.acpl} cp` : '—'}
                  </span>
                </div>
              </div>

              {/* Move Breakdown */}
              <div className="grid grid-cols-4 gap-1.5 pt-1 text-center">
                <div className="bg-emerald-950/40 border border-emerald-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Best (&lt;15cp)</span>
                  <span className="font-bold text-emerald-400 text-xs">{userStats.best_moves || 0}</span>
                </div>
                <div className="bg-cyan-950/40 border border-cyan-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Inaccurate</span>
                  <span className="font-bold text-cyan-400 text-xs">{userStats.inaccuracies || 0}</span>
                </div>
                <div className="bg-amber-950/40 border border-amber-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Mistakes</span>
                  <span className="font-bold text-amber-400 text-xs">{userStats.mistakes || 0}</span>
                </div>
                <div className="bg-rose-950/40 border border-rose-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Blunders</span>
                  <span className="font-bold text-rose-400 text-xs">{userStats.blunders || 0}</span>
                </div>
              </div>
            </div>

            {/* 🤖 Engine Card */}
            <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 space-y-3 relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-neutral-800/80 pb-2.5">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-emerald-500/20 text-emerald-300 flex items-center justify-center">
                    <BrainCircuit className="w-3.5 h-3.5" />
                  </div>
                  <span className="font-sans font-bold text-white text-sm">SE-ResNet-8 ({engineStats.color === 'white' ? 'White' : 'Black'})</span>
                </div>
                <span className="text-emerald-400 font-bold text-xs">{engineStats.accuracy}% Accuracy</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="bg-neutral-900/80 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Performance Elo</span>
                  <span className="text-2xl font-black text-emerald-400 font-mono block mt-0.5">
                    {engineStats.game_elo || 1200}
                  </span>
                </div>
                <div className="bg-neutral-900/80 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Average CPL</span>
                  <span className="text-2xl font-black text-slate-200 font-mono block mt-0.5">
                    {engineStats.acpl !== undefined ? `${engineStats.acpl} cp` : '—'}
                  </span>
                </div>
              </div>

              {/* Move Breakdown */}
              <div className="grid grid-cols-4 gap-1.5 pt-1 text-center">
                <div className="bg-emerald-950/40 border border-emerald-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Best (&lt;15cp)</span>
                  <span className="font-bold text-emerald-400 text-xs">{engineStats.best_moves || 0}</span>
                </div>
                <div className="bg-cyan-950/40 border border-cyan-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Inaccurate</span>
                  <span className="font-bold text-cyan-400 text-xs">{engineStats.inaccuracies || 0}</span>
                </div>
                <div className="bg-amber-950/40 border border-amber-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Mistakes</span>
                  <span className="font-bold text-amber-400 text-xs">{engineStats.mistakes || 0}</span>
                </div>
                <div className="bg-rose-950/40 border border-rose-800/40 p-1.5 rounded">
                  <span className="text-[9px] text-slate-400 block">Blunders</span>
                  <span className="font-bold text-rose-400 text-xs">{engineStats.blunders || 0}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Centipawn Advantage Progression Chart */}
          {evalHistory.length > 0 && (
            <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-sans font-bold text-white text-xs flex items-center gap-1.5">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  Game Advantage & Momentum (Centipawns from White POV)
                </span>
                <span className="text-[10px] text-slate-500 font-mono">
                  +100 cp = 1 pawn advantage
                </span>
              </div>

              <div className="w-full h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={evalHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="evalGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                    <XAxis dataKey="ply" stroke="#525252" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#525252" tick={{ fontSize: 10 }} domain={[-600, 600]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#171717', borderColor: '#404040', borderRadius: '8px', fontSize: '11px' }}
                      formatter={(val, name, item) => [
                        `${typeof val === 'number' ? (val / 100).toFixed(2) : '0.00'} pawns (${item?.payload?.san || ''})`, 
                        item?.payload?.mover === 'white' ? 'White' : 'Black'
                      ]}
                    />
                    <ReferenceLine y={0} stroke="#737373" strokeDasharray="2 2" />
                    <Area type="monotone" dataKey="eval_cp" stroke="#06b6d4" fill="url(#evalGrad)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Blunders & Missed Opportunities Inspector */}
          <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-800 pb-2">
              <span className="font-sans font-bold text-white text-xs flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Missed Tactical Opportunities & Blunders ({missedTactics.length})
              </span>

              {/* Filter Tabs */}
              <div className="flex gap-1 bg-neutral-900 p-1 rounded-lg border border-neutral-800">
                <button
                  onClick={() => setFilter('all')}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-bold transition ${
                    filter === 'all' ? 'bg-neutral-800 text-white shadow' : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  All ({missedTactics.length})
                </button>
                <button
                  onClick={() => setFilter('user')}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-bold transition ${
                    filter === 'user' ? 'bg-cyan-950 text-cyan-300 border border-cyan-700' : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  Your Blunders ({userTacticsCount})
                </button>
                <button
                  onClick={() => setFilter('engine')}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-bold transition ${
                    filter === 'engine' ? 'bg-emerald-950 text-emerald-300 border border-emerald-700' : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  Engine Blunders ({engineTacticsCount})
                </button>
              </div>
            </div>

            {filteredTactics.length === 0 ? (
              <div className="p-6 text-center text-slate-500 italic bg-neutral-900/40 rounded-lg">
                No tactical blunders found in this category! Clean and disciplined play.
              </div>
            ) : (
              <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {filteredTactics.map((tactic, idx) => (
                  <div 
                    key={idx}
                    className="p-3 bg-neutral-900/90 rounded-lg border border-neutral-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-neutral-700 transition"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${
                          tactic.player === 'user'
                            ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                            : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                        }`}>
                          {tactic.player === 'user' ? '👤 You' : '🤖 SE-ResNet'}
                        </span>
                        <span className="font-bold text-white text-xs">
                          Ply {tactic.ply}: Played {tactic.move_san}
                        </span>
                        <span className="text-[10px] text-rose-400 font-bold">
                          (-{(tactic.eval_drop_cp / 100).toFixed(1)} pawns)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 font-sans">
                        {tactic.description}
                      </p>
                    </div>

                    <div className="flex items-center gap-2 self-start sm:self-center">
                      <div className="text-right">
                        <span className="text-[9px] text-slate-500 uppercase block">Stockfish Best</span>
                        <span className="text-sm font-black text-emerald-400 font-sans">
                          {tactic.best_move_san}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 bg-neutral-950 border-t border-neutral-800 flex items-center justify-between gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-neutral-900 hover:bg-neutral-800 text-slate-300 hover:text-white font-bold text-xs transition border border-neutral-800"
          >
            Close Analysis
          </button>

          <button
            onClick={() => {
              onClose();
              if (onNewGame) onNewGame();
            }}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-bold text-xs shadow-lg shadow-emerald-950/50 hover:brightness-110 transition border border-emerald-400/30"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Play Rematch</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default GameReviewModal;

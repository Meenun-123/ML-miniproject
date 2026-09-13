import React, { useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

const ModelTelemetryDashboard = ({
  history = [],
  currentElo = 1200,
  currentAcpl = 50.0,
  latestReview = null,
  isEvaluatingFullGame = false,
}) => {
  const [activeMetric, setActiveMetric] = useState('elo'); // 'elo' | 'acpl' | 'accuracy' | 'blunders'

  // Normalize data for Recharts so both legacy and new schema records graph smoothly
  const formattedData = history.map(g => {
    const rawAcpl = typeof g.acpl === 'number' ? g.acpl : 50.0;
    const computedAccuracy = g.accuracy ?? Math.round(Math.max(0, Math.min(100, 100 * Math.exp(-0.004 * rawAcpl))));
    return {
      ...g,
      game_id: g.game_id,
      game_elo: g.game_elo ?? g.final_elo ?? 1200,
      cumulative_elo: g.cumulative_elo ?? g.final_elo ?? 1200,
      acpl: rawAcpl,
      accuracy: computedAccuracy,
      blunders: g.blunders ?? 0,
      mistakes: g.mistakes ?? 0,
      inaccuracies: g.inaccuracies ?? 0,
      best_moves: g.best_moves ?? 0,
    };
  });

  const latestGame = latestReview || (formattedData.length > 0 ? formattedData[formattedData.length - 1] : null);

  const wins = history.filter(g => g.result === 'win').length;
  const draws = history.filter(g => g.result === 'draw').length;
  const winRate = history.length ? Math.round(((wins + draws * 0.5) / history.length) * 100) : 0;

  // Determine delta from previous game
  let deltaElo = null;
  if (latestGame && latestGame.delta_elo !== undefined) {
    deltaElo = latestGame.delta_elo;
  } else if (formattedData.length >= 2) {
    const curr = formattedData[formattedData.length - 1].game_elo;
    const prev = formattedData[formattedData.length - 2].game_elo;
    deltaElo = curr - prev;
  }

  return (
    <div className="bg-neutral-900/90 border border-neutral-800 rounded-2xl p-5 shadow-2xl space-y-4 font-mono text-xs">
      {/* Top Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full">
        <div className="bg-neutral-950 p-3 rounded-xl border border-neutral-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Cumulative Elo</span>
          <span className="text-xl font-black text-emerald-400">
            {latestGame?.cumulative_elo ?? currentElo}
          </span>
        </div>
        <div className="bg-neutral-950 p-3 rounded-xl border border-neutral-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Game Accuracy</span>
          <span className="text-xl font-black text-cyan-400">
            {latestGame?.accuracy ? `${latestGame.accuracy}%` : `${currentAcpl} cp`}
          </span>
        </div>
        <div className="bg-neutral-950 p-3 rounded-xl border border-neutral-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Win / Draw Rate</span>
          <span className="text-xl font-black text-amber-400">{winRate}%</span>
        </div>
      </div>

      {/* Evaluating Badge */}
      {isEvaluatingFullGame && (
        <div className="flex items-center justify-center gap-2.5 p-3 bg-cyan-950/80 border border-cyan-500/50 rounded-xl text-cyan-300 font-mono text-xs animate-pulse shadow-lg shadow-cyan-950/40">
          <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          <span className="font-semibold tracking-wide">Stockfish is evaluating full-game consistency (Depth 12)...</span>
        </div>
      )}

      {/* Post-Match Stockfish Review Card */}
      {latestGame && !isEvaluatingFullGame && (
        <div className="bg-gradient-to-br from-neutral-950 via-neutral-900/90 to-neutral-950 border border-neutral-800 rounded-xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-neutral-800/80 pb-2">
            <div className="flex items-center gap-2">
              <span className="text-base">♟️</span>
              <span className="font-bold text-white font-sans text-xs">
                Post-Match Stockfish Review (Game #{latestGame.game_id})
              </span>
            </div>
            <span
              className={`px-2.5 py-0.5 rounded font-bold uppercase text-[10px] ${
                latestGame.result === 'win'
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                  : latestGame.result === 'draw'
                  ? 'bg-amber-950 text-amber-400 border border-amber-800'
                  : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}
            >
              {latestGame.result}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-neutral-900/90 p-2.5 rounded-lg border border-neutral-800/80">
              <span className="text-[10px] text-slate-500 uppercase block">Game Elo</span>
              <div className="flex items-center justify-center gap-1.5 mt-0.5">
                <span className="text-lg font-black text-emerald-400">{latestGame.game_elo}</span>
                {deltaElo !== null && (
                  <span
                    className={`text-[10px] font-bold ${
                      deltaElo >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {deltaElo >= 0 ? `▲+${deltaElo}` : `▼${deltaElo}`}
                  </span>
                )}
              </div>
            </div>

            <div className="bg-neutral-900/90 p-2.5 rounded-lg border border-neutral-800/80">
              <span className="text-[10px] text-slate-500 uppercase block">Move Accuracy</span>
              <span className="text-lg font-black text-cyan-400 block mt-0.5">{latestGame.accuracy}%</span>
            </div>

            <div className="bg-neutral-900/90 p-2.5 rounded-lg border border-neutral-800/80">
              <span className="text-[10px] text-slate-500 uppercase block">Cumulative Elo</span>
              <span className="text-lg font-black text-indigo-400 block mt-0.5">{latestGame.cumulative_elo}</span>
            </div>
          </div>

          {/* Move Precision Breakdown */}
          <div className="grid grid-cols-4 gap-1.5 pt-1 text-[10px]">
            <div className="bg-emerald-950/40 border border-emerald-800/40 p-1.5 rounded text-center">
              <span className="text-slate-400 block text-[9px]">Best (&lt;15cp)</span>
              <span className="font-bold text-emerald-400">{latestGame.best_moves ?? 0}</span>
            </div>
            <div className="bg-cyan-950/40 border border-cyan-800/40 p-1.5 rounded text-center">
              <span className="text-slate-400 block text-[9px]">Inaccurate</span>
              <span className="font-bold text-cyan-400">{latestGame.inaccuracies ?? 0}</span>
            </div>
            <div className="bg-amber-950/40 border border-amber-800/40 p-1.5 rounded text-center">
              <span className="text-slate-400 block text-[9px]">Mistakes</span>
              <span className="font-bold text-amber-400">{latestGame.mistakes ?? 0}</span>
            </div>
            <div className="bg-rose-950/40 border border-rose-800/40 p-1.5 rounded text-center">
              <span className="text-slate-400 block text-[9px]">Blunders</span>
              <span className="font-bold text-rose-400">{latestGame.blunders ?? 0}</span>
            </div>
          </div>
        </div>
      )}

      {/* Metric Selector Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <span className="text-slate-400 font-sans text-xs font-semibold">Performance Trajectory</span>
        <div className="flex flex-wrap gap-1 bg-neutral-950 p-1 rounded-lg border border-neutral-800">
          {[
            { id: 'elo', label: 'Elo Curve' },
            { id: 'acpl', label: 'ACPL' },
            { id: 'accuracy', label: 'Accuracy' },
            { id: 'blunders', label: 'Blunders' },
          ].map(m => (
            <button
              key={m.id}
              onClick={() => setActiveMetric(m.id)}
              className={`px-2.5 py-1 rounded text-[11px] font-bold uppercase transition ${
                activeMetric === m.id
                  ? 'bg-neutral-800 text-white shadow-sm'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Recharts Area / Line Curve */}
      <div className="w-full min-h-[180px] h-48 md:h-56 bg-neutral-950/60 p-2 rounded-xl border border-neutral-800/80">
        <ResponsiveContainer width="100%" height="100%" debounce={100}>
          <AreaChart data={formattedData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="gameEloGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="cumEloGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#818cf8" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#818cf8" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="acplGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#a855f7" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="blunderGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="game_id" stroke="#525252" tickLine={false} tickFormatter={v => `G${v}`} />
            <YAxis stroke="#525252" tickLine={false} domain={['dataMin - 20', 'dataMax + 20']} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#262626', borderRadius: '0.75rem' }}
              labelFormatter={label => `Game #${label}`}
            />

            {activeMetric === 'elo' && (
              <>
                <Area
                  name="Game Elo"
                  type="monotone"
                  dataKey="game_elo"
                  stroke="#10b981"
                  strokeWidth={2}
                  fill="url(#gameEloGrad)"
                />
                <Area
                  name="Cumulative Elo"
                  type="monotone"
                  dataKey="cumulative_elo"
                  stroke="#818cf8"
                  strokeWidth={2.5}
                  strokeDasharray="4 4"
                  fill="url(#cumEloGrad)"
                />
              </>
            )}

            {activeMetric === 'acpl' && (
              <Area
                name="ACPL (cp)"
                type="monotone"
                dataKey="acpl"
                stroke="#06b6d4"
                strokeWidth={2.5}
                fill="url(#acplGrad)"
              />
            )}

            {activeMetric === 'accuracy' && (
              <Area
                name="Accuracy (%)"
                type="monotone"
                dataKey="accuracy"
                stroke="#a855f7"
                strokeWidth={2.5}
                fill="url(#accGrad)"
              />
            )}

            {activeMetric === 'blunders' && (
              <Area
                name="Blunders"
                type="monotone"
                dataKey="blunders"
                stroke="#f59e0b"
                strokeWidth={2.5}
                fill="url(#blunderGrad)"
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Match Log Summary */}
      <div className="overflow-x-auto rounded-lg">
        <div className="min-w-[280px] space-y-1.5 max-h-36 overflow-y-auto pr-1">
          {history.slice(-4).reverse().map(game => (
            <div key={game.game_id} className="flex items-center justify-between p-2 rounded-lg bg-neutral-950/40 border border-neutral-800/40 text-[11px]">
              <span className="text-slate-400">
                Game #{game.game_id} ({game.moves_count} plies)
                {game.accuracy ? ` • ${game.accuracy}% acc` : ''}
              </span>
              <div className="flex items-center gap-3">
                <span className="text-cyan-400">{game.acpl} cp</span>
                <span className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                  game.result === 'win' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' :
                  game.result === 'draw' ? 'bg-amber-950 text-amber-400 border border-amber-800' :
                  'bg-rose-950 text-rose-400 border border-rose-800'
                }`}>
                  {game.result}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelTelemetryDashboard;

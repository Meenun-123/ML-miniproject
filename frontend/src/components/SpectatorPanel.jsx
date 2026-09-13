import React from 'react';
import { 
  Swords, 
  BrainCircuit, 
  Cpu, 
  Play, 
  Pause, 
  SkipForward, 
  RotateCcw, 
  Activity, 
  Gauge, 
  Flame, 
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  BookOpen,
  HelpCircle,
  ArrowLeftRight
} from 'lucide-react';
import ModelTelemetryDashboard from './ModelTelemetryDashboard';

const SpectatorPanel = ({
  spectatorStatus = 'idle', // 'idle' | 'playing' | 'paused' | 'game_over'
  activeTurnRole = 'model', // 'model' | 'stockfish'
  modelElo = 2400,
  rollingAcpl = 0.0,
  stockfishLevel = 16,
  stockfishElo = 2430,
  plies = 0,
  lastModelCpl = null,
  temperature = 1.0,
  setTemperature = () => {},
  useBook = true,
  setUseBook = () => {},
  useSearch = true,
  setUseSearch = () => {},
  onPlay = () => {},
  onPause = () => {},
  onStep = () => {},
  onReset = () => {},
  matchHistory = [],
  isAutoPlaying = false,
  setIsAutoPlaying = () => {},
  isStepLoading = false,
  moveExplanation = null,
  modelColor = 'w',
  onSwapSides = () => {},
  turn = 'w',
  telemetryHistory = [],
  latestReview = null,
  isEvaluatingFullGame = false,
}) => {
  const isPlaying = spectatorStatus === 'playing';
  const isGameOver = spectatorStatus === 'game_over';

  // Quality badge for last CPL
  const getCplBadge = (cpl) => {
    if (cpl === null || cpl === undefined) return null;
    if (cpl <= 15) {
      return <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Best Move (0 cp)</span>;
    }
    if (cpl <= 50) {
      return <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">Good (+{cpl} cp)</span>;
    }
    if (cpl <= 100) {
      return <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">Inaccuracy (+{cpl} cp)</span>;
    }
    return <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Blunder (+{cpl} cp)</span>;
  };

  // Evaluation & Tactical Mode Badge
  const getEvalModeBadge = (eval_mode, detail) => {
    if (detail) {
      if (detail.includes("King Hunt") || detail.includes("👑")) {
        return (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-rose-500/25 text-rose-200 border-rose-500/50 shadow-sm flex items-center gap-1">
            👑 King Hunt
          </span>
        );
      }
      if (detail.includes("Exchange Sacrifice") || detail.includes("🗡️")) {
        return (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-purple-500/25 text-purple-200 border-purple-500/50 shadow-sm flex items-center gap-1">
            🗡️ Exchange Sacrifice
          </span>
        );
      }
      if (detail.includes("Tactical Sacrifice") || detail.includes("Positional Sacrifice") || detail.includes("✨")) {
        return (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-500/25 text-emerald-200 border-emerald-500/50 shadow-sm flex items-center gap-1">
            ✨ Sound Sacrifice
          </span>
        );
      }
      if (detail.includes("Tactical Retreat")) {
        return (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/25 text-amber-200 border-amber-500/50 shadow-sm flex items-center gap-1">
            🛡️ Tactical Retreat
          </span>
        );
      }
      if (detail.includes("Tactical Reinforcement")) {
        return (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/25 text-amber-200 border-amber-500/50 shadow-sm flex items-center gap-1">
            🛡️ Tactical Reinforcement
          </span>
        );
      }
    }

    if (eval_mode === 'tenacious_defense') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40">
          🛡️ Tenacious Defense
        </span>
      );
    }
    if (eval_mode === 'strategic_development') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-blue-500/20 text-blue-300 border-blue-500/40">
          🏛️ Strategic Development
        </span>
      );
    }
    if (eval_mode === 'clean_conversion') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-500/40">
          ⚔️ Clean Conversion
        </span>
      );
    }
    if (eval_mode === 'tactical_strike') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40">
          ⚡ Tactical Strike
        </span>
      );
    }
    return (
      <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-indigo-500/20 text-indigo-300 border-indigo-500/40">
        🧘 Prophylaxis
      </span>
    );
  };

  // Mini badge for match action log
  const getLogMiniBadge = (entry) => {
    if (!entry) return null;
    const rationale = entry.tactical_rationale || '';
    if (rationale.includes("King Hunt") || rationale.includes("👑")) {
      return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-rose-500/25 text-rose-200 border border-rose-500/40">👑 Hunt</span>;
    }
    if (rationale.includes("Exchange Sacrifice") || rationale.includes("🗡️")) {
      return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-500/25 text-purple-200 border border-purple-500/40">🗡️ Exch</span>;
    }
    if (rationale.includes("Tactical Sacrifice") || rationale.includes("✨")) {
      return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/25 text-emerald-200 border border-emerald-500/40">✨ Sac</span>;
    }
    if (rationale.includes("Tactical Retreat")) {
      return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/25 text-amber-200 border border-amber-500/40">🛡️ Retreat</span>;
    }
    return null;
  };

  return (
    <div className="bg-neutral-900/70 backdrop-blur-xl border border-neutral-800/80 rounded-2xl shadow-xl p-4 sm:p-5 flex flex-col gap-4 transition-all">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800/80">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Swords className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              Spectator Battle Arena
            </h2>
            <p className="text-xs text-slate-400">Autonomous Sparring & Difficulty Calibration</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 bg-neutral-950/80 rounded-full border border-neutral-800">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-300 font-mono">Ply {plies}</span>
        </div>
      </div>

      {/* Live Competitors Card */}
      <div className="flex flex-wrap items-stretch justify-between gap-3 w-full">
        {/* Model Side */}
        <div className={`min-w-[140px] flex-1 p-3.5 rounded-xl border transition-all ${
          activeTurnRole === 'model' && (isPlaying || isAutoPlaying || isStepLoading)
            ? 'bg-emerald-950/20 border-emerald-500/40 ring-1 ring-emerald-500/30'
            : 'bg-neutral-950/60 border-neutral-800/80'
        }`}>
          <div className="flex flex-wrap items-center justify-between gap-1 mb-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-400">
              <BrainCircuit className="w-4 h-4 shrink-0" />
              <span className="truncate">SE-ResNet-8</span>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-slate-300 font-medium">
              {modelColor === 'w' ? 'White' : 'Black'}
            </span>
          </div>

          {activeTurnRole === 'model' && (isPlaying || isAutoPlaying || isStepLoading) && (
            <div className="mb-2 flex items-center gap-1 text-[10px] text-emerald-300 animate-pulse font-mono">
              <Loader2 className="w-3 h-3 animate-spin shrink-0" /> Thinking
            </div>
          )}

          <div className="flex items-baseline justify-between mb-1.5">
            <span className="text-xs text-slate-400">Est. Elo</span>
            <span className="text-lg font-black text-slate-100 font-mono tracking-tight">{modelElo}</span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500">Rolling ACPL</span>
            <span className={`font-mono font-semibold ${
              rollingAcpl < 30 ? 'text-emerald-400' : rollingAcpl < 60 ? 'text-amber-400' : 'text-rose-400'
            }`}>
              {rollingAcpl.toFixed(1)} cp
            </span>
          </div>
        </div>

        {/* Stockfish Side */}
        <div className={`min-w-[140px] flex-1 p-3.5 rounded-xl border transition-all ${
          activeTurnRole === 'stockfish' && (isPlaying || isAutoPlaying || isStepLoading)
            ? 'bg-cyan-950/20 border-cyan-500/40 ring-1 ring-cyan-500/30'
            : 'bg-neutral-950/60 border-neutral-800/80'
        }`}>
          <div className="flex flex-wrap items-center justify-between gap-1 mb-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-cyan-400">
              <Cpu className="w-4 h-4 shrink-0" />
              <span className="truncate">Stockfish 17</span>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-slate-300 font-medium">
              {modelColor === 'w' ? 'Black' : 'White'}
            </span>
          </div>

          {activeTurnRole === 'stockfish' && (isPlaying || isAutoPlaying || isStepLoading) && (
            <div className="mb-2 flex items-center gap-1 text-[10px] text-cyan-300 animate-pulse font-mono">
              <Loader2 className="w-3 h-3 animate-spin shrink-0" /> Replying
            </div>
          )}

          <div className="flex items-baseline justify-between mb-1.5">
            <span className="text-xs text-slate-400">Target Elo</span>
            <span className="text-lg font-black text-slate-100 font-mono tracking-tight">{stockfishElo}</span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500">Skill Level</span>
            <span className="font-mono font-semibold text-cyan-400">
              Lvl {stockfishLevel} <span className="text-slate-600">/ 20</span>
            </span>
          </div>
        </div>
      </div>

      {/* "Why this move?" Real-time Commentary Card */}
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center justify-between gap-2 px-1">
          <span className="text-xs text-slate-400 font-semibold flex items-center gap-1.5">
            <HelpCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>Tactical & Theoretical Commentary</span>
          </span>
          {moveExplanation && (
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border font-mono ${
              moveExplanation.role === 'model'
                ? moveExplanation.is_book
                  ? 'bg-purple-500/20 text-purple-300 border-purple-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
            }`}>
              {moveExplanation.source}
            </span>
          )}
        </div>

        {moveExplanation ? (
          <div className="min-h-[100px] p-4 space-y-3 bg-neutral-950/80 border border-neutral-800/90 rounded-xl relative overflow-hidden transition-all shadow-inner">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-800/60 pb-2">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-black text-white font-mono bg-neutral-900 px-2 py-0.5 rounded border border-neutral-700">
                  {moveExplanation.san}
                </span>
                <span className="text-xs text-slate-300 font-semibold">
                  {moveExplanation.player}
                </span>
                {moveExplanation.eval_mode && (
                  getEvalModeBadge(moveExplanation.eval_mode, moveExplanation.detail)
                )}
              </div>
              <div className="flex flex-wrap gap-2 items-center">
                {moveExplanation.metric && (
                  <span className="text-[11px] font-mono text-slate-300 bg-neutral-900/90 px-2 py-0.5 rounded border border-neutral-800">
                    {moveExplanation.metric}
                  </span>
                )}
              </div>
            </div>
            <p className="text-sm leading-relaxed text-slate-300 break-words font-sans">
              {moveExplanation.detail}
            </p>

            {(moveExplanation.anticipated_counter || (moveExplanation.search_pv && moveExplanation.search_pv.length > 1)) && (
              <div className="flex flex-wrap items-center gap-2 pt-1.5 border-t border-neutral-800/60 text-xs">
                {moveExplanation.anticipated_counter && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-950/30 border border-amber-700/40 text-amber-200">
                    <span className="text-[11px] font-medium text-amber-400">⚡ Anticipates Counter:</span>
                    <span className="font-mono font-bold">{moveExplanation.anticipated_counter}</span>
                  </div>
                )}
                {moveExplanation.search_pv && moveExplanation.search_pv.length > 1 && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-neutral-900 border border-neutral-800 text-slate-300">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">PV:</span>
                    <span className="font-mono text-xs text-emerald-400">{moveExplanation.search_pv.slice(0, 5).join(' ')}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="min-h-[100px] p-4 bg-neutral-950/40 border border-dashed border-neutral-800 rounded-xl flex items-center justify-center text-center">
            <p className="text-xs text-slate-500 italic">
              Ready for action. Click <span className="text-emerald-400 font-semibold">"Next Move"</span> to advance the match step-by-step and reveal decision reasoning.
            </p>
          </div>
        )}
      </div>

      {/* Match Controls Responsive Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 w-full pt-1">
        {/* Manual Step Button (Default Mode) */}
        <button
          onClick={onStep}
          disabled={isStepLoading || isGameOver || isAutoPlaying}
          className="flex items-center justify-center gap-1.5 px-3 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-800 disabled:text-slate-500 text-white rounded-xl font-bold text-xs md:text-sm whitespace-nowrap transition-all shadow-lg shadow-emerald-950/50 disabled:shadow-none disabled:cursor-not-allowed"
        >
          {isStepLoading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
              <span>Thinking...</span>
            </>
          ) : (
            <>
              <SkipForward className="w-3.5 h-3.5 shrink-0" />
              <span>Next ({turn === 'w' ? 'White' : 'Black'})</span>
            </>
          )}
        </button>

        {/* Auto-Play Toggle */}
        <button
          onClick={() => setIsAutoPlaying(!isAutoPlaying)}
          disabled={isGameOver}
          className={`flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl font-bold text-xs md:text-sm whitespace-nowrap transition-all border ${
            isAutoPlaying
              ? 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border-amber-500/40 shadow-sm'
              : 'bg-neutral-800 hover:bg-neutral-700 text-slate-200 border-neutral-700'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          title={isAutoPlaying ? "Pause continuous auto-play" : "Start continuous auto-play (1.2s delay)"}
        >
          {isAutoPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5 shrink-0" />
              <span>Pause Auto</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 shrink-0 fill-slate-200" />
              <span>Auto-Play</span>
            </>
          )}
        </button>

        {/* Swap Sides */}
        <button
          onClick={onSwapSides}
          disabled={isAutoPlaying || isStepLoading}
          className="flex items-center justify-center gap-1.5 px-3 py-2.5 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 text-slate-300 hover:text-white border border-neutral-700 rounded-xl font-semibold text-xs md:text-sm whitespace-nowrap transition-all"
          title="Swap colors: SE-ResNet plays White or Black"
        >
          <ArrowLeftRight className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span>Swap ({modelColor === 'w' ? 'White' : 'Black'})</span>
        </button>

        {/* Reset */}
        <button
          onClick={onReset}
          disabled={isStepLoading}
          title="Restart match from opening position"
          className="flex items-center justify-center gap-1.5 px-3 py-2.5 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 text-slate-300 hover:text-white border border-neutral-700 rounded-xl font-semibold text-xs md:text-sm whitespace-nowrap transition-all"
        >
          <RotateCcw className="w-3.5 h-3.5 shrink-0" />
          <span>Reset</span>
        </button>
      </div>

      {/* Engine Enhancements: Opening Book & Tactical Search */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => setUseBook(!useBook)}
          className={`flex items-center justify-between p-2.5 rounded-xl border text-xs font-semibold transition-all ${
            useBook
              ? 'bg-purple-950/30 border-purple-500/40 text-purple-200'
              : 'bg-neutral-950/40 border-neutral-800 text-slate-500'
          }`}
        >
          <div className="flex items-center gap-1.5">
            <BookOpen className={`w-3.5 h-3.5 ${useBook ? 'text-purple-400' : 'text-slate-600'}`} />
            <span>Master Book</span>
          </div>
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${useBook ? 'bg-purple-500/20 text-purple-300' : 'bg-neutral-800 text-slate-500'}`}>
            {useBook ? 'ON' : 'OFF'}
          </span>
        </button>

        <button
          onClick={() => setUseSearch(!useSearch)}
          className={`flex items-center justify-between p-2.5 rounded-xl border text-xs font-semibold transition-all ${
            useSearch
              ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
              : 'bg-neutral-950/40 border-neutral-800 text-slate-500'
          }`}
        >
          <div className="flex items-center gap-1.5">
            <BrainCircuit className={`w-3.5 h-3.5 ${useSearch ? 'text-emerald-400' : 'text-slate-600'}`} />
            <span>Minimax D2</span>
          </div>
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${useSearch ? 'bg-emerald-500/20 text-emerald-300' : 'bg-neutral-800 text-slate-500'}`}>
            {useSearch ? 'ON' : 'OFF'}
          </span>
        </button>
      </div>

      {/* Opening Diversity & Temperature Tuning */}
      <div className="p-3 bg-neutral-950/50 border border-neutral-800/80 rounded-xl flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs font-semibold">
          <div className="flex items-center gap-1.5 text-slate-300">
            <Flame className="w-3.5 h-3.5 text-amber-400" />
            <span>Opening Diversity (T)</span>
          </div>
          <span className="font-mono text-amber-400 font-bold">{temperature.toFixed(1)}</span>
        </div>
        <input 
          type="range" 
          min="0.1" 
          max="1.5" 
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full accent-amber-400 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
        />
        <div className="flex items-center justify-between text-[10px] text-slate-500 font-medium">
          <span>T=0.1 (Argmax)</span>
          <span>T=1.0 (Optimal Variation)</span>
          <span>T=1.5 (High Entropy)</span>
        </div>
      </div>

      {/* Real-time Model Telemetry Performance Dashboard */}
      <ModelTelemetryDashboard 
        history={telemetryHistory} 
        currentElo={modelElo} 
        currentAcpl={rollingAcpl} 
        latestReview={latestReview}
        isEvaluatingFullGame={isEvaluatingFullGame}
      />

      {/* Live Match Log */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs text-slate-400 font-semibold px-1">
          <span>Match Action Log</span>
          <span>{matchHistory.length} moves</span>
        </div>

        <div className="max-h-48 overflow-y-auto custom-scrollbar flex flex-col gap-1.5 pr-1">
          {matchHistory.length === 0 ? (
            <div className="text-center py-6 text-xs text-slate-500 italic">
              Match not started. Press Start to watch Model spar with Stockfish.
            </div>
          ) : (
            matchHistory.map((entry, idx) => (
              <div 
                key={idx}
                className="flex items-center justify-between p-2 rounded-lg bg-neutral-950/40 border border-neutral-800/60 text-xs font-mono"
              >
                <div className="flex items-center gap-2">
                  <span className="text-slate-600 w-5">{Math.floor(idx / 2) + 1}.</span>
                  <span className={entry.role === 'model' ? 'text-emerald-400 font-bold' : 'text-cyan-400 font-bold'}>
                    {entry.role === 'model' ? '🤖' : '⚔️'} {entry.san}
                  </span>
                </div>
                <div>
                  {entry.role === 'model' ? (
                    <div className="flex items-center gap-1.5">
                      {entry.is_book || entry.source === 'book' ? (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center gap-1">
                          <BookOpen className="w-2.5 h-2.5" /> Book
                        </span>
                      ) : entry.source === 'search' ? (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                          🧠 D2
                        </span>
                      ) : null}
                      {entry.source === 'search' && getLogMiniBadge(entry)}
                      {getCplBadge(entry.cpl)}
                    </div>
                  ) : (
                    <span className="text-[10px] text-slate-500 font-mono">Stockfish</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SpectatorPanel;

import React from 'react';
import { 
  Shield, 
  ShieldAlert, 
  Swords, 
  RotateCcw, 
  Loader2, 
  Sparkles,
  Trophy,
  ArrowRight
} from 'lucide-react';

const DefenderPanel = ({
  defenderColor,
  defenderStatus,
  pliesSurvived,
  targetPlies,
  matchHistory = [],
  onReset,
  onSwitchToAuto,
  latestMachineMove = null
}) => {
  const isUserTurn = defenderStatus === 'user_turn';
  const isMachineThinking = defenderStatus === 'machine_thinking';
  const isCheckmate = defenderStatus === 'checkmate';
  const isRefuted = defenderStatus === 'refuted';

  const defenderName = defenderColor === 'w' ? 'White' : 'Black';
  const defenderPiece = defenderColor === 'w' ? '♔' : '♚';

  // Survival resilience calculation
  const progressPct = targetPlies > 0 
    ? Math.min(100, Math.round((pliesSurvived / targetPlies) * 100))
    : 0;
  const isOvertime = pliesSurvived > targetPlies;

  return (
    <div className="bg-neutral-900/70 backdrop-blur-xl border border-neutral-800/80 rounded-2xl shadow-xl p-4 sm:p-5 flex flex-col gap-4 transition-all">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800/80">
        <div className="flex items-center gap-2.5">
          <Shield className="text-cyan-400 w-5 h-5" />
          <h2 className="text-base font-bold text-slate-100 tracking-tight">Defender Arena</h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 bg-neutral-950/80 rounded-full border border-neutral-800">
          <span className="text-slate-300">Defending: <strong className="text-cyan-400 font-mono">{defenderPiece} {defenderName}</strong></span>
        </div>
      </div>

      {/* Turn Status Alert Banner */}
      <div className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${
        isMachineThinking 
          ? 'bg-amber-950/25 border-amber-500/40 text-amber-300'
          : isUserTurn
            ? 'bg-cyan-950/30 border-cyan-500/50 text-cyan-200 ring-1 ring-cyan-500/30 shadow-sm'
            : isCheckmate
              ? 'bg-rose-950/40 border-rose-500/60 text-rose-300'
              : 'bg-emerald-950/30 border-emerald-500/50 text-emerald-300'
      }`}>
        {isMachineThinking && (
          <>
            <Loader2 className="w-5 h-5 text-amber-400 animate-spin flex-shrink-0" />
            <div>
              <div className="text-[10px] font-mono font-medium text-amber-400/80 uppercase tracking-wider">AI Striking</div>
              <div className="text-xs font-bold text-amber-200">SE-ResNet calculating attack line...</div>
            </div>
          </>
        )}
        {isUserTurn && (
          <>
            <ShieldAlert className="w-5 h-5 text-cyan-400 animate-pulse flex-shrink-0" />
            <div>
              <div className="text-[10px] font-mono font-medium text-cyan-400/80 uppercase tracking-wider">Your Move</div>
              <div className="text-xs font-bold text-cyan-100">Drag your {defenderName} piece to defend!</div>
            </div>
          </>
        )}
        {isCheckmate && (
          <>
            <Swords className="w-5 h-5 text-rose-400 flex-shrink-0" />
            <div>
              <div className="text-[10px] font-mono font-medium text-rose-400/80 uppercase tracking-wider">Breached</div>
              <div className="text-xs font-bold text-rose-200">Machine delivered checkmate.</div>
            </div>
          </>
        )}
        {isRefuted && (
          <>
            <Trophy className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            <div>
              <div className="text-[10px] font-mono font-medium text-emerald-400/80 uppercase tracking-wider">Triumph</div>
              <div className="text-xs font-bold text-emerald-200">Fortress Held! Checkmate refuted.</div>
            </div>
          </>
        )}
      </div>

      {/* Survival Resilience Meter */}
      <div className="bg-neutral-950/70 p-3.5 rounded-xl border border-neutral-800/80">
        <div className="flex items-center justify-between text-xs mb-2">
          <span className="text-slate-400 font-medium">Survival Resilience</span>
          <span className="font-mono font-bold text-cyan-300">
            {pliesSurvived} / {targetPlies} plies
            {isOvertime && (
              <span className="ml-1.5 text-emerald-400 font-bold font-mono">(+{pliesSurvived - targetPlies})</span>
            )}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-neutral-900 h-2 rounded-full overflow-hidden p-0.5 border border-neutral-800">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              isOvertime 
                ? 'bg-gradient-to-r from-cyan-500 to-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]' 
                : 'bg-gradient-to-r from-amber-500 to-cyan-400'
            }`}
            style={{ width: `${progressPct}%` }}
          />
        </div>

        <div className="flex justify-between items-center mt-2 text-[11px] text-slate-400">
          <span className="font-mono text-[10px]">Threat Level</span>
          <span className="text-slate-300 font-medium">
            {isOvertime ? (
              <span className="text-emerald-400 flex items-center gap-1 font-semibold">
                <Sparkles className="w-3 h-3" /> Fortress Established!
              </span>
            ) : (
              `Survive > ${targetPlies} plies to refute`
            )}
          </span>
        </div>
      </div>

      {/* Latest Machine Threat Telemetry */}
      {latestMachineMove && (
        <div className="bg-neutral-950/60 px-3 py-2 rounded-xl border border-neutral-800/60 text-xs flex items-center justify-between font-mono">
          <div className="text-slate-400">
            Machine: <strong className="text-amber-300">{latestMachineMove.san}</strong>
          </div>
          <div className="text-slate-400">
            Confidence: <strong className="text-emerald-400">{(latestMachineMove.confidence * 100).toFixed(1)}%</strong>
          </div>
        </div>
      )}

      {/* Combat Move History Table */}
      <div>
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
          <span>Combat Log</span>
          <span className="text-[11px] text-slate-500 font-mono font-normal">{matchHistory.length} plies</span>
        </div>
        <div className="bg-neutral-950/70 rounded-xl border border-neutral-800/80 max-h-36 overflow-y-auto custom-scrollbar p-2 space-y-1">
          {matchHistory.length === 0 ? (
            <div className="text-xs text-slate-500 text-center py-4 italic font-mono">
              Awaiting first move...
            </div>
          ) : (
            matchHistory.map((item, idx) => {
              const isMachine = item.role === 'machine';
              return (
                <div 
                  key={idx}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                    isMachine
                      ? 'bg-amber-950/20 text-amber-200 border border-amber-500/20'
                      : 'bg-cyan-950/20 text-cyan-200 border border-cyan-500/20'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500 w-5">{Math.floor(idx / 2) + 1}.</span>
                    <span className="font-semibold text-[10px] px-1.5 py-0.5 rounded bg-neutral-900 border border-neutral-800">
                      {isMachine ? '🤖 AI' : '🛡️ You'}
                    </span>
                    <span className="font-bold">{item.san}</span>
                  </div>
                  {item.confidence && (
                    <span className="text-[10px] opacity-70">
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={onReset}
          className="flex-1 flex items-center justify-center gap-2 bg-neutral-800 hover:bg-neutral-750 text-slate-200 font-medium py-2.5 px-3 rounded-xl text-xs transition-colors border border-neutral-700/60 shadow"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset Position
        </button>

        <button
          onClick={onSwitchToAuto}
          className="flex items-center justify-center gap-1.5 bg-neutral-950/80 hover:bg-neutral-900 text-slate-300 hover:text-cyan-300 border border-neutral-800 py-2.5 px-3 rounded-xl text-xs transition-colors"
          title="Switch to Auto-Rollout to see the engine's full mate solution"
        >
          <span>See Solution</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};

export default DefenderPanel;

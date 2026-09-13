import React from 'react';
import { Loader2, Shield, Cpu, AlertTriangle, Zap, BrainCircuit } from 'lucide-react';

const TurnStateHUD = ({
  turn = 'w',
  isCheck = false,
  isCheckmate = false,
  mode = 'auto',
  defenderColor = 'b',
  modelColor = 'w',
  isMachineThinking = false,
  pliesSurvived = null
}) => {
  const isWhite = turn === 'w';
  const isUserTurn = mode === 'defender' && turn === defenderColor;
  const isEngineTurn = mode === 'defender' ? turn !== defenderColor : true;
  const isModelTurnInSpectator = mode === 'spectator' && turn === modelColor;

  return (
    <div className="w-full bg-neutral-900/80 backdrop-blur-xl border border-neutral-800/80 rounded-xl px-4 py-2.5 mb-3 flex flex-wrap items-center justify-between gap-2 shadow-xl transition-all">
      {/* Left: Turn Orb & Turn Label */}
      <div className="flex items-center gap-3">
        {/* Glowing Status Dot / Orb */}
        <div className="relative flex items-center justify-center">
          {isWhite ? (
            <span className="relative flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-slate-100 opacity-60" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-white border border-slate-200 shadow-[0_0_12px_rgba(255,255,255,0.9)]" />
            </span>
          ) : (
            <span className="relative flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-500 opacity-60" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-rose-600 border border-rose-400 shadow-[0_0_12px_rgba(244,63,94,0.9)]" />
            </span>
          )}
        </div>

        {/* Turn Text */}
        <div className="flex items-center gap-2">
          <span className={`font-bold text-sm tracking-wide ${isWhite ? 'text-slate-100' : 'text-rose-200'}`}>
            {isWhite ? 'White to Move' : 'Black to Move'}
          </span>
          <span className="text-slate-600 text-xs font-mono">•</span>
          
          {/* Who is playing */}
          {mode === 'defender' ? (
            isUserTurn ? (
              <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                <Shield className="w-3 h-3 text-cyan-400" />
                User (Defender)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                <Cpu className="w-3 h-3 text-indigo-400" />
                Engine (SE-ResNet)
              </span>
            )
          ) : mode === 'spectator' ? (
            isModelTurnInSpectator ? (
              <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                <BrainCircuit className="w-3 h-3 text-emerald-400" />
                SE-ResNet-8 ({turn === 'w' ? 'White' : 'Black'})
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                <Cpu className="w-3 h-3 text-cyan-400" />
                Stockfish 17 ({turn === 'w' ? 'White' : 'Black'})
              </span>
            )
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              <Cpu className="w-3 h-3 text-emerald-400" />
              Engine (SE-ResNet)
            </span>
          )}
        </div>
      </div>

      {/* Right: Tactical Tags */}
      <div className="flex items-center gap-2">
        {isMachineThinking && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30 animate-pulse">
            <Loader2 className="w-3 h-3 animate-spin" />
            Calculating Move
          </span>
        )}

        {isCheckmate ? (
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider bg-rose-600/30 text-rose-300 border border-rose-500/60 animate-bounce shadow-[0_0_14px_rgba(244,63,94,0.5)]">
            <Zap className="w-3.5 h-3.5 text-rose-400" />
            CHECKMATE
          </span>
        ) : isCheck ? (
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/25 text-amber-300 border border-amber-500/50 animate-pulse shadow-[0_0_12px_rgba(245,158,11,0.4)]">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            CHECK
          </span>
        ) : null}

        {pliesSurvived !== null && mode === 'defender' && (
          <span className="text-xs font-mono font-medium text-slate-400 bg-neutral-950/60 px-2.5 py-0.5 rounded border border-neutral-800">
            Ply <strong className="text-cyan-400">{pliesSurvived}</strong>
          </span>
        )}
      </div>
    </div>
  );
};

export default TurnStateHUD;

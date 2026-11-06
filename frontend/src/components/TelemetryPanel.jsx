import React from 'react';
import { Activity, Zap, CheckCircle2 } from 'lucide-react';

const TelemetryPanel = ({ solution, currentStep, inferenceTime, backendStatus }) => {
  const isHealthy = backendStatus?.status === 'ok';
  
  return (
    <div className="bg-neutral-900/70 backdrop-blur-xl border border-neutral-800/80 rounded-2xl shadow-xl p-4 sm:p-5 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <Activity className="text-emerald-400 w-5 h-5" />
          <h2 className="text-base font-bold text-slate-100 tracking-tight">System Telemetry</h2>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-0.5 bg-neutral-950/80 rounded-full border border-neutral-800">
          <div className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-red-500'}`} />
          <span className="text-slate-300 font-mono">{isHealthy ? 'Backend Ready' : 'Offline'}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-neutral-950/70 p-3 rounded-xl border border-neutral-800/80">
          <div className="text-[11px] text-slate-400 font-medium mb-1">Inference Latency</div>
          <div className="text-xl font-black font-mono text-emerald-400 tracking-tight">
            {inferenceTime !== null ? `${inferenceTime.toFixed(1)}ms` : '--'}
          </div>
        </div>
        <div className="bg-neutral-950/70 p-3 rounded-xl border border-neutral-800/80">
          <div className="text-[11px] text-slate-400 font-medium mb-1">Model Confidence</div>
          <div className="text-xl font-black font-mono text-amber-400 tracking-tight">
            {solution ? `${(solution.confidence * 100).toFixed(1)}%` : '--'}
          </div>
          {solution && (
            <div className="w-full bg-neutral-800 h-1.5 mt-2 rounded-full overflow-hidden">
              <div 
                className="bg-gradient-to-r from-amber-500 to-emerald-400 h-full rounded-full transition-all duration-300"
                style={{ width: `${solution.confidence * 100}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {solution && (
        <div className="mb-4 space-y-2">
          {solution.strategy_pivoted && (
            <div className="px-3 py-2 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-center justify-between text-xs text-amber-300">
              <span className="font-semibold flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                Pivoted: Candidate #{solution.candidate_rank}
              </span>
              <span className="font-mono text-slate-300 bg-neutral-950/80 px-2 py-0.5 rounded border border-amber-500/20">
                {solution.pivoted_from_move} → {solution.pivoted_to_move}
              </span>
            </div>
          )}
          {solution.plies > 0 && (
            <div className="flex items-center justify-between text-xs px-3 py-1.5 bg-neutral-950/70 rounded-xl border border-neutral-800/80 text-slate-300 font-mono">
              <span>Shortest Mate: <strong className="text-emerald-400">{solution.plies} plies</strong> {solution.mate_in ? `(Mate in ${solution.mate_in})` : ''}</span>
              <span className="text-slate-400">Score: <strong className="text-amber-400">{(solution.path_score * 100).toFixed(0)}%</strong></span>
            </div>
          )}
        </div>
      )}

      {solution && solution.sequence && solution.sequence.length > 0 && (
        <div>
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
            <span className="font-semibold uppercase tracking-wider">Solution Sequence</span>
            <span className="font-mono text-[11px] text-slate-500">{solution.sequence.length} plies</span>
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto custom-scrollbar pr-1">
            {solution.sequence.map((step, idx) => {
              const isCurrent = idx === currentStep;
              const isPast = idx < currentStep;
              return (
                <div 
                  key={idx}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                    isCurrent 
                      ? 'bg-emerald-950/40 border border-emerald-500/50 text-emerald-300 shadow-sm' 
                      : isPast
                        ? 'bg-neutral-950/40 text-slate-500 border border-transparent'
                        : 'bg-neutral-950/70 text-slate-300 border border-neutral-800/60'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500 w-5">{step.step}.</span>
                    <span className="font-bold tracking-wide">{step.san}</span>
                  </div>
                  <span className="text-[10px] opacity-60">{(step.confidence * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default TelemetryPanel;

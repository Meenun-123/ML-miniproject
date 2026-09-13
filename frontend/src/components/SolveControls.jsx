import React from 'react';
import { Cpu, Play, Pause, SkipForward, RotateCcw, Loader2, Zap } from 'lucide-react';

const SolveControls = ({ 
  onSolve, 
  onPlay, 
  onPause, 
  onStep, 
  onReset, 
  isSolving, 
  isPlaying, 
  hasSolution, 
  currentStep, 
  totalSteps 
}) => {
  return (
    <div className="bg-neutral-900/70 backdrop-blur-xl border border-neutral-800/80 rounded-2xl shadow-xl p-4 sm:p-5 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <Cpu className="text-emerald-400 w-5 h-5" />
          <h2 className="text-base font-bold text-slate-100 tracking-tight">AI Engine Control</h2>
        </div>
        <span className="text-[11px] font-mono font-medium text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded-md border border-emerald-500/30">
          SE-ResNet-8
        </span>
      </div>

      <button
        onClick={onSolve}
        disabled={isSolving || isPlaying}
        className="w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:from-neutral-800 disabled:to-neutral-800 disabled:text-neutral-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-emerald-950/40 border border-emerald-500/30 transition-all mb-4 text-sm tracking-wide"
      >
        {isSolving ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-emerald-300" />
            <span>Finding Shortest Mate...</span>
          </>
        ) : (
          <>
            <Zap className="w-4 h-4 text-emerald-300" />
            <span>Calculate Forced Checkmate</span>
          </>
        )}
      </button>

      <div className="flex justify-between items-center bg-neutral-950/70 p-2.5 rounded-xl border border-neutral-800/80">
        <div className="flex items-center gap-1.5">
          {isPlaying ? (
            <button 
              onClick={onPause} 
              disabled={!hasSolution} 
              title="Pause Playback"
              className="p-2 bg-neutral-800/90 hover:bg-neutral-700 rounded-lg text-amber-400 disabled:opacity-40 transition-colors border border-neutral-700/60"
            >
              <Pause className="w-4 h-4" />
            </button>
          ) : (
            <button 
              onClick={onPlay} 
              disabled={!hasSolution || currentStep >= totalSteps - 1} 
              title="Auto Playback"
              className="p-2 bg-neutral-800/90 hover:bg-neutral-700 rounded-lg text-emerald-400 disabled:opacity-40 transition-colors border border-neutral-700/60"
            >
              <Play className="w-4 h-4" />
            </button>
          )}

          <button 
            onClick={onStep} 
            disabled={!hasSolution || currentStep >= totalSteps - 1} 
            title="Step Forward"
            className="p-2 bg-neutral-800/90 hover:bg-neutral-700 rounded-lg text-slate-300 disabled:opacity-40 transition-colors border border-neutral-700/60"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          <button 
            onClick={onReset} 
            title="Reset Board"
            className="p-2 bg-neutral-800/90 hover:bg-neutral-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors border border-neutral-700/60"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
        
        <div className="text-xs font-mono font-medium text-slate-400 pr-2">
          Step <strong className="text-emerald-400 font-mono">{Math.max(0, currentStep + 1)}</strong> of <span className="font-mono">{totalSteps}</span>
        </div>
      </div>
    </div>
  );
};

export default SolveControls;

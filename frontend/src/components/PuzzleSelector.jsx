import React from 'react';
import { BookOpen, Sparkles } from 'lucide-react';

const PuzzleSelector = ({ puzzles, onSelect, customFen, onCustomFenChange, onLoadCustom, selectedFen }) => {
  return (
    <div className="bg-neutral-900/70 backdrop-blur-xl border border-neutral-800/80 rounded-2xl shadow-xl p-4 sm:p-5 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <BookOpen className="text-emerald-400 w-5 h-5" />
          <h2 className="text-base font-bold text-slate-100 tracking-tight">Tactical Library</h2>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-neutral-950/60 px-2 py-0.5 rounded-md border border-neutral-800">
          {puzzles.length} Patterns
        </span>
      </div>
      
      <div className="space-y-2.5 mb-5 max-h-56 overflow-y-auto pr-1 custom-scrollbar">
        {puzzles.map((puzzle, idx) => {
          const isSelected = selectedFen === puzzle.fen;
          return (
            <button
              key={idx}
              onClick={() => onSelect(puzzle.fen, puzzle.matePlies)}
              className={`w-full text-left p-3 rounded-xl transition-all border ${
                isSelected 
                  ? 'bg-emerald-950/30 border-emerald-500/50 shadow-md ring-1 ring-emerald-500/30' 
                  : 'bg-neutral-950/60 border-neutral-800/80 hover:border-slate-700 hover:bg-neutral-900'
              }`}
            >
              <div className="flex justify-between items-center mb-1">
                <span className={`font-semibold text-sm ${isSelected ? 'text-emerald-300' : 'text-slate-200'}`}>
                  {puzzle.name}
                </span>
                <span className="font-mono text-[11px] font-bold px-2 py-0.5 bg-neutral-900 rounded-full border border-neutral-700/80 text-emerald-400">
                  {puzzle.type}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-snug line-clamp-2">{puzzle.description}</p>
            </button>
          );
        })}
      </div>

      <div className="border-t border-neutral-800/80 pt-3.5">
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Custom FEN</label>
          <span className="text-[10px] text-slate-500 font-mono">paste notation</span>
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={customFen}
            onChange={(e) => onCustomFenChange(e.target.value)}
            placeholder="rnbqkbnr/..."
            className="flex-1 bg-neutral-950/80 border border-neutral-800 rounded-xl px-3 py-2 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/30 transition-all"
          />
          <button
            onClick={onLoadCustom}
            className="px-3.5 py-2 bg-neutral-800 hover:bg-neutral-700 text-slate-200 rounded-xl text-xs font-semibold transition-all border border-neutral-700/60 shadow hover:text-white"
          >
            Load
          </button>
        </div>
      </div>
    </div>
  );
};

export default PuzzleSelector;

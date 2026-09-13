import React, { useState, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import TurnStateHUD from './TurnStateHUD';
import SvgArrowOverlay from './SvgArrowOverlay';

const CUSTOM_DARK_STYLE = { backgroundColor: '#1e293b' };
const CUSTOM_LIGHT_STYLE = { backgroundColor: '#334155' };
const CUSTOM_BOARD_STYLE = {
  borderRadius: '1rem',
  boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
};

const ChessboardPanel = ({ 
  position, 
  onPieceDrop, 
  onPromotionPieceSelect,
  customArrows = [], 
  boardWidth = 580,
  boardOrientation = 'white',
  isDraggablePiece,
  customSquareStyles = {},
  turn = 'w',
  isCheck = false,
  isCheckmate = false,
  mode = 'auto',
  defenderColor = 'b',
  modelColor = 'w',
  isMachineThinking = false,
  pliesSurvived = null,
  banner = null
}) => {
  const [width, setWidth] = useState(boardWidth);

  useEffect(() => {
    const container = document.getElementById('board-outer-container');
    if (!container) return;

    const updateWidth = () => {
      const availableWidth = container.clientWidth - 40;
      setWidth(Math.min(boardWidth, Math.max(260, availableWidth)));
    };

    updateWidth();

    const ro = new ResizeObserver(() => {
      updateWidth();
    });
    ro.observe(container);

    window.addEventListener('resize', updateWidth);

    return () => {
      ro.disconnect();
      window.removeEventListener('resize', updateWidth);
    };
  }, [boardWidth]);

  return (
    <div 
      id="board-outer-container"
      className="w-full max-w-[min(92vw,620px)] bg-neutral-900/90 border border-neutral-800/80 rounded-2xl shadow-2xl p-3.5 sm:p-5 flex flex-col items-center relative transition-all"
    >
      <TurnStateHUD 
        turn={turn}
        isCheck={isCheck}
        isCheckmate={isCheckmate}
        mode={mode}
        defenderColor={defenderColor}
        modelColor={modelColor}
        isMachineThinking={isMachineThinking}
        pliesSurvived={pliesSurvived}
      />

      {/* Board wrapper: responsive square container */}
      <div 
        id="board-container" 
        className="w-full max-w-[min(85vw,580px)] aspect-square relative rounded-2xl overflow-hidden shadow-2xl border border-white/10 select-none flex items-center justify-center"
        style={{ width, height: width }}
      >
        <Chessboard 
          position={position}
          onPieceDrop={onPieceDrop}
          onPromotionPieceSelect={onPromotionPieceSelect}
          boardWidth={width}
          boardOrientation={boardOrientation}
          isDraggablePiece={isDraggablePiece}
          customSquareStyles={customSquareStyles}
          animationDuration={0}
          snapToCursor={true}
          autoPromoteToQueen={false}
          promotionDialogVariant="modal"
          customDarkSquareStyle={CUSTOM_DARK_STYLE}
          customLightSquareStyle={CUSTOM_LIGHT_STYLE}
          customBoardStyle={CUSTOM_BOARD_STYLE}
          areArrowsAllowed={false}
        />

        <SvgArrowOverlay 
          arrows={customArrows}
          boardWidth={width}
          boardOrientation={boardOrientation}
        />

        {banner && (
          <div className="absolute inset-0 bg-neutral-950/85 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center p-6 text-center z-30 transition-all border border-white/10">
            <div className={`text-5xl mb-2 ${banner.type === 'checkmate' ? 'animate-bounce' : 'animate-pulse'}`}>
              {banner.emoji || (banner.type === 'checkmate' ? '💥' : '🏆')}
            </div>
            <h3 className={`text-2xl font-black mb-1.5 tracking-tight ${
              banner.type === 'checkmate' ? 'text-rose-400' : 'text-emerald-400'
            }`}>
              {banner.title}
            </h3>
            <p className="text-slate-300 text-sm max-w-xs leading-relaxed">{banner.subtitle}</p>
            {banner.action && (
              <div className="mt-5">{banner.action}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChessboardPanel;

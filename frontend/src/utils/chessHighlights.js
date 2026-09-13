// Dynamic Square Highlights Generator for react-chessboard

export function getCheckedKingSquare(game) {
  if (!game || !game.inCheck()) return null;
  try {
    const turn = game.turn();
    const board = game.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const piece = board[r][c];
        if (piece && piece.type === 'k' && piece.color === turn) {
          return piece.square;
        }
      }
    }
  } catch (e) {
    console.warn('Error finding checked king square:', e);
  }
  return null;
}

export function computeCustomSquareStyles({ game, activeMove, lastMove }) {
  const styles = {};

  // 1. Checked King square: Animated radial crimson glow
  const checkedKing = getCheckedKingSquare(game);
  if (checkedKing) {
    styles[checkedKing] = {
      background: 'radial-gradient(circle, rgba(239,68,68,0.85) 0%, transparent 80%)',
      borderRadius: '50%',
      animation: 'crimsonPulse 1.4s ease-in-out infinite'
    };
  }

  // Source & Destination squares
  const srcSq = activeMove?.from_sq || lastMove?.from;
  const dstSq = activeMove?.to_sq || lastMove?.to;

  // 2. Source square: Subtle amber border highlight
  if (srcSq && srcSq !== checkedKing) {
    styles[srcSq] = {
      boxShadow: 'inset 0 0 0 2px rgba(245, 158, 11, 0.9), inset 0 0 8px rgba(245, 158, 11, 0.4)',
      backgroundColor: 'rgba(245, 158, 11, 0.16)',
      borderRadius: '6px'
    };
  }

  // 3. Tactical destination square: Glowing emerald highlight with soft inset shadow
  if (dstSq && dstSq !== checkedKing) {
    styles[dstSq] = {
      boxShadow: 'inset 0 0 14px rgba(16, 185, 129, 0.85), 0 0 10px rgba(16, 185, 129, 0.45)',
      backgroundColor: 'rgba(16, 185, 129, 0.32)',
      borderRadius: '6px'
    };
  }

  return styles;
}

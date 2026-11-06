import React from 'react';

function getSquareCenter(square, boardWidth, orientation) {
  if (!square || square.length < 2) return null;
  const file = square[0].toLowerCase();
  const rank = parseInt(square[1], 10);
  if (isNaN(rank) || rank < 1 || rank > 8) return null;

  const squareSize = boardWidth / 8;
  let fileIndex = file.charCodeAt(0) - 97;
  let rankIndex = 8 - rank;

  if (fileIndex < 0 || fileIndex > 7) return null;

  if (orientation === 'black') {
    fileIndex = 7 - fileIndex;
    rankIndex = rank - 1;
  }

  return {
    x: (fileIndex + 0.5) * squareSize,
    y: (rankIndex + 0.5) * squareSize,
    squareSize
  };
}

// Builds a single Chess.com-style polygon path with a wide arrowhead and rounded base
function buildStraightArrowPath(start, end, squareSize) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.hypot(dx, dy);
  if (len < 10) return '';

  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;

  // Proportions scaled to square size
  const shaftRadius = squareSize * 0.12;
  const headWidth = squareSize * 0.32;
  const headLen = squareSize * 0.38;

  // Start slightly inside source square, stop near target center
  const baseOffset = squareSize * 0.16;
  const startX = start.x + ux * baseOffset;
  const startY = start.y + uy * baseOffset;

  const tipX = end.x - ux * (squareSize * 0.1);
  const tipY = end.y - uy * (squareSize * 0.1);

  const headBaseLen = Math.max(0, len - headLen - baseOffset);
  const neckX = startX + ux * headBaseLen;
  const neckY = startY + uy * headBaseLen;

  // Key coordinates
  const leftBase = `${startX + nx * shaftRadius},${startY + ny * shaftRadius}`;
  const rightBase = `${startX - nx * shaftRadius},${startY - ny * shaftRadius}`;
  const leftNeck = `${neckX + nx * shaftRadius},${neckY + ny * shaftRadius}`;
  const rightNeck = `${neckX - nx * shaftRadius},${neckY - ny * shaftRadius}`;
  const leftHeadWing = `${neckX + nx * headWidth},${neckY + headWidth * ny}`;
  const rightHeadWing = `${neckX - nx * headWidth},${neckY - headWidth * ny}`;
  const tip = `${tipX},${tipY}`;

  // Polygon with pill cap at the base
  return `M ${leftBase} 
          A ${shaftRadius} ${shaftRadius} 0 0 0 ${rightBase} 
          L ${rightNeck} 
          L ${rightHeadWing} 
          L ${tip} 
          L ${leftHeadWing} 
          L ${leftNeck} 
          Z`;
}

// Builds a Chess.com-style 90° elbow arrow for Knight moves
function buildKnightArrowPath(start, end, squareSize) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const fileDiff = Math.abs(dx);
  const rankDiff = Math.abs(dy);

  // Corner waypoint: move 2 squares along primary axis, then 1 square laterally
  let cornerX = start.x;
  let cornerY = start.y;

  if (fileDiff > rankDiff) {
    cornerX = end.x;
    cornerY = start.y;
  } else {
    cornerX = start.x;
    cornerY = end.y;
  }

  // Segment 1: start -> corner, Segment 2: corner -> end
  const path1 = buildStraightArrowPath(start, { x: cornerX, y: cornerY }, squareSize);
  const path2 = buildStraightArrowPath({ x: cornerX, y: cornerY }, end, squareSize);

  return `${path1} ${path2}`;
}

const COLOR_MAP = {
  green: 'rgba(34, 197, 94, 0.82)',
  orange: 'rgba(255, 170, 0, 0.85)',
  red: 'rgba(239, 68, 68, 0.85)',
  blue: 'rgba(14, 165, 233, 0.85)',
};

const SvgArrowOverlay = ({ arrows = [], boardWidth = 560, boardOrientation = 'white' }) => {
  if (!arrows || arrows.length === 0 || !boardWidth) return null;

  const squareSize = boardWidth / 8;

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none z-10 select-none"
      viewBox={`0 0 ${boardWidth} ${boardWidth}`}
      style={{ width: boardWidth, height: boardWidth }}
    >
      <defs>
        <filter id="chesscom-glow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="1" stdDeviation="2.5" floodColor="#000000" floodOpacity="0.35" />
        </filter>
      </defs>

      {arrows.map((arr, idx) => {
        const fromSquare = arr[0];
        const toSquare = arr[1];
        let color = arr[2] || COLOR_MAP.green;

        if (COLOR_MAP[color]) {
          color = COLOR_MAP[color];
        }

        const start = getSquareCenter(fromSquare, boardWidth, boardOrientation);
        const end = getSquareCenter(toSquare, boardWidth, boardOrientation);
        if (!start || !end) return null;

        // Check for Knight move: 1x2 or 2x1 file/rank offset
        const fileDist = Math.abs(fromSquare.charCodeAt(0) - toSquare.charCodeAt(0));
        const rankDist = Math.abs(parseInt(fromSquare[1], 10) - parseInt(toSquare[1], 10));
        const isKnightMove = (fileDist === 1 && rankDist === 2) || (fileDist === 2 && rankDist === 1);

        const d = isKnightMove
          ? buildKnightArrowPath(start, end, squareSize)
          : buildStraightArrowPath(start, end, squareSize);

        if (!d) return null;

        return (
          <path
            key={idx}
            d={d}
            fill={color}
            filter="url(#chesscom-glow)"
            className="transition-opacity duration-200"
          />
        );
      })}
    </svg>
  );
};

export default SvgArrowOverlay;

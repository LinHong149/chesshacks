import torch
import chess

def board_to_tensor(board: chess.Board):
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        plane = piece_map[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6  # black planes offset

        row = 7 - (square // 8)
        col = square % 8
        planes[plane, row, col] = 1

    return planes
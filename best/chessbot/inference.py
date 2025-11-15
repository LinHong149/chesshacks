import torch
import json
import chess
from .board_encoding import board_to_tensor
from .model import ChessPolicyNet
from .config import MOVE_INDEX_PATH


class ChessBot:
    def __init__(self, model_path):
        with open(MOVE_INDEX_PATH) as f:
            self.move_index_map = json.load(f)
        self.inv_map = {v: k for k, v in self.move_index_map.items()}

        num_moves = len(self.move_index_map)
        self.model = ChessPolicyNet(num_moves)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.cuda()

    def choose_move(self, board):
        with torch.no_grad():
            X = board_to_tensor(board).unsqueeze(0).cuda()
            logits = self.model(X)[0].cpu()

        legal_moves = list(board.legal_moves)
        
        best_move = max(
            legal_moves,
            key=lambda mv: logits[self.move_index_map[mv.uci()]]
        )
        return best_move
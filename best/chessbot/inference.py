import torch
import json
import chess
from .board_encoding import board_to_tensor
from .model import ChessPolicyNet
from .config import MOVE_INDEX_PATH


class ChessBot:
    def __init__(self, model_path, device="cuda"):
        with open(MOVE_INDEX_PATH) as f:
            self.move_index_map = json.load(f)
        self.inv_map = {v: k for k, v in self.move_index_map.items()}

        num_moves = len(self.move_index_map)
        self.model = ChessPolicyNet(num_moves)
        
        # Load checkpoint (may be dict with 'model_state_dict' or just state_dict)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, loss={checkpoint.get('loss', '?'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.device = device
        if device == "cuda" and torch.cuda.is_available():
            self.model.cuda()
        else:
            self.device = "cpu"

    def choose_move(self, board):
        with torch.no_grad():
            X = board_to_tensor(board).unsqueeze(0)
            if self.device == "cuda":
                X = X.cuda()
            logits = self.model(X)[0].cpu()

        legal_moves = list(board.legal_moves)
        
        # Filter to only legal moves and get their scores
        move_scores = {
            mv: logits[self.move_index_map[mv.uci()]].item()
            for mv in legal_moves
            if mv.uci() in self.move_index_map
        }
        
        if not move_scores:
            # Fallback: random move if no valid moves found
            return list(legal_moves)[0]
        
        best_move = max(move_scores.items(), key=lambda x: x[1])[0]
        return best_move
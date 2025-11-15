from .utils import chess_manager, GameContext
from chess import Move
import os
import sys
import torch
import time

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import AlphaZero model components
from model.main import (
    AlphaZeroNet,
    encode_board,
    move_to_action_index,
    action_index_to_move,
    legal_moves_mask,
    mcts_search,
    MCTSNode,
    CHANNELS,
    NUM_RES_BLOCKS,
    MCTS_SIMULATIONS,
)

# Global model instance (loaded once)
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path: str = None):
    """
    Load the trained AlphaZero model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, tries to find latest in checkpoints/
    """
    global _model
    
    if _model is not None:
        return _model
    
    # Initialize model
    _model = AlphaZeroNet(
        in_channels=13,
        channels=CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS
    ).to(_device)
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        if os.path.exists(checkpoints_dir):
            checkpoint_files = [
                f for f in os.listdir(checkpoints_dir) 
                if f.startswith('alphazero_chess_iter_') and f.endswith('.pt')
            ]
            if checkpoint_files:
                # Get latest checkpoint
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_files[-1])
                print(f"Auto-detected checkpoint: {checkpoint_path}")
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            _model.load_state_dict(checkpoint["model_state_dict"])
        else:
            _model.load_state_dict(checkpoint)
        _model.eval()
        print("✓ Model loaded successfully")
    else:
        print("⚠ No checkpoint found, using untrained model")
        _model.eval()
    
    return _model


# Load model on import (runs once)
try:
    load_model()
except Exception as e:
    print(f"⚠ Could not load model: {e}")
    print("Will use random moves as fallback")


@chess_manager.entrypoint
def alphazero_move(ctx: GameContext):
    """
    Use trained AlphaZero model to make a move.
    """
    global _model
    
    # Fallback to random if model not loaded
    if _model is None:
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available")
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return legal_moves[0]
    
    # Calculate time budget for MCTS
    # Use a portion of remaining time, but cap at reasonable limit
    time_budget_ms = min(ctx.timeLeft * 0.1, 5000)  # Use 10% of time, max 5 seconds
    num_simulations = min(MCTS_SIMULATIONS, int(time_budget_ms / 10))  # ~10ms per simulation
    
    if num_simulations < 10:
        num_simulations = 10  # Minimum simulations
    
    print(f"Using {num_simulations} MCTS simulations (time budget: {time_budget_ms:.0f}ms)")
    
    # Run MCTS to get move probabilities
    start_time = time.perf_counter()
    
    # Create root node and run MCTS
    root = MCTSNode(ctx.board.copy())
    policy_target = mcts_search(root, _model, num_simulations=num_simulations)
    
    # Convert policy to move probabilities
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    move_probs = {}
    for move in legal_moves:
        action_idx = move_to_action_index(move)
        prob = policy_target[action_idx].item()
        move_probs[move] = prob
    
    # Normalize probabilities
    total_prob = sum(move_probs.values())
    if total_prob > 0:
        move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
    else:
        # Fallback: uniform distribution
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    
    # Log probabilities
    ctx.logProbabilities(move_probs)
    
    # Select best move (highest probability)
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"Move selected in {elapsed:.1f}ms: {best_move.uci()}")
    
    return best_move


@chess_manager.reset
def reset_model(ctx: GameContext):
    """
    Reset any cached state when a new game begins.
    """
    # Model doesn't need reset, but you could clear caches here if needed
    pass

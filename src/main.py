from .utils import chess_manager, GameContext
from chess import Move
import os
import sys
import torch
import time

# Add paths for both model directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'best'))

# Import PGN-based model
from chessbot.inference import ChessBot

# Global model instance (loaded once)
_bot = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path: str = None):
    """
    Load the trained PGN model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, tries to find latest in models/
    """
    global _bot
    
    if _bot is not None:
        return _bot
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        if os.path.exists(models_dir):
            # Priority order:
            # 1. v3.pth (if exists)
            # 2. v2.pth (if exists)
            # 3. v1.pth (if exists)
            # 4. Latest epoch checkpoint
            # 5. Environment variable MODEL_PATH
            # 6. Hugging Face (if configured)
            
            # Check for v3.pth first (latest model)
            v3_path = os.path.join(models_dir, 'v3.pth')
            if os.path.exists(v3_path):
                checkpoint_path = v3_path
                print(f"Auto-detected checkpoint: {checkpoint_path} (v3)")
            else:
                # Check for v2.pth
                v2_path = os.path.join(models_dir, 'v2.pth')
                if os.path.exists(v2_path):
                    checkpoint_path = v2_path
                    print(f"Auto-detected checkpoint: {checkpoint_path} (v2)")
                else:
                    # Check for v1.pth
                    v1_path = os.path.join(models_dir, 'v1.pth')
                    if os.path.exists(v1_path):
                        checkpoint_path = v1_path
                        print(f"Auto-detected checkpoint: {checkpoint_path} (v1)")
                else:
                    # Fall back to epoch-based naming
                    checkpoint_files = [
                        f for f in os.listdir(models_dir) 
                        if f.startswith('chessbot_policy_epoch_') and f.endswith('.pth')
                    ]
                    if checkpoint_files:
                        # Get latest checkpoint (highest epoch number)
                        def get_epoch(filename):
                            try:
                                return int(filename.split('epoch_')[1].split('.')[0])
                            except:
                                return 0
                        checkpoint_files.sort(key=get_epoch, reverse=True)
                        checkpoint_path = os.path.join(models_dir, checkpoint_files[0])
                        print(f"Auto-detected checkpoint: {checkpoint_path} (epoch-based)")
            
            # Allow override via environment variable
            env_model_path = os.getenv('MODEL_PATH')
            if env_model_path:
                if os.path.isabs(env_model_path):
                    checkpoint_path = env_model_path
                else:
                    checkpoint_path = os.path.join(models_dir, env_model_path)
                print(f"Using model from MODEL_PATH env var: {checkpoint_path}")
    
    # Try downloading from Hugging Face if local file doesn't exist
    # Default to public model if no env var is set
    hf_model_id = os.getenv('HF_MODEL_ID', 'linhonk/chess-bot-v3')  # Public model on Hugging Face
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Local model not found, attempting to download from Hugging Face: {hf_model_id}")
        try:
            from huggingface_hub import hf_hub_download
            import tempfile
            
            # Download model file
            downloaded_path = hf_hub_download(
                repo_id=hf_model_id,
                filename="v3.pth",  # or "pytorch_model.bin" or whatever you named it
                cache_dir=models_dir,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            checkpoint_path = downloaded_path
            print(f"✓ Downloaded model from Hugging Face: {checkpoint_path}")
        except ImportError:
            print("⚠ huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"⚠ Failed to download from Hugging Face: {e}")
            print("  Falling back to local files or random moves")
    
    # Load model if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading PGN model from {checkpoint_path}")
        try:
            _bot = ChessBot(checkpoint_path, device=_device)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            _bot = None
    else:
        print("⚠ No checkpoint found")
        print(f"  Expected path: {checkpoint_path}")
        print(f"  Or place model in: {os.path.join(os.path.dirname(__file__), '..', 'models')}")
        _bot = None
    
    return _bot


# Load model on import (runs once)
try:
    load_model()
except Exception as e:
    print(f"⚠ Could not load model: {e}")
    print("Will use random moves as fallback")


@chess_manager.entrypoint
def pgn_model_move(ctx: GameContext):
    """
    Use trained PGN model to make a move.
    """
    global _bot
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Fallback to random if model not loaded
    if _bot is None:
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return legal_moves[0]
    
    # Use PGN model to get move
    start_time = time.perf_counter()
    
    try:
        # Get model's move choice
        best_move = _bot.choose_move(ctx.board)
        
        # Get move probabilities for logging
        # The model outputs logits, we can convert to probabilities
        with torch.no_grad():
            from chessbot.board_encoding import board_to_tensor
            board_tensor = board_to_tensor(ctx.board).unsqueeze(0)
            if _device == "cuda" and torch.cuda.is_available():
                board_tensor = board_tensor.cuda()
            
            logits = _bot.model(board_tensor)[0].cpu()
            
            # Convert to probabilities (softmax)
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=0)
            
            # Create move probability dict for legal moves
            move_probs = {}
            for move in legal_moves:
                move_uci = move.uci()
                if move_uci in _bot.move_index_map:
                    move_idx = _bot.move_index_map[move_uci]
                    move_probs[move] = probs[move_idx].item()
            
            # Normalize probabilities
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
            else:
                # Fallback: uniform distribution
                move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        # Log probabilities
        ctx.logProbabilities(move_probs)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"Move selected in {elapsed:.1f}ms: {best_move.uci()}")
        
        return best_move
        
    except Exception as e:
        print(f"⚠ Error getting move from model: {e}")
        # Fallback to random
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return legal_moves[0]


@chess_manager.reset
def reset_model(ctx: GameContext):
    """
    Reset any cached state when a new game begins.
    """
    # PGN model doesn't need reset, but you could clear caches here if needed
    pass

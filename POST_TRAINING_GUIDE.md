# Post-Training Guide

After your model finishes training on Modal, here's what to do next:

## Step 1: Download the Trained Model

Your model checkpoints are saved in the Modal volume `chess-pgn-models`. Download them locally:

```bash
# List all available checkpoints
python download_pgn_model.py --list

# Download the latest checkpoint
python download_pgn_model.py --latest

# Download a specific checkpoint
python download_pgn_model.py --name chessbot_policy_epoch_6.pth

# Download all checkpoints
python download_pgn_model.py --all
```

Models will be saved to `./models/` directory by default.

## Step 2: Test the Model

Test your trained model:

```bash
cd best

# Quick test (model plays against itself)
python test_model.py ../models/chessbot_policy_epoch_6.pth

# Interactive game (play against the model)
python test_model.py ../models/chessbot_policy_epoch_6.pth --play --color white
```

## Step 3: Use the Model in Your Code

### Basic Usage

```python
from chessbot.inference import ChessBot
import chess

# Load the model
bot = ChessBot("models/chessbot_policy_epoch_6.pth")

# Create a board
board = chess.Board()

# Get model's move
move = bot.choose_move(board)
print(f"Model suggests: {move.uci()}")

# Make the move
board.push(move)
```

### Integration with Your Server

If you have a FastAPI server (like in `src/main.py`), update it to load your trained model:

```python
from chessbot.inference import ChessBot

# Load model at startup
model = ChessBot("models/chessbot_policy_epoch_6.pth")

@app.post("/move")
def get_move(board_fen: str):
    board = chess.Board(board_fen)
    move = model.choose_move(board)
    return {"move": move.uci()}
```

## Step 4: Evaluate Model Performance

### Compare Different Epochs

Download multiple checkpoints and compare:

```bash
# Download all checkpoints
python download_pgn_model.py --all

# Test each one
python test_model.py models/chessbot_policy_epoch_1.pth
python test_model.py models/chessbot_policy_epoch_3.pth
python test_model.py models/chessbot_policy_epoch_6.pth
```

### Play Against Stockfish

You can evaluate strength by playing against Stockfish:

```python
import chess
import chess.engine
from chessbot.inference import ChessBot

bot = ChessBot("models/chessbot_policy_epoch_6.pth")
board = chess.Board()

# Play against Stockfish
with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as engine:
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = bot.choose_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
        board.push(move)
        print(f"{move.uci()}")
```

## Step 5: Continue Training (Optional)

If you want to train more epochs:

```bash
# Resume training with more epochs
modal run modal_pgn_train.py --epochs 10 --batch-size 1024
```

The model will continue from where it left off (though you'd need to modify the script to load a checkpoint first).

## Step 6: Deploy the Model

### Local Deployment

Run your FastAPI server with the trained model:

```bash
cd src
python serve.py  # or your server file
```

### Cloud Deployment

You can deploy to Modal, AWS, or other platforms. The model file is portable and can run anywhere PyTorch is available.

## Troubleshooting

### Model Not Loading

- Make sure the checkpoint file exists
- Check that `move_index_map.json` is in the correct location
- Verify PyTorch version compatibility

### CUDA Out of Memory

If running locally without GPU:
```python
bot = ChessBot("models/chessbot_policy_epoch_6.pth", device="cpu")
```

### Move Not Found Error

If you get "move not in move_index_map", the model was trained on different moves. Make sure you're using the same `move_index_map.json` that was used during training.

## Next Steps

1. **Evaluate**: Test the model's strength against other engines
2. **Improve**: Train longer, use more data, or tune hyperparameters
3. **Deploy**: Integrate into your chess application
4. **Iterate**: Collect more games, retrain, and improve

Good luck with your chess bot! 🎯


# ChessBot - Supervised Learning from PGN Data

This is a supervised learning approach that trains a chess policy network on real game data from Lichess.

## Workflow

### Step 1: Build Move Index Map

First, scan all PGN files to build a mapping of all unique moves:

```bash
cd best
python build_move_index.py
```

This creates `chessbot/data/move_index_map.json` with all unique moves.

### Step 2: Process PGN Files

Convert PGN games into training data (board-move pairs):

```bash
python process_pgn.py
```

This processes all PGN files and saves board-move pairs as `.pt` files in `chessbot/data/processed/`.

### Step 3: Train the Model

Train the policy network on the processed data:

```bash
python -m chessbot.train
```

Or:

```bash
cd chessbot
python train.py
```

## File Structure

```
best/
├── chessbot/
│   ├── data/
│   │   ├── raw_pgn/          # Input PGN files
│   │   ├── processed/        # Processed training data (.pt files)
│   │   └── move_index_map.json  # Move to index mapping
│   ├── model.py              # Neural network architecture
│   ├── train.py              # Training script
│   ├── dataset.py            # PyTorch dataset
│   ├── board_encoding.py     # Board to tensor conversion
│   └── inference.py          # Model inference
├── build_move_index.py       # Step 1: Build move index
├── process_pgn.py            # Step 2: Process PGN files
└── requirement.txt          # Dependencies
```

## Configuration

Edit `chessbot/config.py` to adjust:
- `BATCH_SIZE`: Training batch size (default: 512)
- `LR`: Learning rate (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 5)

## Model Architecture

The model is a CNN policy network:
- Input: 12-channel board representation (6 piece types × 2 colors)
- Convolutional layers: 64 → 128 → 256 channels
- Fully connected: 256×8×8 → 1024 → num_moves
- Output: Move probabilities

## Usage

### Quick Start

```bash
# 1. Build move index
python build_move_index.py

# 2. Process PGN files (this may take a while)
python process_pgn.py

# 3. Train model
python -m chessbot.train
```

### Check Progress

```bash
# Count processed positions
ls -1 chessbot/data/processed/*.pt | wc -l

# Check move index
python -c "import json; print(len(json.load(open('chessbot/data/move_index_map.json'))))"
```

## Notes

- PGN files should be in `chessbot/data/raw_pgn/`
- Only games with ELO ≥ 2000 are used (configurable in `data_utils.py`)
- Processing large PGN files can take hours
- Training requires GPU for reasonable speed


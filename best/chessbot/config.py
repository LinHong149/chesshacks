import os

# Data directory is inside chessbot/
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PGN_DIR = os.path.join(DATA_DIR, "raw_pgn")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MOVE_INDEX_PATH = os.path.join(DATA_DIR, "move_index_map.json")

# training
BATCH_SIZE = 512
LR = 1e-4
EPOCHS = 5
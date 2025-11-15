#!/usr/bin/env python3
"""
Script to build move index map from PGN files.
Run from the best/ directory.
"""

import sys
import os

# Add current directory to Python path so chessbot can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chessbot.data_utils import build_move_index_map
from chessbot.config import RAW_PGN_DIR, MOVE_INDEX_PATH

if __name__ == "__main__":
    print(f"Building move index map from: {RAW_PGN_DIR}")
    print(f"Output will be saved to: {MOVE_INDEX_PATH}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(MOVE_INDEX_PATH), exist_ok=True)
    
    build_move_index_map(RAW_PGN_DIR, MOVE_INDEX_PATH)
    print("✓ Done!")


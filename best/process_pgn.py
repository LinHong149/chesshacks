#!/usr/bin/env python3
"""
Process PGN files into training data.
Converts games into board-move pairs saved as .pt files.
"""

import sys
import os
import chess
import chess.pgn
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chessbot.data_utils import stream_lichess_pgn, is_high_quality
from chessbot.board_encoding import board_to_tensor
from chessbot.config import RAW_PGN_DIR, PROCESSED_DIR, MOVE_INDEX_PATH


def process_pgn_files():
    """Process all PGN files into training data."""
    # Load move index map
    if not os.path.exists(MOVE_INDEX_PATH):
        print(f"Error: Move index map not found at {MOVE_INDEX_PATH}")
        print("Please run: python build_move_index.py first")
        return
    
    with open(MOVE_INDEX_PATH) as f:
        import json
        move_index_map = json.load(f)
    
    print(f"Loaded move index map with {len(move_index_map)} moves")
    
    # Create processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all PGN files (both .pgn and .pgn.zst)
    pgn_files = []
    if os.path.exists(RAW_PGN_DIR):
        pgn_files.extend([os.path.join(RAW_PGN_DIR, f) for f in os.listdir(RAW_PGN_DIR) 
                          if f.endswith('.pgn') or f.endswith('.zst')])
    
    if not pgn_files:
        print(f"No PGN files found in {RAW_PGN_DIR}")
        return
    
    print(f"Processing {len(pgn_files)} PGN files...")
    
    total_positions = 0
    file_counter = 0
    
    for pgn_file in pgn_files:
        print(f"\nProcessing: {os.path.basename(pgn_file)}")
        
        # Determine if it's compressed
        if pgn_file.endswith('.zst'):
            # Compressed file - use stream
            games = stream_lichess_pgn(pgn_file)
        else:
            # Regular PGN file - read as generator
            def read_pgn_generator():
                with open(pgn_file, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        yield game
            games = read_pgn_generator()
        
        positions_in_file = 0
        
        for game in games:
            if not is_high_quality(game):
                continue
            
            # Replay game and extract positions
            board = game.board()
            for move in game.mainline_moves():
                # Encode board before the move
                board_tensor = board_to_tensor(board)
                
                # Get move index
                move_uci = move.uci()
                if move_uci not in move_index_map:
                    # Skip if move not in index (shouldn't happen, but safety check)
                    board.push(move)
                    continue
                
                move_idx = move_index_map[move_uci]
                
                # Save as .pt file
                output_file = os.path.join(PROCESSED_DIR, f"pos_{file_counter:08d}.pt")
                torch.save({
                    "board": board_tensor,
                    "move": move_idx
                }, output_file)
                
                file_counter += 1
                positions_in_file += 1
                total_positions += 1
                
                board.push(move)
        
        print(f"  Extracted {positions_in_file} positions from {os.path.basename(pgn_file)}")
    
    print(f"\n✓ Processed {total_positions} total positions")
    print(f"✓ Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    process_pgn_files()


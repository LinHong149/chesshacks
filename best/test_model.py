#!/usr/bin/env python3
"""
Test the trained chess model by playing a game or making moves.
"""

import sys
import os
import torch
import chess
import chess.engine

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessbot.inference import ChessBot
from chessbot.config import MOVE_INDEX_PATH


def test_model(model_path: str):
    """Test the model by playing a game against itself."""
    print(f"Loading model from {model_path}...")
    
    # Load model
    bot = ChessBot(model_path)
    print("✓ Model loaded")
    
    # Play a test game
    board = chess.Board()
    moves = []
    
    print("\nPlaying a test game (model vs itself)...")
    print("=" * 50)
    
    while not board.is_game_over():
        move = bot.choose_move(board)
        moves.append(move)
        board.push(move)
        
        print(f"Move {len(moves)}: {move.uci()}")
        
        if len(moves) >= 20:  # Limit to 20 moves for testing
            print("(Stopped after 20 moves)")
            break
    
    result = board.result()
    print(f"\nGame result: {result}")
    print(f"Total moves: {len(moves)}")
    
    # Show final position
    print("\nFinal position:")
    print(board)


def play_against_model(model_path: str, user_color: str = "white"):
    """Play a game against the model."""
    print(f"Loading model from {model_path}...")
    bot = ChessBot(model_path)
    print("✓ Model loaded\n")
    
    board = chess.Board()
    is_white = user_color.lower() == "white"
    
    print("You are playing as", user_color)
    print("Enter moves in UCI format (e.g., 'e2e4')\n")
    
    while not board.is_game_over():
        print(board)
        print()
        
        if board.turn == chess.WHITE == is_white:
            # User's turn
            move_str = input("Your move: ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except:
                print("Invalid move format! Use UCI (e.g., 'e2e4')")
                continue
        else:
            # Model's turn
            print("Model is thinking...")
            move = bot.choose_move(board)
            board.push(move)
            print(f"Model plays: {move.uci()}\n")
    
    print("\nGame over!")
    print(f"Result: {board.result()}")
    print(board)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the trained chess model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play an interactive game against the model"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Your color when playing (default: white)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if args.play:
        play_against_model(args.model_path, args.color)
    else:
        test_model(args.model_path)


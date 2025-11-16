#!/usr/bin/env python3
"""
Download trained PGN model from Modal volume.

Usage:
    python download_pgn_model.py --list                    # List all checkpoints
    python download_pgn_model.py --latest                  # Download latest checkpoint
    python download_pgn_model.py --name chessbot_policy_epoch_5.pth  # Download specific checkpoint
    python download_pgn_model.py --all                     # Download all checkpoints
"""

import subprocess
import sys
import os
import argparse
import re


def list_checkpoints():
    """List all available checkpoints using Modal CLI."""
    print("Fetching checkpoint list from Modal volume 'chess-pgn-models'...")
    
    # Use Modal CLI to list files in volume
    result = subprocess.run(
        ["modal", "volume", "ls", "chess-pgn-models"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        files = result.stdout.strip().split('\n')
        checkpoint_files = [f for f in files if f.endswith('.pth')]
        
        if checkpoint_files:
            print("\nAvailable checkpoints:")
            for f in sorted(checkpoint_files):
                print(f"  - {f}")
        else:
            print("No checkpoint files found in volume.")
        return checkpoint_files
    else:
        print("Error listing checkpoints:", result.stderr, file=sys.stderr)
        return []


def download_checkpoint(checkpoint_name: str, local_dir: str = "./models"):
    """Download a specific checkpoint from Modal volume."""
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {checkpoint_name} to {local_dir}/...")
    result = subprocess.run(
        [
            "modal", "volume", "download",
            "chess-pgn-models",
            checkpoint_name,
            local_dir
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Successfully downloaded {checkpoint_name} to {local_dir}/")
        return True
    else:
        print(f"✗ Failed to download {checkpoint_name}")
        print(result.stderr, file=sys.stderr)
        return False


def download_latest(local_dir: str = "./models"):
    """Download the latest checkpoint based on epoch number."""
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return False
    
    # Extract epoch numbers and find latest
    def get_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    latest = max(checkpoints, key=get_epoch)
    print(f"\nLatest checkpoint: {latest}")
    return download_checkpoint(latest, local_dir)


def download_all(local_dir: str = "./models"):
    """Download all checkpoints."""
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return False
    
    print(f"\nDownloading {len(checkpoints)} checkpoints...")
    success_count = 0
    
    for checkpoint in checkpoints:
        if download_checkpoint(checkpoint, local_dir):
            success_count += 1
    
    print(f"\n✓ Downloaded {success_count}/{len(checkpoints)} checkpoints")
    return success_count == len(checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description="Download PGN model checkpoints from Modal volume"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of checkpoint to download (e.g., chessbot_policy_epoch_5.pth)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Download the latest checkpoint"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all checkpoints"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Local directory to save checkpoints (default: ./models)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.all:
        download_all(args.output_dir)
    elif args.latest:
        download_latest(args.output_dir)
    elif args.name:
        download_checkpoint(args.name, args.output_dir)
    else:
        # Default: list checkpoints
        list_checkpoints()
        print("\nUse --latest to download the latest checkpoint")
        print("Use --name <filename> to download a specific checkpoint")
        print("Use --all to download all checkpoints")


if __name__ == "__main__":
    main()


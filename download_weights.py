#!/usr/bin/env python3
"""
Helper script to download model weights from Modal volume to local machine.

Usage:
    python download_weights.py                    # List all checkpoints
    python download_weights.py --latest            # Download latest checkpoint
    python download_weights.py --name alphazero_chess_iter_10.pt  # Download specific checkpoint
    python download_weights.py --all               # Download all checkpoints
"""

import subprocess
import sys
import os
import argparse


def list_checkpoints():
    """List all available checkpoints using Modal CLI."""
    print("Fetching checkpoint list from Modal...")
    result = subprocess.run(
        ["modal", "run", "modal/app.py::list_checkpoints"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def download_checkpoint(checkpoint_name: str, local_dir: str = "./checkpoints"):
    """Download a specific checkpoint from Modal volume."""
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {checkpoint_name} to {local_dir}/...")
    result = subprocess.run(
        [
            "modal", "volume", "download",
            "alphazero-models",
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


def download_latest(local_dir: str = "./checkpoints"):
    """Download the latest checkpoint."""
    print("Finding latest checkpoint...")
    result = subprocess.run(
        ["modal", "run", "modal/app.py::list_checkpoints"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Failed to list checkpoints", file=sys.stderr)
        return False
    
    # Parse output to find checkpoint names
    lines = result.stdout.split('\n')
    checkpoint_name = None
    for line in lines:
        if 'alphazero_chess_iter_' in line and '.pt' in line:
            # Extract checkpoint name (first one is latest since sorted)
            parts = line.split()
            for part in parts:
                if 'alphazero_chess_iter_' in part and '.pt' in part:
                    checkpoint_name = part.strip()
                    break
            if checkpoint_name:
                break
    
    if not checkpoint_name:
        print("No checkpoints found")
        return False
    
    return download_checkpoint(checkpoint_name, local_dir)


def download_all(local_dir: str = "./checkpoints"):
    """Download all checkpoints."""
    print("Finding all checkpoints...")
    result = subprocess.run(
        ["modal", "run", "modal/app.py::list_checkpoints"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Failed to list checkpoints", file=sys.stderr)
        return False
    
    # Parse output to find all checkpoint names
    lines = result.stdout.split('\n')
    checkpoint_names = []
    for line in lines:
        if 'alphazero_chess_iter_' in line and '.pt' in line:
            parts = line.split()
            for part in parts:
                if 'alphazero_chess_iter_' in part and '.pt' in part:
                    checkpoint_names.append(part.strip())
                    break
    
    if not checkpoint_names:
        print("No checkpoints found")
        return False
    
    print(f"Found {len(checkpoint_names)} checkpoints")
    success_count = 0
    for cp_name in checkpoint_names:
        if download_checkpoint(cp_name, local_dir):
            success_count += 1
    
    print(f"\n✓ Downloaded {success_count}/{len(checkpoint_names)} checkpoints")
    return success_count == len(checkpoint_names)


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from Modal volume"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of checkpoint to download (e.g., alphazero_chess_iter_10.pt)"
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
        default="./checkpoints",
        help="Local directory to save checkpoints (default: ./checkpoints)"
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


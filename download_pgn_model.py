#!/usr/bin/env python3
"""
Download trained PGN model from Modal volume.

Usage:
    python download_pgn_model.py --list                    # List all checkpoints
    python download_pgn_model.py --latest                  # Download latest checkpoint
    python download_pgn_model.py --name chessbot_policy_epoch_5.pth  # Download specific checkpoint
    python download_pgn_model.py --all                     # Download all checkpoints
"""

import sys
import os
import argparse
import re

import subprocess

# Note: Using Modal CLI instead of Python API due to import issues
# The CLI works reliably: modal volume ls and modal run


def list_checkpoints():
    """List all available checkpoints using Modal CLI or Python API."""
    print("Fetching checkpoint list from Modal volume 'chess-pgn-models'...")
    
    # Try CLI first
    try:
        result = subprocess.run(
            ["modal", "volume", "ls", "chess-pgn-models"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Parse the output to extract .pth files
            lines = result.stdout.strip().split('\n')
            checkpoint_files = []
            
            for line in lines:
                if '.pth' in line:
                    parts = line.split()
                    if parts and parts[0].endswith('.pth'):
                        checkpoint_files.append(parts[0])
            
            if checkpoint_files:
                print("\nAvailable checkpoints:")
                for f in sorted(checkpoint_files):
                    print(f"  - {f}")
                return sorted(checkpoint_files)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # No fallback needed - CLI should work
    
    print("No checkpoints found.")
    return []


def download_checkpoint(checkpoint_name: str, local_dir: str = "./models"):
    """Download a specific checkpoint from Modal volume using CLI."""
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {checkpoint_name} to {local_dir}/...")
    
    try:
        # Use Modal CLI to download file
        # Format: modal volume download <volume> <remote_path> <local_path>
        result = subprocess.run(
            [
                "modal", "volume", "download",
                "chess-pgn-models",
                checkpoint_name,
                local_dir
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            # CLI download doesn't exist, use modal run on separate file
            print("Modal CLI 'volume download' not available.")
            print("Using modal run on modal_download.py...")
            return download_via_modal_run_file(checkpoint_name, local_dir)
        
        local_path = os.path.join(local_dir, checkpoint_name)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            size_mb = size / (1024 * 1024)
            print(f"✓ Successfully downloaded {checkpoint_name} ({size_mb:.1f} MB) to {local_dir}/")
            return True
        else:
            print(f"✗ File not found after download")
            return False
            
    except FileNotFoundError:
        print("Error: Modal CLI not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Failed to download {checkpoint_name}: {e}", file=sys.stderr)
        return False


def download_via_modal_run_file(checkpoint_name: str, local_dir: str):
    """Download using modal run on the modal_download.py file."""
    print("Running modal_download.py to download file...")
    
    # Use modal run with local_entrypoint which saves the file
    try:
        result = subprocess.run(
            [
                "modal", "run", 
                "modal_download.py",
                "--filename", checkpoint_name
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"✗ Modal run failed: {result.stderr}", file=sys.stderr)
            print("\nAlternative: Download manually using:")
            print(f"  modal run modal_download.py --filename {checkpoint_name}")
            return False
        
        # Check if file was created in current directory
        if os.path.exists(checkpoint_name):
            local_path = os.path.join(local_dir, checkpoint_name)
            os.makedirs(local_dir, exist_ok=True)
            
            # Move file if it's not already in the right place
            if os.path.abspath(checkpoint_name) != os.path.abspath(local_path):
                os.rename(checkpoint_name, local_path)
            
            size = os.path.getsize(local_path)
            size_mb = size / (1024 * 1024)
            print(f"✓ Successfully downloaded {checkpoint_name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ File not created. Check modal_download.py output.")
            if result.stdout:
                print("Output:", result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("✗ Modal CLI not found")
        return False
    except Exception as e:
        print(f"✗ Download failed: {e}", file=sys.stderr)
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


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

# Modal app for downloading (defined at module level to avoid decorator issues)
try:
    import modal
    download_app = modal.App("temp-download-helper")
    download_volume = modal.Volume.from_name("chess-pgn-models", create_if_missing=False)
    
    @download_app.function(volumes={"/models": download_volume}, timeout=600)
    def get_file_from_volume(filename: str):
        """Download a file from the volume."""
        import os
        path = f"/models/{filename}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None
except ImportError:
    modal = None
    download_app = None


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
    
    # Fallback to Python API
    try:
        import modal
        app = modal.App.lookup("chess-model-download")
        files = app.list_files.remote()
        
        if files:
            print("\nAvailable checkpoints:")
            for f in files:
                print(f"  - {f}")
            return files
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("1. Make sure you're authenticated: modal token new")
        print("2. Deploy the download app: modal deploy modal_download.py")
        print("3. Or use CLI: modal volume ls chess-pgn-models")
    
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
            # CLI download doesn't exist, use modal run instead
            print("Modal CLI 'volume download' not available, using modal run...")
            return download_via_modal_run(checkpoint_name, local_dir)
        
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


def download_via_modal_run(checkpoint_name: str, local_dir: str):
    """Download using Modal Python API."""
    if modal is None or download_app is None:
        print("✗ Modal not available")
        return False
    
    print("Using Modal Python API to download...")
    
    try:
        file_data = get_file_from_volume.remote(checkpoint_name)
        
        if file_data is None:
            print(f"✗ File not found in volume")
            return False
        
        local_path = os.path.join(local_dir, checkpoint_name)
        with open(local_path, "wb") as f:
            f.write(file_data)
        
        size = os.path.getsize(local_path)
        size_mb = size / (1024 * 1024)
        print(f"✓ Successfully downloaded {checkpoint_name} ({size_mb:.1f} MB)")
        return True
        
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


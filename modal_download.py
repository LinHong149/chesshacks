"""
Modal app for downloading model files from volume.

To download a file, run:
    modal run modal_download.py --filename chessbot_policy_epoch_10.pth

This will save the file to the current directory.
"""

import modal
import sys
import os

app = modal.App("chess-model-download")
volume = modal.Volume.from_name("chess-pgn-models", create_if_missing=False)


@app.function(
    volumes={"/models": volume},
    timeout=60,
)
def list_files():
    """List all .pth files in the volume."""
    import os
    files = []
    if os.path.exists("/models"):
        files = [f for f in os.listdir("/models") if f.endswith('.pth')]
    return sorted(files)


@app.function(
    volumes={"/models": volume},
    timeout=600,
)
def download_file(filename: str):
    """Download a file from the volume and return its contents."""
    import os
    file_path = f"/models/{filename}"
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(filename: str = None):
    """Main entrypoint - downloads file to current directory."""
    if not filename:
        # List files
        files = list_files.remote()
        print("Available files:")
        for f in files:
            print(f"  - {f}")
        print("\nUsage: modal run modal_download.py --filename <filename>")
        return
    
    print(f"Downloading {filename}...")
    data = download_file.remote(filename)
    
    if data is None:
        print(f"✗ File {filename} not found in volume")
        return
    
    # Save file to current directory
    with open(filename, "wb") as f:
        f.write(data)
    
    size_mb = len(data) / (1024 * 1024)
    print(f"✓ Downloaded {filename} ({size_mb:.1f} MB) to current directory")


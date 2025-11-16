"""
Modal app for downloading model files from volume.
Run with: modal run modal_download.py::list_files
         modal run modal_download.py::download_file --filename chessbot_policy_epoch_10.pth
"""

import modal

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
def main(action: str = "list", filename: str = None):
    """Main entrypoint."""
    if action == "list":
        files = list_files.remote()
        for f in files:
            print(f)
    elif action == "download" and filename:
        data = download_file.remote(filename)
        if data:
            with open(filename, "wb") as f:
                f.write(data)
            print(f"Downloaded {filename}")
        else:
            print(f"File {filename} not found")


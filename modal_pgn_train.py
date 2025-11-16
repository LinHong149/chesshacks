"""
Modal app for training chess model on PGN data.
Run with: modal run modal_pgn_train.py
"""

import modal

app = modal.App("chess-pgn-train")

# Image with dependencies and code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "python-chess",
        "tqdm",
    )
    .add_local_dir("best/chessbot", "/root/chessbot")
    .add_local_dir("best/chessbot/data", "/root/chessbot/data")  # Upload processed data
)

volume = modal.Volume.from_name("chess-pgn-models", create_if_missing=True)
gpu_config = "A100-40GB"

@app.function(
    image=image,
    gpu=gpu_config,
    cpu=8,
    volumes={"/models": volume},
    timeout=86400,
)
def train_pgn_model(epochs: int = 5, batch_size: int = 512, learning_rate: float = 1e-4):
    import sys
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import json
    
    sys.path.insert(0, "/root")
    from chessbot.dataset import ChessDataset
    from chessbot.model import ChessPolicyNet
    from chessbot.config import MOVE_INDEX_PATH
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with open(MOVE_INDEX_PATH) as f:
        move_index_map = json.load(f)
    num_moves = len(move_index_map)
    
    dataset = ChessDataset()
    print(f"Dataset size: {len(dataset):,} positions")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = ChessPolicyNet(num_moves).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for boards, moves in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            boards = boards.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            logits = model(boards)
            loss = loss_fn(logits, moves)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"/models/chessbot_policy_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            'num_moves': num_moves,
        }, checkpoint_path)
        volume.commit()
        print(f"Saved to {checkpoint_path}")


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=300,
)
def list_model_files():
    """List all files in the model volume."""
    import os
    files = []
    if os.path.exists("/models"):
        files = [f for f in os.listdir("/models") if f.endswith('.pth')]
    return sorted(files)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=600,
)
def download_model_file(filename: str):
    """Download a model file from volume and return its contents."""
    import os
    file_path = f"/models/{filename}"
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(epochs: int = 5, batch_size: int = 512, learning_rate: float = 1e-4):
    train_pgn_model.remote(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
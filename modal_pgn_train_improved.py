"""
Modal app for training chess model on PGN data (IMPROVED VERSION).
Run with: modal run modal_pgn_train_improved.py
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
    .add_local_dir("best/chessbot/data", "/root/chessbot/data")
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
def train_pgn_model(epochs: int = 30, batch_size: int = 1024, learning_rate: float = 5e-5):
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
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {train_size:,} positions, Validation set: {val_size:,} positions")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ChessPolicyNet(num_moves).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for boards, moves in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            boards = boards.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            logits = model(boards)
            loss = loss_fn(logits, moves)
            
            opt.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for boards, moves in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                boards = boards.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                logits = model(boards)
                loss = loss_fn(logits, moves)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = opt.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={current_lr:.2e}")
        
        # Save checkpoint
        checkpoint_path = f"/models/chessbot_policy_epoch_{epoch+1}.pth"
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'num_moves': num_moves,
            'learning_rate': current_lr,
        }, checkpoint_path)
        volume.commit()
        
        if is_best:
            best_path = "/models/chessbot_policy_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'num_moves': num_moves,
            }, best_path)
            volume.commit()
            print(f"✓ Saved best model (val_loss={avg_val_loss:.4f}) to {best_path}")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")

@app.local_entrypoint()
def main(epochs: int = 30, batch_size: int = 1024, learning_rate: float = 5e-5):
    """
    Train with improved defaults:
    - epochs: 30 (more training)
    - batch_size: 1024 (larger batches)
    - learning_rate: 5e-5 (lower, more stable)
    """
    train_pgn_model.remote(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


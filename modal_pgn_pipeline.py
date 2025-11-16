"""
Modal app for complete PGN training pipeline.
Automatically: builds move index → processes PGN → trains model

Run with: modal run modal_pgn_pipeline.py --pgn-files "file1.pgn,file2.pgn"
"""

import modal

app = modal.App("chess-pgn-pipeline")

# Image with all dependencies and PGN files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "python-chess",
        "tqdm",
        "zstandard",
    )
    .add_local_dir("best/chessbot", "/root/chessbot")
    .add_local_dir("best/chessbot/data/raw_pgn", "/pgn_files")  # Include PGN files in image
)

# Volumes for data and models
data_volume = modal.Volume.from_name("chess-pgn-data", create_if_missing=True)
model_volume = modal.Volume.from_name("chess-pgn-models", create_if_missing=True)

# GPU options (prioritizing speed over cost):
# - "A100-40GB" - Good default
# - "A100-80GB" - Faster, more memory (can use larger batches)
# - "H100" - Fastest available (if available in your region)
gpu_config = "H100"  # Fastest GPU - ~4-5x faster than A100-40GB


@app.function(
    image=image,
    volumes={"/data": data_volume},
    cpu=16,  # More CPU cores for faster processing
    timeout=3600,  # 1 hour for processing
)
def build_move_index(pgn_files: str):
    """Build move index map from PGN files.
    
    Args:
        pgn_files: Comma-separated list of PGN filenames (e.g., "file1.pgn,file2.pgn")
    """
    import sys
    import os
    import json
    import glob
    import chess.pgn
    import zstandard as zstd
    import io
    
    sys.path.insert(0, "/root")
    from chessbot.data_utils import stream_lichess_pgn, is_high_quality
    from chessbot.config import MOVE_INDEX_PATH
    
    # Parse comma-separated file list
    if isinstance(pgn_files, str):
        pgn_file_list = [f.strip() for f in pgn_files.split(',')]
    else:
        pgn_file_list = pgn_files
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(MOVE_INDEX_PATH), exist_ok=True)
    
    uci_set = set()
    
    print(f"Building move index from {len(pgn_file_list)} PGN files...")
    
    for pgn_file in pgn_file_list:
        # Check if file exists in mount or data volume
        file_path = None
        if os.path.exists(f"/pgn_files/{pgn_file}"):
            file_path = f"/pgn_files/{pgn_file}"
        elif os.path.exists(f"/data/raw_pgn/{pgn_file}"):
            file_path = f"/data/raw_pgn/{pgn_file}"
        else:
            print(f"Warning: {pgn_file} not found, skipping")
            continue
        
        print(f"Processing {pgn_file}...")
        
        if file_path.endswith('.zst'):
            # Compressed file
            for game in stream_lichess_pgn(file_path):
                if not is_high_quality(game):
                    continue
                for move in game.mainline_moves():
                    uci_set.add(move.uci())
        else:
            # Regular PGN file
            with open(file_path, 'r') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    if not is_high_quality(game):
                        continue
                    for move in game.mainline_moves():
                        uci_set.add(move.uci())
    
    uci_list = sorted(list(uci_set))
    move_index_map = {uci: i for i, uci in enumerate(uci_list)}
    
    # Save to volume FIRST (this is what process_pgn_files will use)
    volume_path = "/data/move_index_map.json"
    with open(volume_path, "w") as f:
        json.dump(move_index_map, f)
    
    # Commit volume immediately to ensure it's available for next step
    data_volume.commit()
    
    # Also save locally (for reference, but process_pgn_files should use volume)
    with open(MOVE_INDEX_PATH, "w") as f:
        json.dump(move_index_map, f)
    
    num_moves = len(move_index_map)
    print(f"✓ Built move index with {num_moves} moves")
    print(f"✓ Saved to volume: {volume_path} (will be used by process_pgn_files)")
    return num_moves


@app.function(
    image=image,
    volumes={"/data": data_volume},
    cpu=32,  # Many CPU cores for parallel PGN processing
    timeout=14400,  # 4 hours for processing large files
)
def process_pgn_files(pgn_files: str, batch_size: int = 50000):
    """Process PGN files into training data.
    
    Args:
        pgn_files: Comma-separated list of PGN filenames (e.g., "file1.pgn,file2.pgn")
        batch_size: Number of positions per batch file
    """
    import sys
    import os
    import json
    import torch
    import chess
    import chess.pgn
    from tqdm import tqdm
    
    sys.path.insert(0, "/root")
    from chessbot.data_utils import stream_lichess_pgn, is_high_quality
    from chessbot.board_encoding import board_to_tensor
    from chessbot.config import MOVE_INDEX_PATH, PROCESSED_DIR
    
    # Parse comma-separated file list
    if isinstance(pgn_files, str):
        pgn_file_list = [f.strip() for f in pgn_files.split(',')]
    else:
        pgn_file_list = pgn_files
    
    # Load move index - MUST use the one from volume (created by build_move_index)
    # CRITICAL: Always use volume path first to get the latest version
    # DO NOT use MOVE_INDEX_PATH as it may have stale data from the image
    move_index_path = "/data/move_index_map.json"
    
    if not os.path.exists(move_index_path):
        raise FileNotFoundError(
            f"Move index not found at /data/move_index_map.json! "
            f"This should have been created by build_move_index. "
            f"Run build_move_index first."
        )
    
    with open(move_index_path) as f:
        move_index_map = json.load(f)
    
    num_moves_in_map = len(move_index_map)
    print(f"Loaded move index map with {num_moves_in_map} moves from {move_index_path}")
    
    # Verify we're using the correct map (sanity check)
    if num_moves_in_map == 0:
        raise ValueError("Move index map is empty! Something went wrong.")
    
    # CRITICAL: Verify this is from the volume (not a stale local copy)
    if not move_index_path.startswith("/data/"):
        raise ValueError(
            f"CRITICAL ERROR: Loaded move_index_map from wrong location: {move_index_path}\n"
            f"This will cause mismatches! Must use /data/move_index_map.json from volume."
        )
    
    # Create processed directory in volume
    processed_dir = "/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    total_positions = 0
    file_counter = 0
    batch_data = []
    
    for pgn_file in pgn_file_list:
        # Find file
        file_path = None
        if os.path.exists(f"/pgn_files/{pgn_file}"):
            file_path = f"/pgn_files/{pgn_file}"
        elif os.path.exists(f"/data/raw_pgn/{pgn_file}"):
            file_path = f"/data/raw_pgn/{pgn_file}"
        else:
            print(f"Warning: {pgn_file} not found, skipping")
            continue
        
        print(f"\nProcessing: {pgn_file}")
        
        # Determine if compressed
        if file_path.endswith('.zst'):
            games = stream_lichess_pgn(file_path)
        else:
            def read_pgn_generator():
                with open(file_path, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        yield game
            games = read_pgn_generator()
        
        positions_in_file = 0
        
        # Process games sequentially (PGN parsing is I/O bound, parallelization has limited benefit)
        for game in games:
            if not is_high_quality(game):
                continue
            
            board = game.board()
            for move in game.mainline_moves():
                board_tensor = board_to_tensor(board)
                
                move_uci = move.uci()
                if move_uci not in move_index_map:
                    board.push(move)
                    continue
                
                move_idx = move_index_map[move_uci]
                
                # Validate move index is in range
                if move_idx >= len(move_index_map):
                    print(f"Warning: Move index {move_idx} out of range (max: {len(move_index_map)-1}), skipping")
                    board.push(move)
                    continue
                
                batch_data.append({
                    "board": board_tensor,
                    "move": move_idx
                })
                
                if len(batch_data) >= batch_size:
                    output_file = os.path.join(processed_dir, f"batch_{file_counter:06d}.pt")
                    torch.save(batch_data, output_file)
                    batch_data = []
                    file_counter += 1
                
                positions_in_file += 1
                total_positions += 1
                board.push(move)
        
        print(f"  Extracted {positions_in_file} positions from {pgn_file}")
    
    # Save remaining batch
    if batch_data:
        output_file = os.path.join(processed_dir, f"batch_{file_counter:06d}.pt")
        torch.save(batch_data, output_file)
        print(f"  Saved final batch with {len(batch_data)} positions")
    
    data_volume.commit()
    
    print(f"\n✓ Processed {total_positions:,} total positions")
    print(f"✓ Saved {file_counter + (1 if batch_data else 0)} batch files to volume")
    return total_positions


@app.function(
    image=image,
    gpu=gpu_config,
    cpu=16,  # More CPU cores for data loading
    volumes={"/data": data_volume, "/models": model_volume},
    timeout=86400,  # 24 hours
)
def train_model(epochs: int = 30, batch_size: int = 4096, learning_rate: float = 5e-5):
    """Train the model on processed data."""
    import sys
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import json
    import os
    
    sys.path.insert(0, "/root")
    from chessbot.model import ChessPolicyNet
    
    # Load move index from volume (must match the one used during processing)
    move_index_path = "/data/move_index_map.json"
    if not os.path.exists(move_index_path):
        raise FileNotFoundError("Move index not found in volume. Run build_move_index first.")
    
    with open(move_index_path) as f:
        move_index_map = json.load(f)
    num_moves = len(move_index_map)
    
    print(f"Loaded move index map with {num_moves} moves")
    print(f"Model will output {num_moves} move probabilities")
    
    # Check if processed data exists and warn if it might be from a different run
    processed_dir = "/data/processed"
    if os.path.exists(processed_dir) and os.listdir(processed_dir):
        # Sample a batch to check move indices
        sample_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])[:1]
        if sample_files:
            sample_batch = torch.load(os.path.join(processed_dir, sample_files[0]))
            if isinstance(sample_batch, list) and len(sample_batch) > 0:
                sample_move_idx = sample_batch[0]["move"]
                if sample_move_idx >= num_moves:
                    print(f"⚠ WARNING: Sample move index {sample_move_idx} >= {num_moves}")
                    print(f"  This suggests processed data was created with a different move_index_map.")
                    print(f"  Consider clearing /data/processed and reprocessing.")
    
    # Create dataset that reads from volume
    class VolumeDataset(torch.utils.data.Dataset):
        def __init__(self, processed_dir, num_moves):
            self.files = sorted([
                os.path.join(processed_dir, f)
                for f in os.listdir(processed_dir)
                if f.endswith(".pt")
            ])
            self._batch_cache = {}
            self._total_length = None
            self.num_moves = num_moves  # Store num_moves for validation
        
        def __len__(self):
            if self._total_length is None:
                total = 0
                for file_path in self.files:
                    if file_path not in self._batch_cache:
                        batch = torch.load(file_path)
                        self._batch_cache[file_path] = batch
                    batch = self._batch_cache[file_path]
                    if isinstance(batch, list):
                        total += len(batch)
                    else:
                        total += 1
                self._total_length = total
            return self._total_length
        
        def __getitem__(self, idx):
            current_idx = 0
            for file_path in self.files:
                if file_path not in self._batch_cache:
                    batch = torch.load(file_path)
                    self._batch_cache[file_path] = batch
                else:
                    batch = self._batch_cache[file_path]
                
                if isinstance(batch, list):
                    if idx < current_idx + len(batch):
                        position = batch[idx - current_idx]
                        board_tensor = position["board"]
                        move_idx = position["move"]
                        
                        # Validate move index is in valid range
                        if move_idx < 0 or move_idx >= self.num_moves:
                            raise ValueError(
                                f"Invalid move index {move_idx} in data (valid range: 0-{self.num_moves-1}). "
                                f"This suggests the move_index_map changed between processing and training."
                            )
                        
                        return board_tensor, move_idx
                    current_idx += len(batch)
                else:
                    if idx == current_idx:
                        board_tensor = batch["board"]
                        move_idx = batch["move"]
                        
                        # Validate move index
                        if move_idx < 0 or move_idx >= self.num_moves:
                            raise ValueError(
                                f"Invalid move index {move_idx} in data (valid range: 0-{self.num_moves-1})"
                            )
                        
                        return board_tensor, move_idx
                    current_idx += 1
            raise IndexError(f"Index {idx} out of range")
    
    dataset = VolumeDataset("/data/processed", num_moves)
    print(f"Dataset size: {len(dataset):,} positions")
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {train_size:,} positions, Validation set: {val_size:,} positions")
    
    # Use more workers for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = ChessPolicyNet(num_moves).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for boards, moves in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            boards = boards.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            
            # Validate moves are in range before training (critical check)
            moves_cpu = moves.cpu()
            invalid_mask = (moves_cpu >= num_moves) | (moves_cpu < 0)
            if invalid_mask.any():
                invalid_moves = moves_cpu[invalid_mask].unique().tolist()
                invalid_count = invalid_mask.sum().item()
                raise ValueError(
                    f"CRITICAL: Found {invalid_count} invalid move indices in batch!\n"
                    f"  Invalid indices: {invalid_moves[:10]}{'...' if len(invalid_moves) > 10 else ''}\n"
                    f"  Valid range: 0-{num_moves-1}\n"
                    f"  Model expects {num_moves} classes, but data has indices up to {moves_cpu.max().item()}\n"
                    f"  SOLUTION: Clear processed data and reprocess:\n"
                    f"    modal run modal_pgn_pipeline.py::clear_processed_data"
                )
            
            logits = model(boards)
            loss = loss_fn(logits, moves)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for boards, moves in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                boards = boards.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                
                # Validate moves are in range
                moves_cpu = moves.cpu()
                invalid_mask = (moves_cpu >= num_moves) | (moves_cpu < 0)
                if invalid_mask.any():
                    invalid_moves = moves_cpu[invalid_mask].unique().tolist()
                    raise ValueError(
                        f"CRITICAL: Found invalid move indices in validation batch!\n"
                        f"  Invalid indices: {invalid_moves[:10]}\n"
                        f"  Valid range: 0-{num_moves-1}\n"
                        f"  Clear processed data and reprocess."
                    )
                
                logits = model(boards)
                loss = loss_fn(logits, moves)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
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
        model_volume.commit()
        
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
            model_volume.commit()
            print(f"✓ Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=300,
)
def clear_processed_data():
    """Clear processed data from volume (use if move_index_map changed)."""
    import os
    import shutil
    
    processed_dir = "/data/processed"
    if os.path.exists(processed_dir):
        files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]
        if files:
            print(f"Clearing {len(files)} processed batch files...")
            for f in files:
                os.remove(os.path.join(processed_dir, f))
            print(f"✓ Cleared {len(files)} files")
        else:
            print("No processed files to clear")
    else:
        print("Processed directory doesn't exist")
    
    data_volume.commit()
    return len(files) if os.path.exists(processed_dir) else 0


@app.function(
    image=image,
    volumes={"/data": data_volume, "/models": model_volume},
    timeout=300,
)
def list_volume_contents():
    """List contents of data and model volumes."""
    import os
    
    print("=" * 60)
    print("DATA VOLUME CONTENTS")
    print("=" * 60)
    
    if os.path.exists("/data"):
        # List move index
        if os.path.exists("/data/move_index_map.json"):
            import json
            with open("/data/move_index_map.json") as f:
                move_map = json.load(f)
            print(f"Move index map: {len(move_map)} moves")
        
        # List processed files
        processed_dir = "/data/processed"
        if os.path.exists(processed_dir):
            files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
            print(f"Processed batch files: {len(files)}")
            if files:
                print(f"  First: {files[0]}")
                print(f"  Last: {files[-1]}")
        else:
            print("No processed directory")
    else:
        print("Data volume not mounted")
    
    print("\n" + "=" * 60)
    print("MODEL VOLUME CONTENTS")
    print("=" * 60)
    
    if os.path.exists("/models"):
        files = sorted([f for f in os.listdir("/models") if f.endswith(".pth")])
        print(f"Model checkpoints: {len(files)}")
        for f in files:
            size = os.path.getsize(f"/models/{f}")
            size_mb = size / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")
    else:
        print("Model volume not mounted")
    
    return {
        "data_files": len(files) if os.path.exists("/data/processed") else 0,
        "model_files": len(files) if os.path.exists("/models") else 0,
    }


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=86400,
)
def run_full_pipeline(
    pgn_files: str,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 5e-5,
    process_batch_size: int = 50000,
):
    """
    Run the complete pipeline: build index → process PGN → train model.
    
    Args:
        pgn_files: Comma-separated list of PGN filenames (e.g., "file1.pgn,file2.pgn")
        epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        process_batch_size: Batch size for processing (positions per file)
    """
    # Parse comma-separated file list
    if isinstance(pgn_files, str):
        pgn_file_list = [f.strip() for f in pgn_files.split(',')]
    else:
        pgn_file_list = pgn_files
    
    print("=" * 60)
    print("CHESS MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 0: ALWAYS clear old processed data (to avoid move_index_map mismatch)
    print("\n[0/4] Clearing old processed data...")
    try:
        cleared = clear_processed_data.remote()
        if cleared > 0:
            print(f"✓ Cleared {cleared} old batch files (ensuring move_index_map consistency)")
        else:
            print("  No old data to clear")
    except Exception as e:
        print(f"  Warning: Could not clear old data: {e}")
        print("  Continuing anyway - if you get move index errors, manually clear:")
        print("    modal run modal_pgn_pipeline.py::clear_processed_data")
    
    # Step 1: Build move index
    print("\n[1/4] Building move index...")
    num_moves = build_move_index.remote(pgn_files)
    print(f"✓ Move index built: {num_moves} unique moves")
    
    # Step 2: Process PGN files (must use the move_index_map just created)
    print("\n[2/4] Processing PGN files...")
    print(f"  CRITICAL: Must use move index map with {num_moves} moves (from step 1)")
    print(f"  If you see a different number, there's a mismatch!")
    num_positions = process_pgn_files.remote(pgn_files, batch_size=process_batch_size)
    print(f"✓ Processed {num_positions:,} positions")
    
    # Step 3: Train model
    print("\n[3/4] Training model...")
    train_model.remote(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    print("\n[4/4] ✓ Training complete!")
    
    return {
        "num_moves": num_moves,
        "num_positions": num_positions,
        "epochs": epochs,
    }


@app.local_entrypoint()
def main(
    pgn_files: str,
    epochs: int = 30,
    batch_size: int = 4096,  # Larger default batch size for H100 (80GB memory)
    learning_rate: float = 5e-5,
):
    """
    Run complete pipeline from PGN files to trained model.
    
    Usage:
        modal run modal_pgn_pipeline.py --pgn-files "file1.pgn,file2.pgn"
        modal run modal_pgn_pipeline.py --pgn-files "*.pgn" --epochs 50
    
    Args:
        pgn_files: Comma-separated list of PGN filenames (or glob pattern)
        epochs: Number of training epochs (default: 30)
        batch_size: Training batch size (default: 1024)
        learning_rate: Learning rate (default: 5e-5)
    """
    import glob
    import os
    
    # Parse PGN files
    if ',' in pgn_files:
        file_list = [f.strip() for f in pgn_files.split(',')]
    else:
        # Try glob pattern
        file_list = glob.glob(f"best/chessbot/data/raw_pgn/{pgn_files}")
        file_list = [os.path.basename(f) for f in file_list]
    
    if not file_list:
        print(f"Error: No PGN files found matching: {pgn_files}")
        print("Make sure files are in best/chessbot/data/raw_pgn/")
        return
    
    print(f"Found {len(file_list)} PGN files: {file_list}")
    
    # Run pipeline
    result = run_full_pipeline.remote(
        pgn_files=file_list,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Processed {result['num_positions']:,} positions")
    print(f"Trained for {result['epochs']} epochs")
    print(f"\nDownload model with: python download_pgn_model.py --latest")


"""
Modal app for training AlphaZero chess model.
Run with: modal run modal/app.py
"""

import modal

# Create a Modal app
app = modal.App("alphazero-chess")

# Define the image with all dependencies
# Note: copy_local_dir path is relative to the project root (where you run modal run)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "chess",
        "python-chess",
    )
    .add_local_dir("../model", "/root/model")
)

# Create a volume for persistent model storage
volume = modal.Volume.from_name("alphazero-models", create_if_missing=True)

# Define the GPU configuration
gpu_config = "A100-40GB"  # Use A100 GPU for faster training (or "A10G", "T4" for cheaper options)


@app.function(
    image=image,
    gpu=gpu_config,
    cpu=8,  # A100 instances typically have 8-16 CPU cores available. 
            # For small model: 4-8 cores is optimal. More cores = faster parallel self-play but higher cost.
    volumes={"/models": volume},
    timeout=86400,  # 24 hours timeout
)
def train_alphazero(
    num_iterations: int = 1000,
    checkpoint_interval: int = 2,
    resume_from_checkpoint: str | None = None,
):
    """
    Train the AlphaZero chess model.
    
    Args:
        num_iterations: Number of training iterations to run
        checkpoint_interval: Save checkpoint every N iterations (use 1 for frequent saves)
        resume_from_checkpoint: Path to checkpoint file to resume from (optional)
    
    Note: If you want to stop training safely, set checkpoint_interval=1 to save after each iteration.
    """
    import sys
    import os
    
    # Add model directory to path
    os.chdir("/root")
    sys.path.insert(0, "/root")
    
    # Import the training code
    from model.main import (
        AlphaZeroNet,
        ReplayBuffer,
        self_play_game,
        train_step,
        CHANNELS,
        NUM_RES_BLOCKS,
        NUM_SELFPLAY_GAMES_PER_ITER,
        NUM_TRAIN_STEPS_PER_ITER,
        MCTS_SIMULATIONS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        REPLAY_BUFFER_SIZE,
        DEVICE,
    )
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.cuda.amp import autocast
    from torch.amp import GradScaler
    
    print(f"Using device: {DEVICE}")
    
    # Verify GPU type
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        if "A100" in gpu_name:
            print("✓ Confirmed: Using A100 GPU")
        else:
            print(f"⚠ Warning: Expected A100-40GB, but got {gpu_name}")
    else:
        print("⚠ Warning: Not using GPU!")
    
    print(f"Starting training for {num_iterations} iterations")
    
    # Enable mixed precision training for 2x speedup on A100
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (FP16)")
    
    # Wrapper for mixed precision training
    def train_step_amp(net, optimizer, replay_buffer, scaler):
        """Training step with mixed precision."""
        from model.main import BATCH_SIZE
        if len(replay_buffer) < BATCH_SIZE:
            return 0.0, 0.0, 0.0
        
        states, policies, values = replay_buffer.sample(BATCH_SIZE)
        states = states.to(DEVICE, non_blocking=True)
        policies = policies.to(DEVICE, non_blocking=True)
        values = values.to(DEVICE, non_blocking=True)
        
        net.train()
        optimizer.zero_grad()
        
        with autocast():
            policy_logits, value_pred = net(states)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, values)
            loss = policy_loss + value_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    # Initialize network
    net = AlphaZeroNet(
        in_channels=13,
        channels=CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    start_iter = 1
    
    # Load checkpoint if resuming
    if resume_from_checkpoint:
        checkpoint_path = f"/models/{resume_from_checkpoint}"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "replay_buffer" in checkpoint:
                # Note: ReplayBuffer doesn't have a direct load method, 
                # so we'd need to implement one or start fresh
                print("Note: Replay buffer not restored, starting fresh")
            start_iter = checkpoint.get("iteration", 1) + 1
            print(f"Resuming from iteration {start_iter}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting fresh")
    
    # Training loop
    for iter_idx in range(start_iter, start_iter + num_iterations):
        print(f"\n=== Iteration {iter_idx} ===")
        
        # 1. Self-play (parallelized for speed)
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def play_one_game(game_id):
            """Play a single self-play game."""
            samples = self_play_game(net, num_simulations=MCTS_SIMULATIONS)
            return samples
        
        print(f"  Playing {NUM_SELFPLAY_GAMES_PER_ITER} self-play games in parallel...")
        # For small model, use 8 workers (optimal for 8 CPU cores on A100)
        # You can increase to 12-16 if Modal allows more cores, but 8 is usually sufficient
        with ThreadPoolExecutor(max_workers=min(8, NUM_SELFPLAY_GAMES_PER_ITER)) as executor:
            futures = [executor.submit(play_one_game, g) for g in range(NUM_SELFPLAY_GAMES_PER_ITER)]
            for i, future in enumerate(futures):
                samples = future.result()
                for s in samples:
                    replay_buffer.add(s.state, s.policy, s.value)
                if (i + 1) % 5 == 0:
                    print(f"    Completed {i+1}/{NUM_SELFPLAY_GAMES_PER_ITER} games")
        
        # 2. Training (with mixed precision if available)
        avg_loss = 0.0
        for step in range(NUM_TRAIN_STEPS_PER_ITER):
            if use_amp and scaler:
                # Use mixed precision training
                loss, pl, vl = train_step_amp(net, optimizer, replay_buffer, scaler)
            else:
                loss, pl, vl = train_step(net, optimizer, replay_buffer)
            avg_loss += loss
        avg_loss /= max(1, NUM_TRAIN_STEPS_PER_ITER)
        print(f"  Avg training loss: {avg_loss:.4f} | Replay size: {len(replay_buffer)}")
        
        # 3. Save checkpoint
        if iter_idx % checkpoint_interval == 0:
            checkpoint_path = f"/models/alphazero_chess_iter_{iter_idx}.pt"
            checkpoint = {
                "iteration": iter_idx,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            volume.commit()  # Persist to Modal volume
            print(f"  Saved checkpoint at iteration {iter_idx} to {checkpoint_path}")
    
    # Save final checkpoint even if not on interval
    final_checkpoint_path = f"/models/alphazero_chess_iter_{iter_idx-1}.pt"
    if not os.path.exists(final_checkpoint_path):
        checkpoint = {
            "iteration": iter_idx - 1,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, final_checkpoint_path)
        volume.commit()
        print(f"  Saved final checkpoint at iteration {iter_idx-1}")
    
    print("\nTraining completed!")
    return f"Trained for {num_iterations} iterations"


@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/models": volume},
    timeout=3600,  # 1 hour timeout
)
def play_game(num_simulations: int = 200):
    """
    Play a single self-play game and return statistics.
    Useful for testing or generating training data.
    """
    import sys
    import os
    os.chdir("/root")
    sys.path.insert(0, "/root")
    
    from model.main import AlphaZeroNet, self_play_game, CHANNELS, NUM_RES_BLOCKS, DEVICE
    import torch
    
    # Load latest checkpoint or create new network
    net = AlphaZeroNet(
        in_channels=13,
        channels=CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS
    ).to(DEVICE)
    
    # Try to load latest checkpoint
    import glob
    
    checkpoint_files = glob.glob("/models/alphazero_chess_iter_*.pt")
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        net.load_state_dict(checkpoint["model_state_dict"])
    
    # Play game
    samples = self_play_game(net, num_simulations=num_simulations)
    
    return {
        "num_positions": len(samples),
        "game_length": len(samples),
    }


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=3600,
)
def get_checkpoint_path(checkpoint_name: str):
    """
    Get the path to a checkpoint in the volume (for downloading via Modal CLI).
    
    Args:
        checkpoint_name: Name of checkpoint file (e.g., "alphazero_chess_iter_10.pt")
    """
    import os
    
    source_path = f"/models/{checkpoint_name}"
    if os.path.exists(source_path):
        print(f"Checkpoint found: {source_path}")
        print(f"To download, use: modal volume download alphazero-models {checkpoint_name} ./checkpoints/")
        return source_path
    else:
        print(f"Checkpoint {checkpoint_name} not found in volume")
        return None


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=300,
)
def list_checkpoints():
    """List all available checkpoints in the volume."""
    import os
    import glob
    
    checkpoint_files = glob.glob("/models/alphazero_chess_iter_*.pt")
    checkpoint_files.sort(key=os.path.getctime, reverse=True)
    
    print("Available checkpoints:")
    for cp in checkpoint_files:
        size = os.path.getsize(cp) / (1024 * 1024)  # Size in MB
        print(f"  {os.path.basename(cp)} ({size:.2f} MB)")
    
    return [os.path.basename(cp) for cp in checkpoint_files]


@app.local_entrypoint()
def main(
    num_iterations: int = 100,
    checkpoint_interval: int = 2,
    resume_from: str | None = None,
):
    """
    Local entrypoint to start training.
    Usage: modal run modal/app.py --num-iterations 1000
    
    To stop training: Press Ctrl+C or cancel in Modal dashboard
    To resume: modal run modal/app.py --resume-from alphazero_chess_iter_10.pt
    To download checkpoint: modal volume download alphazero-models alphazero_chess_iter_10.pt ./checkpoints/
    To list checkpoints: modal run modal/app.py::list_checkpoints
    To change GPU: Edit line 27 (A10G() -> T4() for cheaper, or A100() for faster)
    """
    result = train_alphazero.remote(
        num_iterations=num_iterations,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from,
    )
    print(result)


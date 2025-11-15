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
        "chess",
        "python-chess",
    )
    .add_local_dir("../model", "/root/model")
)

# Create a volume for persistent model storage
volume = modal.Volume.from_name("alphazero-models", create_if_missing=True)

# Define the GPU configuration
gpu_config = modal.gpu.A10G()  # Use A10G GPU, or change to T4() for cheaper option


@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/models": volume},
    timeout=86400,  # 24 hours timeout
    allow_concurrent_inputs=False,
)
def train_alphazero(
    num_iterations: int = 1000,
    checkpoint_interval: int = 10,
    resume_from_checkpoint: str | None = None,
):
    """
    Train the AlphaZero chess model.
    
    Args:
        num_iterations: Number of training iterations to run
        checkpoint_interval: Save checkpoint every N iterations
        resume_from_checkpoint: Path to checkpoint file to resume from (optional)
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
    import torch.optim as optim
    
    print(f"Using device: {DEVICE}")
    print(f"Starting training for {num_iterations} iterations")
    
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
        
        # 1. Self-play
        for g in range(NUM_SELFPLAY_GAMES_PER_ITER):
            print(f"  Self-play game {g+1}/{NUM_SELFPLAY_GAMES_PER_ITER}")
            samples = self_play_game(net, num_simulations=MCTS_SIMULATIONS)
            for s in samples:
                replay_buffer.add(s.state, s.policy, s.value)
        
        # 2. Training
        avg_loss = 0.0
        for step in range(NUM_TRAIN_STEPS_PER_ITER):
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


@app.local_entrypoint()
def main(
    num_iterations: int = 100,
    checkpoint_interval: int = 10,
    resume_from: str | None = None,
):
    """
    Local entrypoint to start training.
    Usage: modal run modal/app.py --num-iterations 1000
    """
    result = train_alphazero.remote(
        num_iterations=num_iterations,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from,
    )
    print(result)


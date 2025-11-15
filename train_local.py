#!/usr/bin/env python3
"""
Local training script for AlphaZero chess model.
Runs training on your local GPU instead of Modal.

Usage:
    python train_local.py                    # Train with default settings
    python train_local.py --iterations 100   # Train for 100 iterations
    python train_local.py --resume-from checkpoints/alphazero_chess_iter_50.pt
"""

import argparse
import os
import sys
import torch
import time

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

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
    BATCH_SIZE,
)

from torch.cuda.amp import autocast
from torch.amp import GradScaler


def train_locally(
    num_iterations: int = 100,
    checkpoint_interval: int = 10,
    resume_from_checkpoint: str = None,
    checkpoint_dir: str = "./checkpoints",
):
    """
    Train AlphaZero model locally on your GPU.
    
    Args:
        num_iterations: Number of training iterations
        checkpoint_interval: Save checkpoint every N iterations
        resume_from_checkpoint: Path to checkpoint file to resume from
        checkpoint_dir: Directory to save checkpoints
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ Using GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("⚠ Warning: No GPU detected, training will be slow on CPU")
    
    print(f"Using device: {DEVICE}")
    print(f"Starting training for {num_iterations} iterations")
    print(f"Model architecture: {CHANNELS} channels, {NUM_RES_BLOCKS} residual blocks")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print()
    
    # Enable mixed precision if GPU available
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (FP16)")
    
    # Wrapper for mixed precision training
    def train_step_amp(net, optimizer, replay_buffer, scaler):
        """Training step with mixed precision."""
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
            import torch.nn.functional as F
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
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=DEVICE)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint.get("iteration", 1) + 1
        print(f"Resuming from iteration {start_iter}")
    elif resume_from_checkpoint:
        print(f"⚠ Checkpoint {resume_from_checkpoint} not found, starting fresh")
    
    # Training loop
    total_start_time = time.time()
    
    for iter_idx in range(start_iter, start_iter + num_iterations):
        iter_start_time = time.time()
        print(f"\n=== Iteration {iter_idx} ===")
        
        # 1. Self-play (parallelized for speed)
        from concurrent.futures import ThreadPoolExecutor
        
        def play_one_game(game_id):
            """Play a single self-play game."""
            return self_play_game(net, num_simulations=MCTS_SIMULATIONS)
        
        print(f"  Playing {NUM_SELFPLAY_GAMES_PER_ITER} self-play games...")
        # Use 4-8 workers depending on CPU cores
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), NUM_SELFPLAY_GAMES_PER_ITER, 8)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(play_one_game, g) for g in range(NUM_SELFPLAY_GAMES_PER_ITER)]
            for i, future in enumerate(futures):
                samples = future.result()
                for s in samples:
                    replay_buffer.add(s.state, s.policy, s.value)
                if (i + 1) % 5 == 0:
                    print(f"    Completed {i+1}/{NUM_SELFPLAY_GAMES_PER_ITER} games")
        
        # 2. Training
        print(f"  Training ({NUM_TRAIN_STEPS_PER_ITER} steps)...")
        avg_loss = 0.0
        for step in range(NUM_TRAIN_STEPS_PER_ITER):
            if use_amp and scaler:
                loss, pl, vl = train_step_amp(net, optimizer, replay_buffer, scaler)
            else:
                loss, pl, vl = train_step(net, optimizer, replay_buffer)
            avg_loss += loss
        avg_loss /= max(1, NUM_TRAIN_STEPS_PER_ITER)
        print(f"  Avg training loss: {avg_loss:.4f} | Replay size: {len(replay_buffer)}")
        
        # 3. Save checkpoint
        if iter_idx % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"alphazero_chess_iter_{iter_idx}.pt")
            checkpoint = {
                "iteration": iter_idx,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Print iteration time
        iter_time = time.time() - iter_start_time
        print(f"  Iteration time: {iter_time:.1f}s ({iter_time/60:.1f} min)")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f"alphazero_chess_iter_{iter_idx}.pt")
    if not os.path.exists(final_checkpoint_path):
        checkpoint = {
            "iteration": iter_idx,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, final_checkpoint_path)
        print(f"  ✓ Saved final checkpoint: {final_checkpoint_path}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average time per iteration: {total_time/num_iterations:.1f}s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero chess model locally")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)"
    )
    
    args = parser.parse_args()
    
    train_locally(
        num_iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval,
        resume_from_checkpoint=args.resume_from,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()


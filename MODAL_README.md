# Modal Training Setup

This project is configured to run AlphaZero chess training on Modal's cloud infrastructure.

## Prerequisites

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

## Usage

### Basic Training

Start training with default settings:
```bash
modal run modal/app.py
```

### Custom Training Parameters

Train for a specific number of iterations:
```bash
modal run modal/app.py --num-iterations 1000
```

Change checkpoint interval (save every N iterations):
```bash
modal run modal/app.py --num-iterations 1000 --checkpoint-interval 5
```

Resume from a checkpoint:
```bash
modal run modal/app.py --num-iterations 1000 --resume-from alphazero_chess_iter_100.pt
```

### Play a Test Game

Test the model by playing a self-play game:
```python
import modal
app = modal.App.lookup("alphazero-chess")
result = app.play_game.remote(num_simulations=200)
print(result)
```

## Configuration

### GPU Selection

The default GPU is `A10G`. To change it, edit `modal/app.py`:

```python
# For cheaper option (T4 GPU)
gpu_config = modal.gpu.T4()

# For more powerful option (A100)
gpu_config = modal.gpu.A100()
```

### Timeout Settings

Training function has a 24-hour timeout. To change:
```python
@app.function(
    timeout=172800,  # 48 hours
    ...
)
```

## Model Checkpoints

Checkpoints are saved to a Modal volume named `alphazero-models`. They are automatically persisted and can be accessed across runs.

### Downloading Checkpoints

To download a checkpoint locally:
```python
import modal

app = modal.App.lookup("alphazero-chess")
volume = modal.Volume.from_name("alphazero-models")

# Mount volume and copy file
with volume.mount("/models") as mount:
    # Checkpoint will be at /models/alphazero_chess_iter_X.pt
    pass
```

Or use Modal CLI:
```bash
modal volume download alphazero-models /path/to/local/dir
```

## Monitoring

View logs in real-time:
```bash
modal app logs alphazero-chess
```

Or check the Modal dashboard at https://modal.com/apps

## Cost Considerations

- **A10G GPU**: ~$1.10/hour
- **T4 GPU**: ~$0.40/hour (cheaper but slower)
- **A100 GPU**: ~$4.00/hour (faster but expensive)

Training typically runs for many hours, so costs can add up. Monitor usage in the Modal dashboard.

## Troubleshooting

### Import Errors

If you get import errors, make sure the `model/` directory structure is correct and the image includes it:
```python
.copy_local_dir("model", "/root/model")
```

### Out of Memory

If you run out of GPU memory:
1. Reduce `BATCH_SIZE` in `model/main.py`
2. Reduce `MCTS_SIMULATIONS`
3. Use a larger GPU (A100 instead of T4)

### Checkpoint Not Found

Checkpoints are saved to `/models/` in the Modal volume. Make sure:
1. The volume is properly mounted
2. `volume.commit()` is called after saving
3. The checkpoint filename matches when resuming

## Advanced Usage

### Programmatic Training

You can also call the training function programmatically:

```python
import modal

app = modal.App.lookup("alphazero-chess")

# Start training
result = app.train_alphazero.remote(
    num_iterations=500,
    checkpoint_interval=10,
    resume_from_checkpoint=None
)

print(result)
```

### Custom Configurations

To modify training hyperparameters, edit `model/main.py`:
- `NUM_SELFPLAY_GAMES_PER_ITER`: Games per iteration
- `NUM_TRAIN_STEPS_PER_ITER`: Training steps per iteration
- `MCTS_SIMULATIONS`: MCTS simulations per move
- `LEARNING_RATE`: Learning rate
- `BATCH_SIZE`: Training batch size


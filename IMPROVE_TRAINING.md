# How to Improve Your Chess Model Training

Your model is making repetitive moves (rook moving back and forth), which suggests it needs better training. Here are several strategies:

## 1. **More Training Data** ✅ (Most Important)

You currently have ~3.5M positions. More diverse data helps significantly:

### Add More PGN Files:
```bash
# Download more Lichess games
# Place them in best/chessbot/data/raw_pgn/

# Then reprocess:
cd best
python process_pgn.py
```

### Better Data Sources:
- Download games from different time periods
- Mix different skill levels (but keep ELO filter)
- Include tournament games, rapid, blitz, and classical time controls

## 2. **Train for More Epochs**

You trained for 10 epochs. Try 20-50 epochs:

```bash
modal run modal_pgn_train.py --epochs 30 --batch-size 1024
```

## 3. **Better Training Configuration**

### Learning Rate Scheduling
Reduce learning rate as training progresses:

```python
# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

### Better Hyperparameters
- **Lower learning rate**: Try `1e-5` or `5e-5` instead of `1e-4`
- **Larger batch size**: 1024 or 2048 (if GPU memory allows)
- **Weight decay**: Add regularization (0.01)

## 4. **Improve Data Quality**

### Higher ELO Filter
Currently filtering at 2000 ELO. Try 2200+:

```python
# In data_utils.py
def is_high_quality(game, min_elo=2200):  # Increased from 2000
```

### Position Diversity
- Sample positions from different game phases (opening, middlegame, endgame)
- Balance positions (equal vs. imbalanced)
- Include more tactical positions

## 5. **Better Model Architecture**

Your model is relatively small. Consider:

- **More channels**: 64 → 128 or 256
- **More layers**: Add more convolutional blocks
- **Residual connections**: Add skip connections
- **Attention mechanisms**: Self-attention for better pattern recognition

## 6. **Training Techniques**

### Validation Set
Split data into train/validation to monitor overfitting:

```python
# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)
```

### Early Stopping
Stop training if validation loss stops improving

### Gradient Clipping
Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 7. **Data Augmentation**

Augment training data by:
- Flipping board (mirror positions)
- Rotating board (for symmetry)
- Adding noise (slight variations)

## Quick Wins (Try These First):

1. **Train longer**: `--epochs 30` instead of 10
2. **Lower learning rate**: `--learning-rate 5e-5`
3. **More data**: Process more PGN files
4. **Higher ELO filter**: 2200+ instead of 2000

## Recommended Training Command:

```bash
modal run modal_pgn_train.py \
  --epochs 30 \
  --batch-size 1024 \
  --learning-rate 5e-5
```

Then evaluate and iterate!


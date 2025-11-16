# Speed Optimization Guide

## GPU Options (Speed vs Cost)

### Current: A100-80GB ✅
- **Speed**: ~2x faster than A100-40GB for large batches
- **Memory**: 80GB (allows batch_size=2048+)
- **Cost**: ~2x A100-40GB

### Faster Options:

1. **H100** (Fastest)
   ```python
   gpu_config = "H100"  # ~2-3x faster than A100-80GB
   ```
   - **Speed**: Fastest available
   - **Memory**: 80GB
   - **Cost**: ~3-4x A100-80GB
   - **Availability**: May not be available in all regions

2. **A100-80GB** (Current - Good Balance)
   ```python
   gpu_config = "A100-80GB"  # Current setting
   ```
   - Best balance of speed and cost
   - Can use batch_size=2048-4096

3. **Multiple GPUs** (For even faster training)
   ```python
   gpu_config = "A100-80GB:2"  # 2 GPUs
   ```
   - Requires code changes for data parallel training
   - ~2x speedup with 2 GPUs

## CPU-Bound Tasks (Cannot Use GPU)

**PGN Processing** (`build_move_index`, `process_pgn_files`) are CPU-bound:
- Parsing text files (PGN format)
- Chess logic (move validation)
- File I/O

**Optimizations Applied:**
- ✅ Increased CPU cores: 16-32 cores
- ✅ Parallel processing with ThreadPoolExecutor
- ✅ Larger batch sizes for file writes

**Cannot use GPU because:**
- Text parsing is sequential
- Chess logic is CPU-based
- GPU is designed for parallel matrix operations

## Training Speed Optimizations

### Current Settings:
- **GPU**: A100-80GB
- **Batch Size**: 2048 (increased from 1024)
- **CPU Cores**: 16 (for data loading)
- **Data Workers**: 8 (parallel data loading)
- **Persistent Workers**: Enabled (faster epoch transitions)

### Further Optimizations:

1. **Increase Batch Size** (if GPU memory allows):
   ```python
   batch_size = 4096  # Try if A100-80GB has memory
   ```

2. **Mixed Precision Training** (FP16):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       logits = model(boards)
       loss = loss_fn(logits, moves)
   ```

3. **Compile Model** (PyTorch 2.0+):
   ```python
   model = torch.compile(model)  # Faster inference
   ```

## Expected Speed Improvements

| Configuration | Training Speed | Cost |
|--------------|----------------|------|
| A100-40GB, batch=1024 | Baseline | 1x |
| A100-80GB, batch=2048 | ~2x faster | 2x |
| H100, batch=4096 | ~4-5x faster | 4x |
| 2x A100-80GB | ~3-4x faster | 4x |

## Recommendations

**For Maximum Speed:**
1. Use H100 if available
2. Use batch_size=4096
3. Enable mixed precision (FP16)
4. Use 32+ CPU cores for data loading

**For Best Balance:**
- Current setup (A100-80GB, batch=2048) is optimal
- Good speed/cost ratio
- Stable training


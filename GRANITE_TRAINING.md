

# Granite Docling Fine-Tuning Guide

Complete guide for fine-tuning Granite Docling VLM on your custom datasets.

## Quick Start

### 1. Test Configuration (Recommended First Step)

Quick test with small data subset to verify everything works:

```bash
# Single GPU test
python train_granite_docling.py --config configs/granite_docling_test.py

# Expected: ~200 samples (100 from each dataset), 1 epoch, ~5-10 minutes
```

### 2. Full Training

Once test passes, run full training:

```bash
# Single GPU
python train_granite_docling.py --config configs/granite_docling_config.py

# Multi-GPU with torchrun (4 GPUs)
torchrun --nproc_per_node=4 train_granite_docling.py --config configs/granite_docling_config.py

# Expected: ~91K samples, 3 epochs, several hours depending on GPUs
```

### 3. Resume Training

If training interrupted, resume from checkpoint:

```bash
python train_granite_docling.py \
    --config configs/granite_docling_config.py \
    --resume ./checkpoints/granite_docling_finetuned/checkpoint_epoch_1
```

---

## Configuration

All configuration is in Python files under `configs/`. This makes it easy to:
- Track configurations in git
- Use Python logic for complex setups
- Import and reuse configs

### Configuration Structure

```python
def get_config():
    return {
        'model': {...},      # Model checkpoint
        'datasets': [...],   # List of datasets to use
        'training': {...},   # Training hyperparameters
        'wandb': {...}       # Optional W&B logging
    }
```

### Key Configuration Options

#### Datasets

```python
'datasets': [
    {
        'name': 'DoclingMatix',              # Name for logging
        'path': '../DoclingMatix_00000',     # Path to dataset
        'max_turns': None,                   # Limit QA turns (None = all)
        'max_samples': None,                 # Limit samples (None = all)
        'max_images_per_sample': None        # Use all images (None = all, required for OCR)
    },
    {
        'name': 'SynthFormulaNet',
        'path': '../SynthFormulaNet_00000',
        'max_turns': None,
        'max_samples': 10000,                # Use only 10K samples
        'max_images_per_sample': None        # Use all images
    },
]
```

**Supported Formats:**
- DoclingMatix: Multiple images, multiple QA turns per sample
- SynthFormulaNet: Single image, single QA turn per sample
- Any dataset following the Docling JSON format

#### Training Hyperparameters

```python
'training': {
    'num_epochs': 3,
    'batch_size': 1,                    # Per-device batch size (keep low for large documents)
    'gradient_accumulation_steps': 8,   # Accumulate gradients over N steps
    'val_ratio': 0.05,                  # 5% data for validation

    # Learning rates (IMPORTANT!)
    'lr_vision': 5e-5,                  # Vision encoder (lower)
    'lr_text': 5e-5,                    # Text model (lower)
    'lr_connector': 5e-4,               # Connector (higher - new task)

    'weight_decay': 0.01,
    'max_grad_norm': 1.0,               # Gradient clipping
    'min_lr': 1e-6,                     # Minimum LR for scheduler

    # Memory optimization
    'gradient_checkpointing': True,     # Trade compute for memory

    'num_workers': 4,                   # Data loading workers

    'log_interval': 10,                 # Log every N batches
    'val_interval': 1,                  # Validate every N epochs
    'save_interval': 1,                 # Save every N epochs
    'save_dir': './checkpoints/...',    # Where to save checkpoints
}
```

**Learning Rate Tips:**
- Vision & Text: Lower LR (5e-5) since pre-trained
- Connector: Higher LR (5e-4) since adapting to new task
- Adjust based on validation loss

**Gradient Accumulation:**
- Effective batch size = `batch_size × gradient_accumulation_steps × num_GPUs`
- Example: batch_size=1, accumulation=8, 8 GPUs → effective batch = 64
- Allows large effective batch sizes without OOM
- Slower per-step but same convergence

#### Weights & Biases (Optional)

```python
'wandb': {
    'enabled': True,                    # Enable W&B logging
    'project': 'granite-docling',
    'run_name': 'experiment_1',
}
```

---

## Adding New Datasets

### 1. Prepare Dataset

Your dataset should follow the Docling JSON format:

```
MyDataset_00000/
├── anns/
│   └── train/
│       ├── 00000_0.json
│       ├── 00000_1.json
│       └── ...
└── images/
    └── train/
        ├── 00000_0.png
        ├── 00000_1.png
        └── ...
```

**Annotation format (two options supported):**

Option 1: Multiple images, multiple QA turns (DoclingMatix style):
```json
{
  "img_pth": ["images/train/00000_0.png", "images/train/00000_1.png"],
  "anns": [
    {"user": "Question 1?", "assistant": "Answer 1", "source": "..."},
    {"user": "Question 2?", "assistant": "Answer 2", "source": "..."}
  ]
}
```

Option 2: Single image, single QA turn (SynthFormulaNet style):
```json
{
  "img_pth": "images/train/00000_0.png",
  "anns": {
    "user": "Question?",
    "assistant": "Answer",
    "source": "..."
  }
}
```

### 2. Add to Config

Edit your config file (e.g., `configs/granite_docling_config.py`):

```python
'datasets': [
    # ... existing datasets ...
    {
        'name': 'MyCustomDataset',
        'path': '../MyDataset_00000',
        'max_turns': 5,           # Optional: limit QA turns
        'max_samples': None       # Optional: limit samples
    },
]
```

### 3. Test

Always test with a small subset first:

```python
{
    'name': 'MyCustomDataset_test',
    'path': '../MyDataset_00000',
    'max_samples': 100  # Test with 100 samples first
}
```

---

## Training Outputs

### Checkpoints

Saved to `./checkpoints/{save_dir}/`:

```
checkpoints/granite_docling_finetuned/
├── checkpoint_epoch_0/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── training_state.pt          # Optimizer/scheduler state
├── checkpoint_epoch_1/
│   └── ...
└── final_model/
    └── ...
```

### Logs

Console output shows:
- Dataset loading progress
- Training loss per batch
- Validation loss per epoch
- Checkpoint save locations

If W&B enabled, metrics logged:
- `train/loss`: Per-batch training loss
- `train/avg_loss`: Average training loss
- `train/lr`: Current learning rate
- `val/loss`: Validation loss

---

## Multi-GPU Training

### Using torchrun (Recommended)

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 train_granite_docling.py --config configs/granite_docling_config.py

# 8 GPUs on 2 nodes (node 0)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=NODE0_IP \
    train_granite_docling.py --config configs/granite_docling_config.py

# 8 GPUs on 2 nodes (node 1)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=NODE0_IP \
    train_granite_docling.py --config configs/granite_docling_config.py
```

**Notes:**
- Batch size is **per-device** (total = batch_size × num_GPUs)
- Dataset automatically sharded across GPUs
- Gradients synchronized automatically
- Only master process saves checkpoints and logs

---

## Tips & Best Practices

### 1. Start Small
- Always test with `configs/granite_docling_test.py` first
- Use `max_samples` to limit data during development
- Verify one epoch completes successfully

### 2. Monitor Validation Loss
- If val loss increases: reduce learning rate or add regularization
- If val loss plateaus: increase learning rate or train longer
- Use W&B to track trends

### 3. Effective Batch Size
- Real batch size = `batch_size × gradient_accumulation_steps × num_GPUs`
- For 4 GPUs with batch_size=1, accumulation=8: effective batch = 32
- Adjust learning rate proportionally if changing batch size
- Use gradient accumulation to increase batch size without OOM

### 4. Mixed Precision (Future)
Currently not implemented, but can add:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 5. Dataset Balancing
If one dataset is much larger:
```python
{
    'name': 'LargeDataset',
    'max_samples': 10000  # Limit to match smaller datasets
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size` in config (start with 1 for large documents)
2. Reduce `num_workers` in config
3. Enable `gradient_checkpointing: True` (trades compute for memory)
4. Use gradient accumulation (not yet implemented)
5. **Note:** Do NOT reduce images per sample - all images are needed for OCR/document understanding

### Slow Data Loading

**Solutions:**
1. Increase `num_workers` (4-8 usually optimal)
2. Use SSD for dataset storage
3. Ensure images are reasonably sized (<2MB each)

### Loss is NaN

**Possible causes:**
1. Learning rate too high → reduce by 10x
2. Gradient explosion → reduce `max_grad_norm`
3. Bad data → check dataset for corrupted samples

### Validation Loss Not Improving

**Solutions:**
1. Train longer (more epochs)
2. Increase `lr_connector` (currently learning new task)
3. Check if overfitting (train loss << val loss)
4. Add more diverse data

---

## Example Configurations

### Small-Scale Testing (Local Development)
```python
# configs/local_dev.py
'datasets': [
    {'name': 'DoclingMatix', 'path': '...', 'max_samples': 50},
    {'name': 'SynthFormula', 'path': '...', 'max_samples': 50},
],
'training': {
    'num_epochs': 1,
    'batch_size': 1,
    'num_workers': 0,  # Single process for debugging
}
```

### Medium-Scale (Single GPU Production)
```python
# configs/single_gpu.py
'datasets': [
    {'name': 'DoclingMatix', 'path': '...', 'max_samples': None},
    {'name': 'SynthFormula', 'path': '...', 'max_samples': 10000},
],
'training': {
    'num_epochs': 5,
    'batch_size': 4,
    'num_workers': 4,
}
```

### Large-Scale (Multi-GPU Production)
```python
# configs/multi_gpu.py
'datasets': [
    {'name': 'DoclingMatix', 'path': '...', 'max_samples': None},
    {'name': 'SynthFormula', 'path': '...', 'max_samples': None},
    {'name': 'CustomData1', 'path': '...', 'max_samples': None},
    {'name': 'CustomData2', 'path': '...', 'max_samples': None},
],
'training': {
    'num_epochs': 10,
    'batch_size': 2,  # × 8 GPUs = effective batch 16
    'num_workers': 8,
}
```

---

## What's Next?

After training:
1. **Evaluate**: Use `test_granite_docling.py` to test on real images
2. **Push to Hub**: Model has `push_to_hub()` method
3. **Deploy**: Model compatible with HuggingFace transformers
4. **Iterate**: Add more data, adjust hyperparameters, train longer

**Happy Training! 🚀**

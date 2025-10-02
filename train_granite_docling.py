"""
Training script for Granite Docling VLM with Docling datasets

Usage:
    # Single GPU
    python train_granite_docling.py --config configs/granite_docling_config.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train_granite_docling.py --config configs/granite_docling_config.py
"""

import os
# Disable tokenizer parallelism warning when using multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import argparse
import importlib.util
import time
from pathlib import Path
from functools import partial
from typing import List, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import wandb

from models.granite_docling_vlm import GraniteDoclingVLM
from data.docling_dataset import create_docling_dataset, docling_collate_fn


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank):
    """Check if current process is master"""
    return rank == 0


def load_config(config_path: str):
    """Load configuration from Python file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def create_datasets(config, model):
    """Create training and validation datasets"""
    datasets = []

    for dataset_config in config['datasets']:
        print(f"Loading dataset: {dataset_config['name']} from {dataset_config['path']}")

        dataset = create_docling_dataset(
            data_root=dataset_config['path'],
            split='train',
            tokenizer=model.tokenizer,
            image_processor=model.image_processor,
            processor=model.processor if hasattr(model, 'processor') else None,
            mp_image_token_length=64,
            max_turns=dataset_config.get('max_turns', None),
            max_images_per_sample=dataset_config.get('max_images_per_sample', None)
        )

        # Apply cutoff if specified
        if 'max_samples' in dataset_config and dataset_config['max_samples'] is not None:
            indices = list(range(min(len(dataset), dataset_config['max_samples'])))
            dataset = torch.utils.data.Subset(dataset, indices)

        datasets.append(dataset)
        print(f"  Loaded {len(dataset)} samples")

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"\nTotal samples: {len(combined_dataset)}")

    # Split into train/val
    val_size = int(len(combined_dataset) * config['training']['val_ratio'])
    train_size = len(combined_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config, model, rank, world_size):
    """Create training and validation dataloaders"""
    collate_fn = partial(docling_collate_fn, image_processor=model.image_processor)

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader


def setup_optimizer_and_scheduler(model, config, total_steps):
    """Setup optimizer and learning rate scheduler"""
    # Different learning rates for different components
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if 'vision_model' in n],
            'lr': config['training']['lr_vision']
        },
        {
            'params': [p for n, p in model.named_parameters() if 'text_model' in n],
            'lr': config['training']['lr_text']
        },
        {
            'params': [p for n, p in model.named_parameters() if 'connector' in n or 'lm_head' in n],
            'lr': config['training']['lr_connector']
        }
    ]

    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config['training'].get('min_lr', 1e-6)
    )

    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, scheduler, epoch, config, rank, device, start_time, scaler=None, global_step=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Gradient accumulation settings
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    accumulation_counter = 0

    # Mixed precision settings
    use_amp = config['training'].get('mixed_precision', None) in ['fp16', 'bf16']
    amp_dtype = torch.bfloat16 if config['training'].get('mixed_precision') == 'bf16' else torch.float16

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch.get('image_grid_thw')
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)

        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=amp_dtype):
                logits, loss = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    targets=labels
                )
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
        else:
            logits, loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                targets=labels
            )
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1

        # Only update weights after accumulating gradients
        if accumulation_counter % gradient_accumulation_steps == 0:
            # Gradient clipping
            if config['training'].get('max_grad_norm', 0) > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

            # Optimizer step
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Increment global step counter after weight update
            global_step += 1

        total_loss += loss.item() * gradient_accumulation_steps  # Undo scaling for logging
        num_batches += 1

        # Batch-based checkpoint saving (after every batch, not optimization step)
        save_steps = config['training'].get('save_steps', None)
        if save_steps is not None and num_batches % save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, f"batch_{num_batches}", config, rank, scaler)

        # Logging
        if is_master(rank) and batch_idx % config['training'].get('log_interval', 10) == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            effective_batch = config['training']['batch_size'] * gradient_accumulation_steps

            # Calculate elapsed time
            elapsed = time.time() - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            print(f"[{time_str}] Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | Avg Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | Effective BS: {effective_batch}")

            if config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/avg_loss': avg_loss,
                    'train/lr': lr,
                    'train/epoch': epoch,
                    'train/step': epoch * len(train_loader) + batch_idx,
                    'train/elapsed_time': elapsed
                })

    return total_loss / num_batches if num_batches > 0 else 0, global_step


def validate(model, val_loader, epoch, config, rank, device, start_time):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                targets=labels
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    if is_master(rank):
        # Calculate elapsed time
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[{time_str}] Validation | Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        if config.get('wandb', {}).get('enabled', False):
            wandb.log({
                'val/loss': avg_loss,
                'val/epoch': epoch,
                'val/elapsed_time': elapsed
            })

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, config, rank, scaler=None):
    """Save model checkpoint and manage checkpoint rotation"""
    if not is_master(rank):
        return

    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint_path = save_dir / f"checkpoint_{epoch}"

    print(f"Saving checkpoint to {checkpoint_path}")
    model_to_save.save_pretrained(str(checkpoint_path))

    # Save processor (tokenizer + image processor)
    if hasattr(model_to_save, 'processor') and model_to_save.processor is not None:
        model_to_save.processor.save_pretrained(str(checkpoint_path))
    elif hasattr(model_to_save, 'tokenizer') and hasattr(model_to_save, 'image_processor'):
        if model_to_save.tokenizer is not None:
            model_to_save.tokenizer.save_pretrained(str(checkpoint_path))
        if model_to_save.image_processor is not None:
            model_to_save.image_processor.save_pretrained(str(checkpoint_path))

    # Save training state
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    # Save scaler state if using mixed precision
    if scaler is not None:
        training_state['scaler_state_dict'] = scaler.state_dict()

    torch.save(training_state, checkpoint_path / "training_state.pt")

    # Manage checkpoint rotation (keep only N most recent)
    max_checkpoints = config['training'].get('max_checkpoints', None)
    if max_checkpoints is not None and max_checkpoints > 0:
        # Get all checkpoint directories (both epoch and step based)
        all_checkpoints = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]

        # Sort by modification time (oldest first)
        all_checkpoints.sort(key=lambda x: x.stat().st_mtime)

        # Remove old checkpoints if we exceed the limit
        while len(all_checkpoints) > max_checkpoints:
            old_checkpoint = all_checkpoints.pop(0)
            print(f"Removing old checkpoint: {old_checkpoint.name}")
            import shutil
            shutil.rmtree(old_checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Load config
    config = load_config(args.config)

    # Initialize wandb on master process
    if is_master(rank) and config.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb'].get('run_name', 'granite_docling_training'),
            config=config
        )

    # Load model
    if is_master(rank):
        print("Loading Granite Docling model...")

    if args.resume:
        model = GraniteDoclingVLM.from_pretrained(args.resume)
    else:
        model = GraniteDoclingVLM.from_pretrained(config['model']['checkpoint'])

    # Enable gradient checkpointing to save memory
    if config['training'].get('gradient_checkpointing', True):
        if is_master(rank):
            print("Enabling gradient checkpointing...")
        if hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()

    model = model.to(device)

    # Wrap with DDP for multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create datasets and dataloaders
    if is_master(rank):
        print("\nCreating datasets...")

    model_ref = model.module if hasattr(model, 'module') else model
    train_dataset, val_dataset = create_datasets(config, model_ref)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config, model_ref, rank, world_size
    )

    # Setup optimizer and scheduler
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    total_steps = config['training']['num_epochs'] * len(train_loader) // gradient_accumulation_steps
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    # Setup mixed precision training
    scaler = None
    mixed_precision = config['training'].get('mixed_precision', None)
    if mixed_precision in ['fp16', 'bf16']:
        if is_master(rank):
            precision_name = 'BF16' if mixed_precision == 'bf16' else 'FP16'
            print(f"Enabling {precision_name} mixed precision training...")
        if mixed_precision == 'fp16':
            scaler = GradScaler()

    # Resume from checkpoint if needed
    start_epoch = 0
    if args.resume:
        training_state_path = Path(args.resume) / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            if scaler is not None and 'scaler_state_dict' in state:
                scaler.load_state_dict(state['scaler_state_dict'])
            start_epoch = state['epoch'] + 1
            if is_master(rank):
                print(f"Resuming from epoch {start_epoch}")

    # Training loop
    if is_master(rank):
        print(f"\nStarting training for {config['training']['num_epochs']} epochs...")

    # Start timer
    start_time = time.time()

    # Track global step across epochs
    global_step = 0

    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, global_step = train_epoch(model, train_loader, optimizer, scheduler, epoch, config, rank, device, start_time, scaler, global_step)

        # Validate
        if (epoch + 1) % config['training'].get('val_interval', 1) == 0:
            val_loss = validate(model, val_loader, epoch, config, rank, device, start_time)

        # Epoch-based checkpoint saving
        if (epoch + 1) % config['training'].get('save_interval', 1) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, config, rank, scaler)

    # Final save
    if is_master(rank):
        # Calculate total training time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"\n[{time_str}] Training completed!")
        model_to_save = model.module if hasattr(model, 'module') else model
        final_path = Path(config['training']['save_dir']) / "final_model"
        model_to_save.save_pretrained(str(final_path))
        print(f"Final model saved to {final_path}")
        print(f"Total training time: {time_str}")

    # Cleanup
    if config.get('wandb', {}).get('enabled', False) and is_master(rank):
        wandb.finish()

    cleanup_distributed()


if __name__ == '__main__':
    main()

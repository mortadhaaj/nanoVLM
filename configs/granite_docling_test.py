"""
Quick test configuration for Granite Docling fine-tuning

This config uses a small subset of data for quick testing/debugging.
Use this to verify everything works before running full training.
"""

def get_config():
    return {
        # Model configuration
        'model': {
            'checkpoint': '../.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/',
        },

        # Dataset configuration - SMALL SUBSETS for testing
        'datasets': [
            {
                'name': 'DoclingMatix_test',
                'path': '../DoclingMatix_00000',
                'max_turns': 3,        # Limit QA turns
                'max_samples': 100,    # Only 100 samples for quick test
                'max_images_per_sample': None  # Use all images (OCR needs full document)
            },
            {
                'name': 'SynthFormulaNet_test',
                'path': '../SynthFormulaNet_00000',
                'max_turns': None,
                'max_samples': 100,    # Only 100 samples for quick test
                'max_images_per_sample': None  # Use all images
            },
        ],

        # Training configuration - FAST settings for testing
        'training': {
            'num_epochs': 60,          # Just 1 epoch for testing
            'batch_size': 1,
            'gradient_accumulation_steps': 4,  # Effective batch size = 1 * 4 = 4
            'val_ratio': 0.1,         # 10% for validation

            # Learning rates
            'lr_vision': 5e-4,
            'lr_text': 5e-4,
            'lr_connector': 5e-4,

            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'min_lr': 1e-6,

            # Memory and speed optimization
            'gradient_checkpointing': True,
            'mixed_precision': 'bf16',  # 'bf16' (recommended for H100) or 'fp16' or None

            # Data loading
            'num_workers': 2,  # Fewer workers for testing

            # Logging and saving
            'log_interval': 5,       # Log more frequently
            'val_interval': 1,
            'save_interval': 1,
            'save_steps': None,      # Save checkpoint every N steps (None = disabled)
            'max_checkpoints': 5,    # Keep only 3 most recent checkpoints
            'save_dir': './checkpoints/granite_docling_test',
        },

        # W&B logging (optional)
        'wandb': {
            'enabled': False,
            'project': 'granite-docling-test',
            'run_name': 'quick_test',
        }
    }

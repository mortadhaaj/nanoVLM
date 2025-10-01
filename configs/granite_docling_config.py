"""
Configuration for fine-tuning Granite Docling VLM

This config uses both DoclingMatix and SynthFormulaNet datasets.
You can easily add more datasets by extending the 'datasets' list.
"""

def get_config():
    return {
        # Model configuration
        'model': {
            'checkpoint': '../.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/',
        },

        # Dataset configuration
        'datasets': [
            {
                'name': 'DoclingMatix',
                'path': '../DoclingMatix_00000',
                'max_turns': None,  # Use all QA turns (or set to a number to limit)
                'max_samples': None,  # Use all samples (or set to limit for quick testing)
                'max_images_per_sample': None  # Use all images - essential for OCR/document understanding
            },
            {
                'name': 'SynthFormulaNet',
                'path': '../SynthFormulaNet_00000',
                'max_turns': None,
                'max_samples': None,  # 90K samples - set to smaller number for testing
                'max_images_per_sample': None  # Use all images
            },
            # Add more datasets here:
            # {
            #     'name': 'MyCustomDataset',
            #     'path': '../MyDataset_00000',
            #     'max_turns': 5,
            #     'max_samples': 10000
            # },
        ],

        # Training configuration
        'training': {
            'num_epochs': 3,
            'batch_size': 3,  # Per-device batch size
            'gradient_accumulation_steps': 8,  # Effective batch size = 1 * 8 = 8 (or 64 with 8 GPUs)
            'val_ratio': 0.05,  # 5% for validation

            # Learning rates (different for each component)
            'lr_vision': 5e-5,      # Vision encoder
            'lr_text': 5e-5,        # Text model
            'lr_connector': 5e-4,   # Connector + LM head (higher LR)

            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'min_lr': 1e-6,

            # Memory optimization
            'gradient_checkpointing': True,

            # Data loading
            'num_workers': 8,

            # Logging and saving
            'log_interval': 10,      # Log every N batches
            'val_interval': 1,       # Validate every N epochs
            'save_interval': 1,      # Save checkpoint every N epochs
            'save_dir': './checkpoints/granite_docling_finetuned',
        },

        # W&B logging (optional)
        'wandb': {
            'enabled': False,  # Set to True to enable W&B logging
            'project': 'granite-docling-finetuning',
            'run_name': 'docling_formula_combined',
        }
    }

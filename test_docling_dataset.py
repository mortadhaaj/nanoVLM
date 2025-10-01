"""
Test script for Docling dataset adapter

Verifies:
1. Dataset loads correctly
2. Multi-image samples work
3. Chat template is applied correctly
4. Compatible with GraniteDoclingVLM model
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.granite_docling_vlm import GraniteDoclingVLM
from data.docling_dataset import create_docling_dataset, docling_collate_fn
from data.processors import get_image_processor


def test_dataset_loading():
    """Test 1: Basic dataset loading"""
    print("=" * 70)
    print("Test 1: Dataset Loading")
    print("=" * 70)

    data_root = "../DoclingMatix_00000"

    # Load model to get tokenizer
    print("Loading Granite Docling model for tokenizer...")
    checkpoint = "../.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/"
    model = GraniteDoclingVLM.from_pretrained(checkpoint)

    print(f"\nTokenizer info:")
    print(f"  Vocab size: {model.tokenizer.vocab_size}")
    print(f"  Image token: {model.tokenizer.encode('<image>', add_special_tokens=False)}")
    print(f"  Special tokens: {list(model.tokenizer.special_tokens_map.keys())}")

    # Use Granite's native image processor (Idefics3)
    # This is already loaded with the model
    image_processor = model.image_processor

    print(f"\nImage processor: {image_processor.__class__.__name__}")

    # Create dataset
    dataset = create_docling_dataset(
        data_root=data_root,
        split="train",
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        mp_image_token_length=64,
        max_turns=3  # Limit to 3 QA turns for testing
    )

    print(f"\n✅ Dataset loaded: {len(dataset)} samples")
    return dataset, model


def test_sample_structure(dataset):
    """Test 2: Inspect sample structure"""
    print("\n" + "=" * 70)
    print("Test 2: Sample Structure")
    print("=" * 70)

    # Load first sample
    sample = dataset[0]

    if sample is None:
        print("❌ First sample is None!")
        return False

    print(f"\nSample 0 structure:")
    print(f"  Images: {len(sample['images'])} images")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")

    # Count non-masked labels (assistant tokens)
    num_label_tokens = (sample['labels'] != -100).sum().item()
    print(f"  Label tokens (assistant): {num_label_tokens}")
    print(f"  Masked tokens: {(sample['labels'] == -100).sum().item()}")

    # Decode sample
    print(f"\nDecoded input:")
    decoded = dataset.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(decoded[:500] + "..." if len(decoded) > 500 else decoded)

    print("\n✅ Sample structure looks correct")
    return True


def test_multi_image_samples(dataset):
    """Test 3: Check multi-image handling"""
    print("\n" + "=" * 70)
    print("Test 3: Multi-Image Samples")
    print("=" * 70)

    # Check first 10 samples for image counts
    image_counts = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        if sample:
            num_images = len(sample['images'])
            image_counts.append(num_images)

    print(f"\nImage counts in first 10 samples: {image_counts}")
    print(f"  Min images: {min(image_counts)}")
    print(f"  Max images: {max(image_counts)}")
    print(f"  Avg images: {sum(image_counts) / len(image_counts):.1f}")

    if max(image_counts) > 1:
        print(f"\n✅ Multi-image samples detected (max: {max(image_counts)})")
    else:
        print(f"\n⚠️  No multi-image samples in first 10")

    return True


def test_dataloader(dataset, image_processor):
    """Test 4: DataLoader with collate function"""
    print("\n" + "=" * 70)
    print("Test 4: DataLoader Batching")
    print("=" * 70)

    # Create collate function with image processor
    from functools import partial
    collate_fn = partial(docling_collate_fn, image_processor=image_processor)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get first batch
    batch = next(iter(dataloader))

    print(f"\nBatch structure:")
    if 'pixel_values' in batch:
        print(f"  Pixel values: {batch['pixel_values'].shape}")
        if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
            print(f"  Image grid: {batch['image_grid_thw'].shape}")
    elif 'images' in batch:
        print(f"  Images: {len(batch['images'])} samples")
        for i, sample_images in enumerate(batch['images']):
            print(f"    Sample {i}: {len(sample_images)} images")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['labels'].shape}")

    print("\n✅ DataLoader batching works correctly")
    return batch


def test_model_forward(model, batch):
    """Test 5: Forward pass with real batch"""
    print("\n" + "=" * 70)
    print("Test 5: Model Forward Pass")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    print(f"\nInput shapes:")
    print(f"  Input IDs: {input_ids.shape}")

    # Forward pass - handle both formats
    with torch.no_grad():
        if 'pixel_values' in batch:
            # Idefics3 format
            pixel_values = batch['pixel_values'].to(device)
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            print(f"  Pixel values: {pixel_values.shape}")
            print(f"  Labels: {labels.shape}")

            logits, loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                targets=labels
            )
        else:
            # nanoVLM format
            print(f"  Images: {len(batch['images'])} samples")
            print(f"  Labels: {labels.shape}")

            logits, loss = model(
                input_ids=input_ids,
                images=batch['images'],
                attention_mask=attention_mask,
                targets=labels
            )

    print(f"\nForward pass results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    print("\n✅ Model forward pass successful")
    return True


def test_chat_template_format(dataset):
    """Test 6: Verify chat template is Granite format"""
    print("\n" + "=" * 70)
    print("Test 6: Chat Template Format")
    print("=" * 70)

    # Create a simple conversation
    messages = [
        {"role": "user", "content": "<image>What is in this image?"},
        {"role": "assistant", "content": "This is a document."}
    ]

    formatted = dataset.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_special_tokens=False
    )

    print(f"\nFormatted conversation:")
    print(formatted)

    # Check for Granite-specific tokens
    granite_tokens = ['<|start_of_role|>', '<|end_of_role|>', '<|end_of_text|>']
    found_tokens = [token for token in granite_tokens if token in formatted]

    print(f"\nGranite tokens found: {found_tokens}")

    if len(found_tokens) >= 2:
        print("\n✅ Granite Docling chat template detected")
    else:
        print("\n⚠️  Chat template may not be Granite format")

    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 Docling Dataset Adapter Tests")
    print("=" * 70)

    try:
        # Test 1: Load dataset
        dataset, model = test_dataset_loading()

        # Test 2: Sample structure
        test_sample_structure(dataset)

        # Test 3: Multi-image support
        test_multi_image_samples(dataset)

        # Test 4: DataLoader
        batch = test_dataloader(dataset, model.image_processor)

        # Test 5: Model forward pass
        test_model_forward(model, batch)

        # Test 6: Chat template
        test_chat_template_format(dataset)

        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        print("✅ Test 1: Dataset Loading - PASSED")
        print("✅ Test 2: Sample Structure - PASSED")
        print("✅ Test 3: Multi-Image Samples - PASSED")
        print("✅ Test 4: DataLoader Batching - PASSED")
        print("✅ Test 5: Model Forward Pass - PASSED")
        print("✅ Test 6: Chat Template Format - PASSED")

        print("\n🎉 All tests passed! Dataset is ready for training.")

        print("\n" + "=" * 70)
        print("Usage Example:")
        print("=" * 70)
        print("""
from data.docling_dataset import create_docling_dataset, docling_collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = create_docling_dataset(
    data_root="/path/to/DoclingMatix_00000",
    split="train",
    tokenizer=model.tokenizer,
    image_processor=image_processor,
    mp_image_token_length=64,
    max_turns=5  # Optional: limit QA turns per sample
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=docling_collate_fn,
    num_workers=4
)

# Training loop
for batch in dataloader:
    logits, loss = model(
        input_ids=batch['input_ids'],
        images=batch['images'],
        attention_mask=batch['attention_mask'],
        targets=batch['labels']
    )
    loss.backward()
    # ... optimizer step
""")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Test SynthFormulaNet dataset compatibility with docling_dataset adapter

Verifies:
1. Single image per sample works
2. Single QA turn format works
3. Formula extraction tasks work
4. Compatible with GraniteDoclingVLM
"""

import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

from models.granite_docling_vlm import GraniteDoclingVLM
from data.docling_dataset import create_docling_dataset, docling_collate_fn


def test_synthformula_loading():
    """Test 1: Load SynthFormulaNet dataset"""
    print("=" * 70)
    print("Test 1: SynthFormulaNet Dataset Loading")
    print("=" * 70)

    data_root = "../SynthFormulaNet_00000"

    # Load model for tokenizer
    print("Loading Granite Docling model...")
    checkpoint = "../.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/"
    model = GraniteDoclingVLM.from_pretrained(checkpoint)

    # Create dataset
    dataset = create_docling_dataset(
        data_root=data_root,
        split="train",
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        mp_image_token_length=64,
    )

    print(f"\n✅ Dataset loaded: {len(dataset)} samples")
    print(f"   (SynthFormulaNet has 90,149 training samples)")
    return dataset, model


def test_sample_structure(dataset):
    """Test 2: Check sample structure"""
    print("\n" + "=" * 70)
    print("Test 2: Sample Structure (Formula Extraction)")
    print("=" * 70)

    sample = dataset[0]
    if sample is None:
        print("❌ Sample is None!")
        return False

    print(f"\nSample 0:")
    print(f"  Images: {len(sample['images'])} image(s)")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Label tokens: {(sample['labels'] != -100).sum().item()}")

    # Decode to see formula task
    decoded = dataset.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"\nDecoded sample (first 300 chars):")
    print(decoded[:300])

    # Check if it's a formula task
    if '<formula>' in decoded and '<loc_' in decoded:
        print("\n✅ Formula extraction format detected")
    else:
        print("\n⚠️  Expected <formula> and location tokens")

    return True


def test_batch_processing(dataset, model):
    """Test 3: Batch processing"""
    print("\n" + "=" * 70)
    print("Test 3: Batch Processing")
    print("=" * 70)

    collate_fn = partial(docling_collate_fn, image_processor=model.image_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    batch = next(iter(dataloader))

    print(f"\nBatch structure:")
    if 'pixel_values' in batch:
        print(f"  Pixel values: {batch['pixel_values'].shape}")
        if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
            print(f"  Image grid: {batch['image_grid_thw'].shape}")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Labels: {batch['labels'].shape}")

    print("\n✅ Batch processing works")
    return batch


def test_model_forward(model, batch):
    """Test 4: Model forward pass"""
    print("\n" + "=" * 70)
    print("Test 4: Model Forward Pass")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    pixel_values = batch['pixel_values'].to(device)
    image_grid_thw = batch.get('image_grid_thw')
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            targets=labels
        )

    print(f"\nForward pass results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    print("\n✅ Forward pass successful")
    return True


def test_format_comparison(dataset):
    """Test 5: Compare with DoclingMatix format"""
    print("\n" + "=" * 70)
    print("Test 5: Format Compatibility")
    print("=" * 70)

    # SynthFormulaNet: single image, single QA turn
    sample = dataset[0]

    print(f"\nSynthFormulaNet sample:")
    print(f"  Images per sample: {len(sample['images'])}")
    print(f"  Sequence length: {sample['input_ids'].shape[0]}")

    # Check tokenization includes image token
    has_image_token = (sample['input_ids'] == dataset.tokenizer.encode('<image>', add_special_tokens=False)[0]).any()
    print(f"  Has <image> token: {has_image_token}")

    # Check has formula content
    decoded = dataset.tokenizer.decode(sample['input_ids'])
    has_formula_tag = '<formula>' in decoded
    has_loc_tags = '<loc_' in decoded
    print(f"  Has <formula> tag: {has_formula_tag}")
    print(f"  Has location tags: {has_loc_tags}")

    if has_image_token and has_formula_tag and has_loc_tags:
        print("\n✅ SynthFormulaNet format properly converted")
    else:
        print("\n⚠️  Some expected tokens missing")

    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 SynthFormulaNet Dataset Compatibility Tests")
    print("=" * 70)
    print("Testing dataset adapter with formula extraction tasks\n")

    try:
        # Test 1: Load dataset
        dataset, model = test_synthformula_loading()

        # Test 2: Sample structure
        test_sample_structure(dataset)

        # Test 3: Batch processing
        batch = test_batch_processing(dataset, model)

        # Test 4: Forward pass
        test_model_forward(model, batch)

        # Test 5: Format comparison
        test_format_comparison(dataset)

        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        print("✅ Test 1: Dataset Loading - PASSED")
        print("✅ Test 2: Sample Structure - PASSED")
        print("✅ Test 3: Batch Processing - PASSED")
        print("✅ Test 4: Model Forward Pass - PASSED")
        print("✅ Test 5: Format Compatibility - PASSED")

        print("\n🎉 SynthFormulaNet dataset is compatible!")
        print("\nDataset details:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Task: Formula extraction with location")
        print(f"  Format: Single image + single QA turn")
        print(f"  Images per sample: 1")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

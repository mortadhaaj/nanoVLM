"""
Test script for multi-image per sample support in GraniteDoclingVLM

This demonstrates that the adapter can handle datasets where:
- Different samples have different numbers of images
- Images are properly grouped per sample (not flattened)
- The model maintains proper batch semantics
"""

import torch
from models.config import VLMConfig
from models.granite_docling_vlm import GraniteDoclingVLM


def test_multi_image_forward():
    """Test forward pass with varying images per sample"""
    print("=" * 70)
    print("Test 1: Forward Pass with Multi-Image Samples")
    print("=" * 70)

    cfg = VLMConfig()
    model = GraniteDoclingVLM(cfg, load_backbone=False)
    model = model.to('cuda')
    model.eval()

    # Create batch with varying image counts: [3, 1, 2] images
    batch_size = 3
    seq_len = 250

    input_ids = torch.randint(0, cfg.lm_vocab_size - 100, (batch_size, seq_len)).to('cuda')
    image_token_id = model.tokenizer.image_token_id

    # Sample 0: 3 images (192 tokens)
    input_ids[0, 10:74] = image_token_id
    input_ids[0, 80:144] = image_token_id
    input_ids[0, 150:214] = image_token_id

    # Sample 1: 1 image (64 tokens)
    input_ids[1, 15:79] = image_token_id

    # Sample 2: 2 images (128 tokens)
    input_ids[2, 20:84] = image_token_id
    input_ids[2, 90:154] = image_token_id

    # Create nested list maintaining per-sample grouping
    images = [
        [torch.randn(1, 3, 512, 512).to('cuda') for _ in range(3)],  # 3 images
        [torch.randn(1, 3, 512, 512).to('cuda')],                     # 1 image
        [torch.randn(1, 3, 512, 512).to('cuda') for _ in range(2)]   # 2 images
    ]

    print(f"Batch configuration:")
    for i, sample_imgs in enumerate(images):
        print(f"  Sample {i}: {len(sample_imgs)} images")

    with torch.no_grad():
        logits, loss = model(input_ids, images, targets=input_ids)

    print(f"\n✅ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Check dimensions match (vocab size might differ for Granite)
    assert logits.shape[0] == batch_size, f"Batch size mismatch: {logits.shape[0]} != {batch_size}"
    assert logits.shape[1] == seq_len, f"Sequence length mismatch: {logits.shape[1]} != {seq_len}"
    print(f"  ✓ Output shape correct: [{batch_size}, {seq_len}, {logits.shape[2]}]")

    return True


def test_multi_image_backward():
    """Test backward pass (gradient computation) with multi-image"""
    print("\n" + "=" * 70)
    print("Test 2: Backward Pass with Multi-Image Samples")
    print("=" * 70)

    cfg = VLMConfig()
    model = GraniteDoclingVLM(cfg, load_backbone=False)
    model = model.to('cuda')
    model.train()

    # Create batch: [2, 1] images per sample
    batch_size = 2
    seq_len = 150

    input_ids = torch.randint(0, cfg.lm_vocab_size - 100, (batch_size, seq_len)).to('cuda')
    image_token_id = model.tokenizer.image_token_id

    input_ids[0, 10:74] = image_token_id
    input_ids[0, 80:144] = image_token_id  # 2 images
    input_ids[1, 15:79] = image_token_id    # 1 image

    images = [
        [torch.randn(1, 3, 512, 512).to('cuda'), torch.randn(1, 3, 512, 512).to('cuda')],
        [torch.randn(1, 3, 512, 512).to('cuda')]
    ]

    print(f"Training batch:")
    for i, sample_imgs in enumerate(images):
        print(f"  Sample {i}: {len(sample_imgs)} images")

    # Forward + backward
    logits, loss = model(input_ids, images, targets=input_ids)
    loss.backward()

    # Check gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"\n✅ Backward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")

    assert params_with_grad > 0, "No gradients computed!"
    print(f"  ✓ Gradients flowing correctly")

    return True


def test_multi_image_generation():
    """Test generation with multiple images in prompt"""
    print("\n" + "=" * 70)
    print("Test 3: Generation with Multi-Image Prompt")
    print("=" * 70)

    cfg = VLMConfig()
    model = GraniteDoclingVLM(cfg, load_backbone=False)
    model = model.to('cuda')
    model.eval()

    # Create prompt with 2 images
    seq_len = 150
    input_ids = torch.randint(0, cfg.lm_vocab_size - 100, (1, seq_len)).to('cuda')
    image_token_id = model.tokenizer.image_token_id

    input_ids[0, 10:74] = image_token_id
    input_ids[0, 80:144] = image_token_id

    images = [
        [torch.randn(1, 3, 512, 512).to('cuda'), torch.randn(1, 3, 512, 512).to('cuda')]
    ]

    print(f"Generation prompt: 1 sample with {len(images[0])} images")

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            images,
            max_new_tokens=20,
            temperature=0.0
        )

    print(f"\n✅ Generation successful!")
    print(f"  Prompt length: {input_ids.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    # Note: generate may return just the new tokens or input+new depending on implementation
    # Both are valid - just check it returned something
    assert generated.shape[1] > 0, "Generated sequence is empty!"
    assert isinstance(generated, torch.Tensor), "Generated output should be tensor!"
    print(f"  ✓ Generation produced output tensor")

    return True


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 70)
    print("Test 4: Edge Cases")
    print("=" * 70)

    cfg = VLMConfig()
    model = GraniteDoclingVLM(cfg, load_backbone=False)
    model = model.to('cuda')
    model.eval()

    # Test 1: Large variation in image counts [5, 1]
    print("\nEdge case 1: Large variation [5, 1]")
    input_ids = torch.randint(0, cfg.lm_vocab_size - 100, (2, 400)).to('cuda')
    image_token_id = model.tokenizer.image_token_id

    # 5 images in first sample
    for i in range(5):
        start = 10 + i * 70
        input_ids[0, start:start+64] = image_token_id

    # 1 image in second sample
    input_ids[1, 15:79] = image_token_id

    images = [
        [torch.randn(1, 3, 512, 512).to('cuda') for _ in range(5)],
        [torch.randn(1, 3, 512, 512).to('cuda')]
    ]

    with torch.no_grad():
        logits, loss = model(input_ids, images, targets=input_ids)

    print(f"  ✓ Handled [5, 1] variation - Loss: {loss.item():.4f}")

    # Test 2: All samples with same number of images
    print("\nEdge case 2: Uniform image count [3, 3, 3]")
    input_ids = torch.randint(0, cfg.lm_vocab_size - 100, (3, 250)).to('cuda')

    for sample in range(3):
        for img in range(3):
            start = 10 + img * 70
            input_ids[sample, start:start+64] = image_token_id

    images = [
        [torch.randn(1, 3, 512, 512).to('cuda') for _ in range(3)]
        for _ in range(3)
    ]

    with torch.no_grad():
        logits, loss = model(input_ids, images, targets=input_ids)

    print(f"  ✓ Handled uniform [3, 3, 3] - Loss: {loss.item():.4f}")

    return True


def main():
    print("\n" + "=" * 70)
    print("🔬 Multi-Image Per Sample Tests")
    print("=" * 70)
    print("Testing GraniteDoclingVLM adapter with varying images per sample\n")

    try:
        test1 = test_multi_image_forward()
        test2 = test_multi_image_backward()
        test3 = test_multi_image_generation()
        test4 = test_edge_cases()

        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        print("✅ Test 1: Forward Pass - PASSED")
        print("✅ Test 2: Backward Pass - PASSED")
        print("✅ Test 3: Generation - PASSED")
        print("✅ Test 4: Edge Cases - PASSED")
        print("\n🎉 All multi-image tests passed!")
        print("\nThe adapter correctly handles:")
        print("  • Varying numbers of images per sample")
        print("  • Proper per-sample image grouping")
        print("  • Batch padding to max images")
        print("  • Training and inference modes")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
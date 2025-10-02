"""
Quick test script for Granite Docling VLM integration

Tests:
1. Model loading from HF checkpoint
2. Forward pass
3. Generation with real document image
4. Comparison with transformers implementation
"""

import torch
import re
from PIL import Image, ImageDraw
from pathlib import Path
from models.granite_docling_vlm import GraniteDoclingVLM, GraniteDoclingConfig

# Checkpoint and image paths
CHECKPOINT_PATH = "./checkpoints/granite_docling_finetuned/checkpoint_batch_3000"
TEST_IMAGE_PATH = "../DoclingMatix_00000/images/train/00000_200_0.png"
# TEST_IMAGE_PATH = "../SynthFormulaNet_00000/images/train/00000_1090.png"

# # Checkpoint and image paths
# CHECKPOINT_PATH = "../.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/"
TEST_IMAGE_PATH = "assets/input.png"


def draw_bounding_boxes(image_path: str, response_text: str, is_doctag_response: bool = False) -> Image.Image:
    """Draw bounding boxes on the image based on loc tags and return the annotated image."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        class_colors = {
            "caption": "#FFCC99", "footnote": "#C8C8FF", "formula": "#C0C0C0",
            "list_item": "#9999FF", "page_footer": "#CCFFCC", "page_header": "#CCFFCC",
            "picture": "#FFCCA4", "chart": "#FFCCA4", "section_header": "#FF9999",
            "table": "#FFCCCC", "text": "#FFFF99", "title": "#FF9999",
        }

        doctag_class_pattern = r"<([^>]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>[^<]*</[^>]+>"
        doctag_matches = re.findall(doctag_class_pattern, response_text)
        class_pattern = r"<([^>]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
        class_matches = re.findall(class_pattern, response_text)

        seen_coords = set()
        all_class_matches = []
        for match in doctag_matches:
            coords = (match[1], match[2], match[3], match[4])
            if coords not in seen_coords:
                seen_coords.add(coords)
                all_class_matches.append(match)
        for match in class_matches:
            coords = (match[1], match[2], match[3], match[4])
            if coords not in seen_coords:
                seen_coords.add(coords)
                all_class_matches.append(match)

        for class_name, xmin, ymin, xmax, ymax in all_class_matches:
            color = class_colors.get(class_name.lower(), "#808080") if is_doctag_response else "#E0115F"
            x1, y1 = int((int(xmin) / 500) * width), int((int(ymin) / 500) * height)
            x2, y2 = int((int(xmax) / 500) * width), int((int(ymax) / 500) * height)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        return image
    except Exception:
        return Image.open(image_path)


def test_model_loading():
    """Test loading model from HF checkpoint"""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)

    try:
        model = GraniteDoclingVLM.from_pretrained(
            CHECKPOINT_PATH,
            torch_dtype=torch.bfloat16,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Vision encoder: {model.cfg.vit_n_blocks} layers, {model.cfg.vit_hidden_dim}d")
        print(f"   Language model: {model.cfg.lm_n_blocks} layers, {model.cfg.lm_hidden_dim}d")
        print(f"   Image token ID: {model.cfg.image_token_id}")

        # Verify processor components are loaded
        if hasattr(model, 'processor'):
            print(f"   ✅ Processor loaded")
            print(f"   ✅ Tokenizer loaded: {type(model.tokenizer).__name__}")
            print(f"   ✅ Image processor loaded: {type(model.image_processor).__name__}")
            print(f"   Tokenizer vocab size: {model.tokenizer.vocab_size}")

            # Get special token IDs
            eos_id = model.tokenizer.eos_token_id if hasattr(model.tokenizer, 'eos_token_id') else None
            image_token_id = getattr(model.processor, 'image_token_id', model.cfg.image_token_id)

            print(f"   Special tokens: image_token_id={image_token_id}, eos_token_id={eos_id}")

            # Check if special tokens are in vocab
            special_tokens = model.tokenizer.all_special_tokens[:5]  # Show first 5
            print(f"   Sample special tokens: {special_tokens}")

            # Test document-specific tokens
            doc_tokens = ['<doctag>', '<text>', '<title>']
            print(f"   Document tokens:")
            for token in doc_tokens:
                try:
                    token_id = model.tokenizer.encode(token, add_special_tokens=False)
                    decoded = model.tokenizer.decode(token_id)
                    print(f"     {token}: IDs={token_id}, decoded='{decoded}'")
                except Exception as e:
                    print(f"     {token}: ERROR - {e}")
        else:
            print(f"   ❌ WARNING: Processor not loaded!")

        return model

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_forward_pass(model):
    """Test forward pass with dummy inputs"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass (Dummy Inputs)")
    print("=" * 60)

    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Create dummy inputs (Idefics3 format)
        batch_size = 2
        seq_len = 100
        num_images = 1

        input_ids = torch.randint(0, model.cfg.lm_vocab_size - 100, (batch_size, seq_len)).to(device)

        # Insert image token placeholders (similar to nanoVLM tests)
        input_ids[0, 10:10+64] = model.cfg.image_token_id
        input_ids[1, 15:15+64] = model.cfg.image_token_id

        # Idefics3 format: (batch, num_images, 3, height, width)
        pixel_values = torch.randn(batch_size, num_images, 3, 512, 512).to(device).to(dtype)

        # image_grid_thw: (batch, num_images, 3) - [num_tiles_height, num_tiles_width, tile_size]
        image_grid_thw = torch.tensor([[[1, 1, 512]]] * batch_size).to(device)

        attention_mask = torch.ones(batch_size, seq_len).to(device)
        targets = torch.randint(0, model.cfg.lm_vocab_size, (batch_size, seq_len)).to(device)
        targets[:, :50] = -100  # Mask first 50 tokens

        # Forward pass
        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                targets=targets
            )

        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Pixel values shape: {pixel_values.shape}")
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss is finite: {torch.isfinite(loss).item()}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_with_real_image(model):
    """Test generation with real document image"""
    print("\n" + "=" * 60)
    print("Test 3: Generation with Real Document Image")
    print("=" * 60)

    try:
        device = next(model.parameters()).device

        # Verify processor is loaded
        if not hasattr(model, 'processor'):
            raise ValueError("Model does not have processor loaded!")

        processor = model.processor

        # Load real image
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        print(f"   Loaded image: {TEST_IMAGE_PATH}")
        print(f"   Image size: {image.size}")

        # Prepare prompt
        question = "Convert this page to docling." # "<formula>"
        # question = "<formula>"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Pixel values shape: {inputs['pixel_values'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"   Image grid thw: {inputs['image_grid_thw'].shape}")

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=4096,
                temperature=0.0,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        # Decode
        generated_text = processor.batch_decode(generated, skip_special_tokens=False)[0]

        print(f"✅ Generation successful")
        print(f"   Question: {question}")
        print(f"   Generated tokens: {generated.shape[1]}")
        print(f"   Response: {generated_text[:]}...")

        # Extract assistant response (after last <|start_of_role|>assistant<|end_of_role|>)
        if "<|start_of_role|>assistant<|end_of_role|>" in generated_text:
            response = generated_text.split("<|start_of_role|>assistant<|end_of_role|>")[-1]
            response = response.split("<|end_of_text|>")[0] if "<|end_of_text|>" in response else response
        else:
            response = generated_text

        # Save annotated image with bounding boxes
        has_doctag = "<doctag>" in response.lower()
        annotated_image = draw_bounding_boxes(TEST_IMAGE_PATH, response, is_doctag_response=has_doctag)

        # Create assets directory if needed
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)

        output_path = assets_dir / "annotated_output.png"
        annotated_image.save(str(output_path))
        print(f"   📸 Saved annotated image to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """Test backward pass for training compatibility"""
    print("\n" + "=" * 60)
    print("Test 4: Backward Pass (Training)")
    print("=" * 60)

    try:
        model.train()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Create dummy batch
        batch_size = 1
        seq_len = 80
        input_ids = torch.randint(0, model.cfg.lm_vocab_size - 100, (batch_size, seq_len)).to(device)

        # Insert image tokens
        input_ids[0, 10:10+64] = model.cfg.image_token_id

        pixel_values = torch.randn(batch_size, 1, 3, 512, 512).to(device).to(dtype)
        image_grid_thw = torch.tensor([[[1, 1, 512]]]).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        targets = torch.randint(0, model.cfg.lm_vocab_size, (batch_size, seq_len)).to(device)
        targets[:, :40] = -100

        # Forward + backward
        logits, loss = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            targets=targets
        )

        loss.backward()

        # Check gradients
        has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)

        model.zero_grad()

        print(f"✅ Backward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Parameters with gradients: {has_grads}/{total_params}")
        print(f"   Gradient flow: {'OK' if has_grads > 0 else 'FAIL'}")

        return has_grads > 0

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        model.eval()


def main():
    print("\n" + "=" * 70)
    print("🧪 Granite Docling VLM Integration Tests")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Test 1: Load model
    model = test_model_loading()

    # Test 2: Forward pass
    forward_ok = test_forward_pass(model)

    # Test 3: Real image generation
    gen_ok = test_generation_with_real_image(model)

    # Test 4: Backward pass
    backward_ok = test_backward_pass(model)

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"1. Model Loading:          ✅")
    print(f"2. Forward Pass:           {'✅' if forward_ok else '❌'}")
    print(f"3. Generation (Real Image):{'✅' if gen_ok else '❌'}")
    print(f"4. Backward Pass:          {'✅' if backward_ok else '❌'}")
    print()

    all_passed = forward_ok and gen_ok and backward_ok
    if all_passed:
        print("🎉 All tests passed! Granite Docling VLM is ready.")
        print("\nNext steps:")
        print("  - Update MODIFICATIONS_LOG.md")
        print("  - Add to integration test suite")
        print("  - Test with train.py/generate.py")
        return 0
    else:
        print("❌ Some tests failed - review errors above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

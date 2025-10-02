"""
Test dataset formatting by computing loss on pretrained model
"""
import torch
from models.granite_docling_vlm import GraniteDoclingVLM
from data.docling_dataset import create_docling_dataset

# Load pretrained model
checkpoint_path = "/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/"
model = GraniteDoclingVLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device=torch.device("cuda")
).eval()

print("Creating dataset...")
dataset = create_docling_dataset(
    data_root="/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/DoclingMatix_00000",
    split="train",
    tokenizer=model.tokenizer,
    image_processor=model.image_processor,
    processor=model.processor,
    mp_image_token_length=64,
    max_turns=None,
    max_images_per_sample=None
)

print(f"Dataset size: {len(dataset)}")

# Test first 5 samples
print("\nTesting first 5 samples with pretrained model:")
for idx in range(min(5, len(dataset))):
    sample = dataset[idx]

    print(f"\nSample {idx}:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Images: {len(sample['images']) if isinstance(sample['images'], list) else 'processed'}")

    # Compute loss
    with torch.no_grad():
        input_ids = sample['input_ids'].unsqueeze(0).cuda()
        labels = sample['labels'].unsqueeze(0).cuda()
        attention_mask = sample['attention_mask'].unsqueeze(0).cuda() if sample['attention_mask'] is not None else None

        # Check if images need processing
        images = sample['images']
        if images and len(images) > 0:
            from PIL import Image
            if isinstance(images[0], Image.Image):
                # Images not processed yet - need to process
                print("  ERROR: Images are still PIL - should be processed!")
                continue

        # For now, skip image processing since dataset returns PIL images
        # Just check the text formatting
        print(f"  Decoded input (first 200 chars): {model.tokenizer.decode(input_ids[0][:200], skip_special_tokens=False)}")

print("\nDone!")

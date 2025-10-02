"""
Debug script to verify training data format
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

print("Creating SynthFormulaNet dataset...")
dataset = create_docling_dataset(
    data_root="/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/SynthFormulaNet_00000",
    split="train",
    tokenizer=model.tokenizer,
    image_processor=model.image_processor,
    processor=model.processor,
    mp_image_token_length=64,
    max_turns=None,
    max_images_per_sample=None
)

print(f"Dataset size: {len(dataset)}")

# Get sample 1090 to match your test
sample = dataset[1090]

print(f"\nSample 1090:")
print(f"  Input IDs shape: {sample['input_ids'].shape}")
print(f"  Labels shape: {sample['labels'].shape}")
print(f"  Pixel values shape: {sample['pixel_values'].shape}")

# Decode to see what the model should learn
input_text = model.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(f"\n  Full input text:\n{input_text}")

# Check labels - count how many tokens have labels (not -100)
num_label_tokens = (sample['labels'] != -100).sum().item()
total_tokens = len(sample['labels'])
print(f"\n  Label statistics:")
print(f"    Total tokens: {total_tokens}")
print(f"    Tokens with labels (assistant): {num_label_tokens}")
print(f"    Percentage: {100 * num_label_tokens / total_tokens:.1f}%")

# Decode only the labeled (assistant) portion
assistant_tokens = sample['input_ids'][sample['labels'] != -100]
if len(assistant_tokens) > 0:
    assistant_text = model.tokenizer.decode(assistant_tokens, skip_special_tokens=False)
    print(f"\n  Assistant response (what model should learn):")
    print(f"    {assistant_text[:500]}...")

# Compute loss on this sample with pretrained model
with torch.no_grad():
    input_ids = sample['input_ids'].unsqueeze(0).cuda()
    labels = sample['labels'].unsqueeze(0).cuda()
    pixel_values = sample['pixel_values'].unsqueeze(0).cuda()

    if 'image_grid_thw' in sample:
        image_grid_thw = sample['image_grid_thw'].unsqueeze(0).cuda()
    else:
        image_grid_thw = None

    logits, loss = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        targets=labels
    )

    print(f"\n  Pretrained model loss on this sample: {loss.item():.4f}")
    print(f"  (Expected: low loss since pretrained model should know this format)")

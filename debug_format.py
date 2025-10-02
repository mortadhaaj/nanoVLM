"""
Debug script to compare training vs inference data formatting
"""
import torch
from models.granite_docling_vlm import GraniteDoclingVLM
from data.docling_dataset import DoclingDataset
from PIL import Image

# Load model
checkpoint_path = "/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/.cache/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe/"
model = GraniteDoclingVLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device=torch.device("cuda")
)

print("=" * 80)
print("TRAINING FORMAT (from DoclingDataset)")
print("=" * 80)

# Create dataset sample
dataset = DoclingDataset(
    data_root="/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/DoclingMatix_00000",
    split="train",
    tokenizer=model.tokenizer,
    image_processor=model.image_processor,
    processor=model.processor,
    mp_image_token_length=64,
    max_turns=None,
    max_images_per_sample=None
)

# Find the sample with image 00000_133_0.png to match inference
target_file = None
for idx in range(len(dataset)):
    ann_file = dataset.ann_files[idx]
    if "00000_133" in str(ann_file):
        target_file = idx
        break

if target_file is None:
    print("WARNING: Could not find 00000_133 sample, using first sample instead")
    target_file = 0

print(f"Using sample index: {target_file}, file: {dataset.ann_files[target_file]}")

# Get sample
sample = dataset[target_file]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
print(f"\nDecoded input (first 2000 chars):")
decoded_input = model.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(decoded_input[:2000])
print("\n...")

print("\n" + "=" * 80)
print("INFERENCE FORMAT (from test_granite_docling.py)")
print("=" * 80)

# Load same image
image = Image.open("/eph/nvme0/azureml/cr/j/c95496a9776c4474b5d3d85e5fb3d9c2/exe/wd/outputs/granite_docling/DoclingMatix_00000/images/train/00000_133_0.png").convert("RGB")
question = "Convert this page to docling."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }
]

prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = model.processor(text=prompt, images=[image], return_tensors="pt")

print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"\nDecoded input (first 2000 chars):")
decoded_inference = model.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
print(decoded_inference[:2000])
print("\n...")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Training input length: {sample['input_ids'].shape[0]}")
print(f"Inference input length: {inputs['input_ids'].shape[1]}")
print(f"\nAre they identical? {decoded_input == decoded_inference}")

# Find first difference
if decoded_input != decoded_inference:
    min_len = min(len(decoded_input), len(decoded_inference))
    for i in range(min_len):
        if decoded_input[i] != decoded_inference[i]:
            print(f"\nFirst difference at position {i}:")
            print(f"Training:  ...{decoded_input[max(0,i-50):i+50]}...")
            print(f"Inference: ...{decoded_inference[max(0,i-50):i+50]}...")
            break

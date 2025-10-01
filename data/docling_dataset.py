"""
Docling Dataset Adapter for nanoVLM

This adapter converts Docling annotation format to nanoVLM's VQADataset format.

Dataset Format:
- Annotations: JSON files in anns/train/*.json
- Each JSON has:
  - "img_pth": list of image paths (multiple images per sample)
  - "anns": list of QA pairs with "user", "assistant", "source" fields

This adapter:
1. Loads Docling annotations
2. Converts to nanoVLM format with multi-image support
3. Uses Granite Docling chat template
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset


class DoclingDataset(Dataset):
    """
    Dataset for Docling annotations in nanoVLM format.

    Supports:
    - Multiple images per sample
    - Multiple QA turns per sample
    - Granite Docling chat template
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        tokenizer=None,
        image_processor=None,
        mp_image_token_length: int = 64,
        max_turns: int = None,  # Limit number of QA turns per sample
        max_images_per_sample: int = None,  # Limit images per sample to save memory
    ):
        """
        Args:
            data_root: Path to DoclingMatix_00000 directory
            split: "train" or "val"
            tokenizer: Tokenizer with Granite chat template
            image_processor: Image processor for vision encoder
            mp_image_token_length: Number of tokens per image (64 for nanoVLM)
            max_turns: Maximum QA turns to include (None = all)
            max_images_per_sample: Maximum images per sample (None = all, use 1-2 to save memory)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_turns = max_turns
        self.max_images_per_sample = max_images_per_sample

        # Load annotation files
        self.ann_dir = self.data_root / "anns" / split
        self.img_dir = self.data_root / "images" / split

        if not self.ann_dir.exists():
            raise ValueError(f"Annotation directory not found: {self.ann_dir}")
        if not self.img_dir.exists():
            raise ValueError(f"Image directory not found: {self.img_dir}")

        # Get all annotation files
        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        print(f"Found {len(self.ann_files)} annotation files in {self.ann_dir}")

        # Calculate prefix length for loss masking
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.ann_files)

    def _get_prefix_len(self):
        """Calculate prefix length before assistant content for loss masking"""
        if self.tokenizer is None:
            return 0

        random_string = "xzyvd"
        templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string}],
            tokenize=False,
            add_special_tokens=False
        )
        random_string_location = templated.find(random_string)
        return len(self.tokenizer.encode(templated[:random_string_location]))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with:
                - images: List of PIL Images or processed tensors
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Labels for loss (-100 for non-assistant tokens)
        """
        ann_file = self.ann_files[idx]

        # Load annotation
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Normalize img_pth to list format
        img_paths = data['img_pth']
        if isinstance(img_paths, str):
            # Single image (SynthFormulaNet format)
            img_paths = [img_paths]

        # Limit images per sample if specified (to save memory)
        if self.max_images_per_sample is not None and len(img_paths) > self.max_images_per_sample:
            img_paths = img_paths[:self.max_images_per_sample]

        # Process images
        images = self._load_images(img_paths)

        # Normalize anns to list format
        anns = data['anns']
        if isinstance(anns, dict):
            # Single QA turn (SynthFormulaNet format)
            anns = [anns]

        # Process annotations into messages
        messages = self._create_messages(anns, num_images=len(images))

        if len(messages) == 0:
            return None

        # Prepare inputs and labels
        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        # For Granite Docling: Keep images as PIL (processor will be called in collator)
        # For nanoVLM: Process images here
        if self.image_processor and self.image_processor.__class__.__name__ != 'Idefics3ImageProcessor':
            # nanoVLM-style processing
            images = self._process_images(images) if images else []

        return {
            "images": images,  # PIL images for Granite, processed tensors for nanoVLM
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _load_images(self, img_paths: List[str]) -> List[Image.Image]:
        """Load images from paths"""
        images = []
        for img_path in img_paths:
            # img_path is relative to data_root (e.g., "images/train/00000_0_0.png")
            full_path = self.data_root / img_path

            if not full_path.exists():
                print(f"Warning: Image not found: {full_path}")
                continue

            try:
                img = Image.open(full_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                continue

        return images

    def _create_messages(self, anns: List[Dict], num_images: int) -> List[Dict]:
        """
        Convert Docling annotations to chat messages.

        For Granite Docling, images are inserted at the beginning of the first user message.

        Args:
            anns: List of {user, assistant, source} dicts
            num_images: Number of images in this sample

        Returns:
            List of message dicts for chat template
        """
        messages = []

        # Limit number of turns if specified
        turns = anns[:self.max_turns] if self.max_turns else anns
        turns = anns[-1:] if self.max_turns else anns

        for idx, turn in enumerate(turns):
            user_content = turn['user']
            assistant_content = turn['assistant']

            # For first turn, prepend image tokens
            if idx == 0 and num_images > 0:
                # Add <image> tokens for each image
                image_tokens = "<image>" * num_images
                user_content = image_tokens + user_content

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        return messages

    def _prepare_inputs_and_loss_mask(self, messages: List[Dict]) -> tuple:
        """
        Apply chat template and create loss mask.

        Returns:
            (input_ids, mask, attention_mask)
            - mask: True for assistant tokens (compute loss), False for others
        """
        # Apply chat template
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )

        # Create loss mask (1 for assistant tokens only)
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)

            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"])
        )

    def _get_labels(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Create labels for causal LM (shift by 1, mask non-assistant tokens)"""
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift for causal LM
        labels[-1] = -100  # Last token has no target
        return labels

    def _process_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Process images with image processor.

        For Granite Docling: Use the model's native image_processor (Idefics3)
        For nanoVLM: Use the custom split image processor

        Returns: List of processed image tensors (one per original image)
        """
        if self.image_processor is None:
            return images

        # Check if this is Granite's Idefics3ImageProcessor or nanoVLM's custom processor
        processor_name = self.image_processor.__class__.__name__

        if processor_name == 'Idefics3ImageProcessor':
            # Granite Docling processor - process each image individually
            # Returns dict with pixel_values: [1, num_patches, C, H, W]
            processed = []
            for img in images:
                proc_output = self.image_processor(
                    images=[img],
                    return_tensors="pt"
                )
                # Extract pixel_values and remove batch dimension: [num_patches, C, H, W]
                pixel_values = proc_output['pixel_values'][0]
                processed.append(pixel_values)

            return processed

        else:
            # nanoVLM custom processor - returns (patches_list, split_counts)
            # Each image becomes multiple patches
            processed = []
            for img in images:
                proc_img, _ = self.image_processor(img)
                # proc_img is a list of patch tensors
                processed.append(proc_img)

            return processed


def create_docling_dataset(
    data_root: str,
    split: str,
    tokenizer,
    image_processor,
    mp_image_token_length: int = 64,
    **kwargs
) -> DoclingDataset:
    """
    Factory function to create Docling dataset.

    Usage:
        from data.docling_dataset import create_docling_dataset

        dataset = create_docling_dataset(
            data_root="/path/to/DoclingMatix_00000",
            split="train",
            tokenizer=model.tokenizer,
            image_processor=get_image_processor(...),
            mp_image_token_length=64,
            max_turns=5  # Optional: limit QA turns
        )
    """
    return DoclingDataset(
        data_root=data_root,
        split=split,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=mp_image_token_length,
        **kwargs
    )


# Example collator for DataLoader
def docling_collate_fn(batch, image_processor=None):
    """
    Collate function for Docling dataset with variable-length sequences.

    Handles:
    - Variable number of images per sample
    - Variable sequence lengths
    - Padding
    - PIL image processing (for Granite) or pre-processed tensors (for nanoVLM)

    Args:
        batch: List of samples from dataset
        image_processor: Optional image processor for PIL images (Granite Docling)
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Get max sequence length
    max_seq_len = max(item['input_ids'].shape[0] for item in batch)

    # Prepare batched tensors
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_seq_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)

    # Collect images (nested list format for multi-image support)
    images = []

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']

        sample_images = item['images']

        # Check if images are PIL or already processed
        if sample_images and len(sample_images) > 0:
            from PIL import Image
            if isinstance(sample_images[0], Image.Image):
                # PIL images - keep as PIL for now (will process all at batch level)
                images.append(sample_images)
            else:
                # Already processed tensors (nanoVLM)
                images.append(sample_images)
        else:
            images.append([])

    # For Granite Docling with PIL images: process entire batch at once
    if images and len(images) > 0:
        from PIL import Image
        if isinstance(images[0][0] if images[0] else None, Image.Image):
            if image_processor is None:
                raise ValueError("image_processor required for PIL images")

            # Flatten all PIL images to process as batch
            all_pil_images = []
            image_sample_mapping = []  # Track which sample each image belongs to
            for sample_idx, sample_imgs in enumerate(images):
                for img in sample_imgs:
                    all_pil_images.append(img)
                    image_sample_mapping.append(sample_idx)

            if len(all_pil_images) > 0:
                # Process all images at once using Idefics3ImageProcessor
                proc_output = image_processor(images=all_pil_images, return_tensors="pt")

                # For Idefics3, return processor outputs directly
                return {
                    'pixel_values': proc_output['pixel_values'],  # [total_images, num_patches, C, H, W]
                    'image_grid_thw': proc_output.get('image_grid_thw', None),
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }

    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

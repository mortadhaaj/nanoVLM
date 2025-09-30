"""
Granite Docling VLM - nanoVLM-compatible wrapper for IBM Granite Docling

This module provides a nanoVLM-compatible interface to the IBM Granite Docling 258M model,
which uses Idefics3 architecture (Vision + LLaMA connector).

Architecture:
- Vision: Idefics3 vision encoder (768d, 12 layers, patch_size=16, img_size=512)
- Text: LLaMA (576d hidden, 30 layers, 9 attention heads, 3 KV heads)
- Connector: Idefics3 connector with scale_factor=4
- Total params: ~258M

Original checkpoint: ibm-granite/granite-docling-258M

Compatibility:
- Provides nanoVLM-compatible forward() and generate() signatures
- Accepts VLMConfig and converts it to GraniteDoclingConfig
- Handles images as list (nanoVLM format) and converts to pixel_values + image_grid_thw
"""

import os
import json
import glob
import tempfile
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file

# Transformers components for Idefics3 + LLaMA
from transformers.models.idefics3.configuration_idefics3 import Idefics3Config
from transformers.models.idefics3.modeling_idefics3 import Idefics3Model
from transformers import AutoProcessor

# nanoVLM config for compatibility
from models.config import VLMConfig


@dataclass
class GraniteDoclingConfig:
    """
    Configuration for Granite Docling VLM that mirrors nanoVLM's VLMConfig structure
    but contains Idefics3-specific parameters.
    """
    # Vision encoder config (Idefics3 vision)
    vit_hidden_dim: int = 768
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_n_blocks: int = 12
    vit_inter_dim: int = 3072
    vit_dropout: float = 0.0
    vit_ln_eps: float = 1e-6

    # Language model config (LLaMA)
    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_vocab_size: int = 100352
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_n_blocks: int = 30
    lm_max_position_embeddings: int = 8192
    lm_rms_eps: float = 1e-5
    lm_rope_theta: float = 100000.0
    lm_tie_weights: bool = True

    # Connector config
    scale_factor: int = 4

    # Special tokens
    image_token_id: int = 100270
    bos_token_id: int = 100264
    eos_token_id: int = 100257
    pad_token_id: int = 100257

    # Model type
    model_type: str = "idefics3"
    architecture: str = "GraniteDoclingVLM"

    # HF checkpoint path
    hf_checkpoint: str = "ibm-granite/granite-docling-258M"

    # nanoVLM compatibility fields
    mp_image_token_length: int = 64  # For compatibility with nanoVLM tests

    @classmethod
    def from_vlm_config(cls, vlm_cfg: VLMConfig, hf_checkpoint: str = None):
        """
        Create GraniteDoclingConfig from nanoVLM's VLMConfig.
        This allows using Granite Docling with existing nanoVLM infrastructure.
        """
        if hf_checkpoint is None:
            hf_checkpoint = "ibm-granite/granite-docling-258M"

        # Use default Granite values but allow override from checkpoint
        return cls(hf_checkpoint=hf_checkpoint)

    def to_idefics3_config(self) -> Idefics3Config:
        """Convert to native Idefics3Config for loading transformers components"""
        # This creates the config that transformers expects
        return Idefics3Config(
            vocab_size=self.lm_vocab_size,
            image_token_id=self.image_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            scale_factor=self.scale_factor,
            tie_word_embeddings=self.lm_tie_weights,
            # Vision config
            vision_config={
                "hidden_size": self.vit_hidden_dim,
                "intermediate_size": self.vit_inter_dim,
                "num_hidden_layers": self.vit_n_blocks,
                "num_attention_heads": self.vit_n_heads,
                "patch_size": self.vit_patch_size,
                "image_size": self.vit_img_size,
                "num_channels": 3,
                "layer_norm_eps": self.vit_ln_eps,
                "attention_dropout": self.vit_dropout,
            },
            # Text config (LLaMA)
            text_config={
                "hidden_size": self.lm_hidden_dim,
                "intermediate_size": self.lm_inter_dim,
                "num_hidden_layers": self.lm_n_blocks,
                "num_attention_heads": self.lm_n_heads,
                "num_key_value_heads": self.lm_n_kv_heads,
                "vocab_size": self.lm_vocab_size,
                "max_position_embeddings": self.lm_max_position_embeddings,
                "rms_norm_eps": self.lm_rms_eps,
                "rope_theta": self.lm_rope_theta,
                "tie_word_embeddings": self.lm_tie_weights,
            }
        )


class GraniteDoclingVLM(nn.Module):
    """
    Granite Docling VLM - wrapper around Idefics3 for nanoVLM compatibility.

    This provides a nanoVLM-style interface to the IBM Granite Docling model while
    using native Idefics3 components from transformers.

    Key features:
    - Accepts both GraniteDoclingConfig and VLMConfig
    - Forward method accepts images as list (nanoVLM format) or pixel_values (Idefics3 format)
    - Generate method compatible with nanoVLM signature
    - Supports both training and inference modes
    """

    def __init__(self, cfg: Union[GraniteDoclingConfig, VLMConfig], load_backbone=True):
        super().__init__()

        # Convert VLMConfig to GraniteDoclingConfig if needed
        if isinstance(cfg, VLMConfig):
            self.cfg = GraniteDoclingConfig.from_vlm_config(cfg)
        else:
            self.cfg = cfg

        # Convert to Idefics3Config and create the model
        idefics3_config = self.cfg.to_idefics3_config()

        if load_backbone:
            # Load from HF checkpoint
            print(f"Loading Granite Docling from: {self.cfg.hf_checkpoint}")
            self.model = Idefics3Model.from_pretrained(
                self.cfg.hf_checkpoint,
                config=idefics3_config
            )

            # Also load tokenizer for compatibility with integration tests
            print(f"Loading processor for tokenizer compatibility...")
            processor = AutoProcessor.from_pretrained(self.cfg.hf_checkpoint, trust_remote_code=True)
            self.processor = processor
            self.tokenizer = processor.tokenizer
            self.image_processor = processor.image_processor
        else:
            # Initialize from scratch
            self.model = Idefics3Model(idefics3_config)
            # Create a minimal tokenizer mock for compatibility
            # (real tokenizer would need to be loaded from HF)
            class MockTokenizer:
                def __init__(self, image_token_id, eos_token_id, vocab_size):
                    self.image_token_id = image_token_id
                    self.eos_token_id = eos_token_id
                    self.vocab_size = vocab_size

            self.tokenizer = MockTokenizer(
                self.cfg.image_token_id,
                self.cfg.eos_token_id,
                self.cfg.lm_vocab_size
            )
            self.processor = None
            self.image_processor = None

        # LM head (for generation)
        self.lm_head = nn.Linear(self.cfg.lm_hidden_dim, self.cfg.lm_vocab_size, bias=False)

        # Tie weights (standard in Idefics3)
        self.tie_weights()

    def tie_weights(self):
        """Tie lm_head to text model embeddings"""
        self.lm_head.weight = self.model.text_model.embed_tokens.weight

    # nanoVLM compatibility properties
    @property
    def vision_encoder(self):
        """Compatibility property - returns vision model"""
        return self.model.vision_model

    @property
    def decoder(self):
        """Compatibility property - returns text model"""
        return self.model.text_model

    @property
    def MP(self):
        """Compatibility property - returns connector"""
        return self.model.connector

    def _process_images_to_pixel_values(self, images, device):
        """
        Convert nanoVLM images format (list of tensors or single tensor) to Idefics3 format.

        Supports multiple images per sample while maintaining per-sample grouping.

        Args:
            images: One of:
                   - Nested list: [[img1, img2], [img3]] - multiple images per sample
                   - Flat list: [img1, img2] - one image per sample
                   - Single tensor: [B, C, H, W] - one image per sample

        Returns:
            pixel_values: [B, max_num_images, C, H, W] (padded to max images in batch)
            image_grid_thw: [B, max_num_images, 3] where each is [tiles_h, tiles_w, patch_size]
        """
        if images is None:
            return None, None

        # Handle tensor input (from generate.py / test)
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # [B, C, H, W]
                # Add num_images dimension: [B, 1, C, H, W]
                pixel_values = images.unsqueeze(1).to(device)
                batch_size = images.shape[0]
                img_size = images.shape[-1]
                # Create grid for each image in batch
                image_grid_thw = torch.tensor(
                    [[[1, 1, img_size]]] * batch_size,
                    device=device
                )  # [B, 1, 3]
                return pixel_values, image_grid_thw
            elif images.dim() == 5:  # Already [B, num_images, C, H, W]
                return images.to(device), None
            else:
                raise ValueError(f"Unexpected tensor dimensions: {images.shape}")

        # Handle list input (from train.py collator)
        if not isinstance(images, list):
            raise ValueError(f"Unexpected images type: {type(images)}")

        if not images:
            return None, None

        # Determine if we have nested list (multiple images per sample)
        is_nested = isinstance(images[0], list)

        if is_nested:
            # Each element is a list of images for that sample
            # images = [[img1_s0, img2_s0], [img1_s1], [img1_s2, img2_s2, img3_s2]]
            batch_size = len(images)
            max_images_per_sample = max(len(sample_images) for sample_images in images)

            # Get image dimensions from first image
            first_img = images[0][0]
            if first_img.dim() == 3:  # [C, H, W]
                C, H, W = first_img.shape
            elif first_img.dim() == 4:  # [1, C, H, W]
                _, C, H, W = first_img.shape
            else:
                raise ValueError(f"Unexpected image tensor shape: {first_img.shape}")

            # Initialize output tensors
            pixel_values = torch.zeros(
                batch_size, max_images_per_sample, C, H, W,
                dtype=first_img.dtype, device=device
            )
            image_grid_thw = torch.zeros(
                batch_size, max_images_per_sample, 3,
                dtype=torch.long, device=device
            )

            # Fill in images for each sample
            for sample_idx, sample_images in enumerate(images):
                for img_idx, img_tensor in enumerate(sample_images):
                    # Handle both [C, H, W] and [1, C, H, W]
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

                    # Extract single image: [C, H, W]
                    img_tensor = img_tensor.squeeze(0) if img_tensor.shape[0] == 1 else img_tensor

                    pixel_values[sample_idx, img_idx] = img_tensor.to(device)
                    image_grid_thw[sample_idx, img_idx] = torch.tensor(
                        [1, 1, img_tensor.shape[-1]], dtype=torch.long
                    )

            return pixel_values, image_grid_thw

        else:
            # Flat list: treat each image as a separate sample with 1 image
            # images = [img1, img2, img3]
            batch_size = len(images)

            # Get image dimensions
            first_img = images[0]
            if first_img.dim() == 3:  # [C, H, W]
                C, H, W = first_img.shape
            elif first_img.dim() == 4:  # [1, C, H, W]
                _, C, H, W = first_img.shape
            else:
                raise ValueError(f"Unexpected image tensor shape: {first_img.shape}")

            # Initialize output: [B, 1, C, H, W]
            pixel_values = torch.zeros(
                batch_size, 1, C, H, W,
                dtype=first_img.dtype, device=device
            )
            image_grid_thw = torch.zeros(
                batch_size, 1, 3,
                dtype=torch.long, device=device
            )

            for idx, img_tensor in enumerate(images):
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

                img_tensor = img_tensor.squeeze(0) if img_tensor.shape[0] == 1 else img_tensor
                pixel_values[idx, 0] = img_tensor.to(device)
                image_grid_thw[idx, 0] = torch.tensor(
                    [1, 1, img_tensor.shape[-1]], dtype=torch.long
                )

            return pixel_values, image_grid_thw

    def prepare_inputs(self, text: str, images: List, **kwargs):
        """
        Prepare inputs using the processor (convenience method).

        Args:
            text: Input text (prompt)
            images: List of PIL images
            **kwargs: Additional processor arguments

        Returns:
            dict with input_ids, pixel_values, image_grid_thw, attention_mask
        """
        if not hasattr(self, 'processor'):
            raise ValueError(
                "Model does not have a processor. "
                "Load model with from_pretrained() to auto-load the processor."
            )

        return self.processor(text=text, images=images, return_tensors="pt", **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List] = None,  # nanoVLM format
        pixel_values: Optional[torch.FloatTensor] = None,  # Idefics3 format
        image_grid_thw: Optional[torch.LongTensor] = None,  # Idefics3 format
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass compatible with both nanoVLM and Idefics3 interfaces.

        Args:
            input_ids: (batch, seq_len) token IDs
            images: List of image tensors (nanoVLM format) - if provided, overrides pixel_values
            pixel_values: (batch, num_images, channels, height, width) image pixels (Idefics3 format)
            image_grid_thw: (batch, num_images, 3) image grid dimensions (Idefics3 format)
            attention_mask: (batch, seq_len) attention mask
            targets: (batch, seq_len) target token IDs for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar tensor if targets provided, else None
        """
        # Convert nanoVLM images format to Idefics3 format if needed
        if images is not None:
            pixel_values, image_grid_thw = self._process_images_to_pixel_values(images, input_ids.device)

        # Forward through Idefics3Model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        hidden_states = outputs[0]  # (batch, seq, hidden)
        logits = self.lm_head(hidden_states)  # (batch, seq, vocab)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Same as nanoVLM: compute loss over all tokens with -100 masking
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List] = None,  # nanoVLM format
        pixel_values: Optional[torch.FloatTensor] = None,  # Idefics3 format
        image_grid_thw: Optional[torch.LongTensor] = None,  # Idefics3 format
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        greedy: bool = False,  # nanoVLM compatibility
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Greedy or sampling-based generation compatible with both nanoVLM and Idefics3.

        Args:
            input_ids: (batch, seq) token IDs
            images: List of image tensors (nanoVLM format)
            pixel_values: Image pixels (Idefics3 format)
            image_grid_thw: Image grid (Idefics3 format)
            attention_mask: Attention mask
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k/top_p: Sampling parameters
            greedy: Force greedy decoding (nanoVLM compatibility)
            eos_token_id: End-of-sequence token

        Returns:
            generated_ids: (batch, original_len + new_tokens)
        """
        # Convert nanoVLM images format if provided
        if images is not None:
            pixel_values, image_grid_thw = self._process_images_to_pixel_values(images, input_ids.device)

        # Handle greedy flag from nanoVLM
        if greedy:
            temperature = 0.0

        if eos_token_id is None:
            eos_token_id = self.cfg.eos_token_id

        device = next(self.parameters()).device
        batch_size = input_ids.size(0)

        # Move inputs to device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            logits, past_key_values = self._generate_step(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                pixel_values=pixel_values if past_key_values is None else None,
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            # Sample next token
            next_logits = logits[:, -1, :]  # (batch, vocab)

            if temperature > 0:
                # Apply temperature and sampling
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        # Return only the newly generated tokens
        return generated[:, input_ids.size(1):]

    def _generate_step(self, input_ids, pixel_values, image_grid_thw, attention_mask, past_key_values):
        """Single generation step with KV caching"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        past_kv = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        return logits, past_kv

    # -------------------------
    # Pretrained loading/saving
    # -------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> "GraniteDoclingVLM":
        """
        Load model from HF checkpoint or local directory.

        Args:
            repo_id_or_path: HF repo ID or local path
            torch_dtype: Optional dtype to cast model
            device: Optional device to move model to
            strict: Whether to strictly enforce state dict matching
        """
        from transformers import AutoProcessor

        # Check if local path or HF repo
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Config not found at {config_path}")

            # Load config
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            # Download from HF
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

        # Convert to GraniteDoclingConfig
        cfg = _config_from_idefics3(config_dict)
        cfg.hf_checkpoint = repo_id_or_path

        # Load processor (includes tokenizer + image_processor)
        print(f"Loading processor (tokenizer + image_processor) from: {repo_id_or_path}")
        processor = AutoProcessor.from_pretrained(repo_id_or_path, trust_remote_code=True)

        # Create model (will load Idefics3 backbone from HF)
        model = cls(cfg, load_backbone=True)

        # Attach processor components to model
        model.processor = processor
        model.tokenizer = processor.tokenizer
        model.image_processor = processor.image_processor

        # Apply dtype and device
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        if device is not None:
            model = model.to(device)

        model.eval()
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and config to directory"""
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        # Save model weights
        state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        safe_save_file(state_dict, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """Push model to Hugging Face Hub"""
        from huggingface_hub import create_repo, upload_folder

        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print(f"Created repo: {repo_url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            # Create README
            readme_path = os.path.join(tmpdir, "README.md")
            with open(readme_path, "w") as f:
                f.write(_generate_model_card(repo_id, self.cfg))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmpdir,
                commit_message="Upload Granite Docling VLM"
            )


# -------------------------
# Helper functions
# -------------------------

def _config_from_idefics3(config_dict: dict) -> GraniteDoclingConfig:
    """Convert Idefics3 config dict to GraniteDoclingConfig"""
    text_cfg = config_dict.get("text_config", {})
    vision_cfg = config_dict.get("vision_config", {})

    return GraniteDoclingConfig(
        # Vision
        vit_hidden_dim=vision_cfg.get("hidden_size", 768),
        vit_patch_size=vision_cfg.get("patch_size", 16),
        vit_img_size=vision_cfg.get("image_size", 512),
        vit_n_heads=vision_cfg.get("num_attention_heads", 12),
        vit_n_blocks=vision_cfg.get("num_hidden_layers", 12),
        vit_inter_dim=vision_cfg.get("intermediate_size", 3072),
        vit_ln_eps=vision_cfg.get("layer_norm_eps", 1e-6),
        # Language
        lm_hidden_dim=text_cfg.get("hidden_size", 576),
        lm_inter_dim=text_cfg.get("intermediate_size", 1536),
        lm_vocab_size=text_cfg.get("vocab_size", 100352),
        lm_n_heads=text_cfg.get("num_attention_heads", 9),
        lm_n_kv_heads=text_cfg.get("num_key_value_heads", 3),
        lm_n_blocks=text_cfg.get("num_hidden_layers", 30),
        lm_max_position_embeddings=text_cfg.get("max_position_embeddings", 8192),
        lm_rms_eps=text_cfg.get("rms_norm_eps", 1e-5),
        lm_rope_theta=text_cfg.get("rope_theta", 100000.0),
        # Special tokens
        image_token_id=config_dict.get("image_token_id", 100270),
        bos_token_id=config_dict.get("bos_token_id", 100264),
        eos_token_id=config_dict.get("eos_token_id", 100257),
        pad_token_id=config_dict.get("pad_token_id", 100257),
        scale_factor=config_dict.get("scale_factor", 4),
    )


def _generate_model_card(repo_id: str, cfg: GraniteDoclingConfig) -> str:
    """Generate model card for HF Hub"""
    return f"""---
library_name: nanovlm
tags:
- vision-language
- multimodal
- granite
- docling
- idefics3
license: mit
---

# {repo_id}

Granite Docling VLM integrated with nanoVLM framework.

## Model Details

- **Architecture**: Idefics3 (Vision + LLaMA)
- **Vision Encoder**: {cfg.vit_n_blocks} layer ViT ({cfg.vit_hidden_dim}d, patch_size={cfg.vit_patch_size})
- **Language Model**: {cfg.lm_n_blocks} layer LLaMA ({cfg.lm_hidden_dim}d, {cfg.lm_n_heads} heads)
- **Parameters**: ~258M
- **Original Checkpoint**: {cfg.hf_checkpoint}

## Usage

```python
from models.granite_docling_vlm import GraniteDoclingVLM

model = GraniteDoclingVLM.from_pretrained("{repo_id}")
```

## Citation

Based on IBM Granite Docling and nanoVLM framework.
"""
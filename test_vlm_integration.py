"""
nanoVLM Integration Testing Suite

This script validates that a modified VLM implementation is compatible with:
1. Existing VLMConfig from models/config.py
2. Existing train.py training loop
3. Expected model interfaces and methods

Use this when you modify vision_language_model.py to ensure it still works
with the rest of the codebase.

Usage:
    python test_vlm_integration.py
    python test_vlm_integration.py --config-only  # Just test config compatibility
    python test_vlm_integration.py --verbose
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Import existing components
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig, TrainConfig
from data.processors import get_tokenizer, get_image_processor


class VLMIntegrationTester:
    """Test VLM compatibility with existing codebase"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _log(self, message, level="INFO"):
        """Log message"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def _record_test(self, test_name, passed, details=None, error=None):
        """Record test result"""
        self.results["tests"][test_name] = {
            "passed": passed,
            "details": details or {},
            "error": str(error) if error else None
        }
        status = "✅ PASS" if passed else "❌ FAIL"
        self._log(f"{status}: {test_name}")
        if error and self.verbose:
            self._log(f"  Error: {error}", "ERROR")

    def test_config_compatibility(self):
        """Test 1: VLM can be initialized with VLMConfig"""
        self._log("=" * 60)
        self._log("Test 1: VLMConfig Compatibility")
        self._log("=" * 60)

        try:
            # Create default config
            cfg = VLMConfig()

            self._log(f"  Initializing VLM with default VLMConfig...")

            # Try to initialize VLM
            model = VisionLanguageModel(cfg, load_backbone=False)

            # Check required attributes exist
            required_attrs = ['vision_encoder', 'decoder', 'MP', 'cfg', 'tokenizer']
            missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]

            if missing_attrs:
                raise AttributeError(f"Missing required attributes: {missing_attrs}")

            details = {
                "has_vision_encoder": hasattr(model, 'vision_encoder'),
                "has_decoder": hasattr(model, 'decoder'),
                "has_mp": hasattr(model, 'MP'),
                "has_tokenizer": hasattr(model, 'tokenizer'),
                "config_type": type(cfg).__name__
            }

            self._log(f"  ✓ All required components present")
            self._record_test("config_compatibility", True, details)
            return True, model, cfg

        except Exception as e:
            self._record_test("config_compatibility", False, error=e)
            return False, None, None

    def test_forward_signature(self, model, cfg):
        """Test 2: Forward method has correct signature"""
        self._log("\n" + "=" * 60)
        self._log("Test 2: Forward Method Signature")
        self._log("=" * 60)

        try:
            # Check forward method exists and accepts expected args
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())

            # Expected parameters (from train.py usage)
            expected_params = ['input_ids', 'images', 'attention_mask', 'targets']
            missing_params = [p for p in expected_params if p not in params]

            if missing_params:
                raise ValueError(f"Forward missing parameters: {missing_params}")

            self._log(f"  Forward signature: {sig}")
            self._log(f"  ✓ All expected parameters present")

            details = {
                "signature": str(sig),
                "parameters": params
            }

            self._record_test("forward_signature", True, details)
            return True

        except Exception as e:
            self._record_test("forward_signature", False, error=e)
            return False

    def test_forward_pass(self, model, cfg):
        """Test 3: Forward pass with dummy data (training-like)"""
        self._log("\n" + "=" * 60)
        self._log("Test 3: Forward Pass (Training Mode)")
        self._log("=" * 60)

        try:
            model = model.to(self.device)
            model.train()  # Set to training mode

            batch_size = 2
            seq_len = 100

            # Create dummy training batch (similar to train.py)
            input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len)).to(self.device)
            images = [torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size).to(self.device) for _ in range(batch_size)]
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            targets = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len)).to(self.device)
            targets[:, :50] = -100  # Mask first 50 tokens (like in training)

            self._log(f"  Input shape: {input_ids.shape}")
            self._log(f"  Images: {len(images)} images")
            self._log(f"  Targets shape: {targets.shape}")

            # Forward pass
            logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=targets)

            # Validate outputs
            if loss is None:
                raise ValueError("Loss should not be None when targets are provided")

            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss should be torch.Tensor, got {type(loss)}")

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite: {loss.item()}")

            details = {
                "input_shape": list(input_ids.shape),
                "output_shape": list(logits.shape),
                "loss_value": f"{loss.item():.4f}",
                "loss_finite": torch.isfinite(loss).item(),
                "output_dtype": str(logits.dtype)
            }

            self._log(f"  Output shape: {details['output_shape']}")
            self._log(f"  Loss: {details['loss_value']}")
            self._log(f"  ✓ Forward pass successful with loss")

            self._record_test("forward_pass_training", True, details)
            return True

        except Exception as e:
            self._record_test("forward_pass_training", False, error=e)
            return False

    def test_generate_method(self, model, cfg):
        """Test 4: Generate method exists and works"""
        self._log("\n" + "=" * 60)
        self._log("Test 4: Generate Method")
        self._log("=" * 60)

        try:
            model.eval()

            # Check generate method exists
            if not hasattr(model, 'generate'):
                raise AttributeError("Model missing 'generate' method")

            import inspect
            sig = inspect.signature(model.generate)
            params = list(sig.parameters.keys())

            # Expected generate parameters (from generate.py)
            expected_gen_params = ['input_ids', 'images', 'max_new_tokens']
            missing_gen_params = [p for p in expected_gen_params if p not in params]

            if missing_gen_params:
                raise ValueError(f"Generate missing parameters: {missing_gen_params}")

            # Test generation
            batch_size = 1
            seq_len = 20
            input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len)).to(self.device)
            images = torch.randn(batch_size, 3, cfg.vit_img_size, cfg.vit_img_size).to(self.device)

            with torch.no_grad():
                generated = model.generate(input_ids, images, max_new_tokens=10)

            if not isinstance(generated, torch.Tensor):
                raise TypeError(f"Generate should return torch.Tensor, got {type(generated)}")

            details = {
                "generate_signature": str(sig),
                "input_length": seq_len,
                "generated_length": generated.shape[1],
                "output_shape": list(generated.shape)
            }

            self._log(f"  Generate signature: {sig}")
            self._log(f"  Generated {details['generated_length']} tokens")
            self._log(f"  ✓ Generate method works")

            self._record_test("generate_method", True, details)
            return True

        except Exception as e:
            self._record_test("generate_method", False, error=e)
            return False

    def test_backward_pass(self, model, cfg):
        """Test 5: Gradients flow correctly (for training)"""
        self._log("\n" + "=" * 60)
        self._log("Test 5: Backward Pass & Gradients")
        self._log("=" * 60)

        try:
            model.train()

            # Create dummy batch
            batch_size = 1
            seq_len = 50
            input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len)).to(self.device)
            images = [torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size).to(self.device)]
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            targets = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len)).to(self.device)
            targets[:, :25] = -100

            # Forward pass
            logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=targets)

            # Backward pass
            loss.backward()

            # Check gradients exist
            has_grads = []
            no_grads = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        has_grads.append(name)
                    else:
                        no_grads.append(name)

            if no_grads:
                self._log(f"  Warning: {len(no_grads)} parameters have no gradients", "WARN")

            # Zero gradients (cleanup)
            model.zero_grad()

            details = {
                "loss_value": f"{loss.item():.4f}",
                "params_with_grads": len(has_grads),
                "params_without_grads": len(no_grads),
                "gradient_flow": "OK" if has_grads else "FAIL"
            }

            self._log(f"  Parameters with gradients: {len(has_grads)}")
            self._log(f"  Parameters without gradients: {len(no_grads)}")
            self._log(f"  ✓ Gradients computed successfully")

            success = len(has_grads) > 0
            self._record_test("backward_pass", success, details)
            return success

        except Exception as e:
            self._record_test("backward_pass", False, error=e)
            return False

    def test_save_load_methods(self, model):
        """Test 6: Save/load methods exist (for checkpointing)"""
        self._log("\n" + "=" * 60)
        self._log("Test 6: Save/Load Methods")
        self._log("=" * 60)

        try:
            # Check methods exist
            required_methods = ['save_pretrained', 'push_to_hub']
            required_class_methods = ['from_pretrained']

            missing_methods = [m for m in required_methods if not hasattr(model, m)]
            missing_class_methods = [m for m in required_class_methods if not hasattr(VisionLanguageModel, m)]

            if missing_methods:
                raise AttributeError(f"Model missing methods: {missing_methods}")
            if missing_class_methods:
                raise AttributeError(f"VisionLanguageModel class missing methods: {missing_class_methods}")

            # Check if methods are callable
            for method_name in required_methods:
                method = getattr(model, method_name)
                if not callable(method):
                    raise TypeError(f"{method_name} is not callable")

            details = {
                "has_save_pretrained": hasattr(model, 'save_pretrained'),
                "has_from_pretrained": hasattr(VisionLanguageModel, 'from_pretrained'),
                "has_push_to_hub": hasattr(model, 'push_to_hub')
            }

            self._log(f"  ✓ save_pretrained: {details['has_save_pretrained']}")
            self._log(f"  ✓ from_pretrained: {details['has_from_pretrained']}")
            self._log(f"  ✓ push_to_hub: {details['has_push_to_hub']}")

            self._record_test("save_load_methods", True, details)
            return True

        except Exception as e:
            self._record_test("save_load_methods", False, error=e)
            return False

    def test_tokenizer_compatibility(self, model, cfg):
        """Test 7: Tokenizer works with model config"""
        self._log("\n" + "=" * 60)
        self._log("Test 7: Tokenizer Compatibility")
        self._log("=" * 60)

        try:
            # Get tokenizer (same way as train.py)
            tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

            # Check special tokens
            if not hasattr(tokenizer, 'image_token_id'):
                raise AttributeError("Tokenizer missing 'image_token_id'")

            if tokenizer.image_token_id is None:
                raise ValueError("image_token_id is None")

            # Check model tokenizer matches
            if not hasattr(model, 'tokenizer'):
                raise AttributeError("Model missing 'tokenizer' attribute")

            # Test tokenization
            test_text = "What is in this image?"
            tokens = tokenizer.encode(test_text)

            details = {
                "tokenizer_type": type(tokenizer).__name__,
                "vocab_size": tokenizer.vocab_size,
                "has_image_token": hasattr(tokenizer, 'image_token_id'),
                "image_token_id": tokenizer.image_token_id,
                "test_encoding_length": len(tokens)
            }

            self._log(f"  Tokenizer vocab size: {details['vocab_size']}")
            self._log(f"  Image token ID: {details['image_token_id']}")
            self._log(f"  ✓ Tokenizer compatible")

            self._record_test("tokenizer_compatibility", True, details)
            return True

        except Exception as e:
            self._record_test("tokenizer_compatibility", False, error=e)
            return False

    def test_train_config_compatibility(self):
        """Test 8: TrainConfig can be loaded"""
        self._log("\n" + "=" * 60)
        self._log("Test 8: TrainConfig Compatibility")
        self._log("=" * 60)

        try:
            # Create train config
            train_cfg = TrainConfig()

            # Check required training attributes
            required_train_attrs = [
                'lr_mp', 'lr_vision_backbone', 'lr_language_backbone',
                'batch_size', 'gradient_accumulation_steps', 'max_grad_norm'
            ]

            missing_train_attrs = [attr for attr in required_train_attrs if not hasattr(train_cfg, attr)]

            if missing_train_attrs:
                raise AttributeError(f"TrainConfig missing attributes: {missing_train_attrs}")

            details = {
                "lr_mp": train_cfg.lr_mp,
                "lr_vision": train_cfg.lr_vision_backbone,
                "lr_language": train_cfg.lr_language_backbone,
                "batch_size": train_cfg.batch_size,
                "grad_accum_steps": train_cfg.gradient_accumulation_steps
            }

            self._log(f"  Batch size: {details['batch_size']}")
            self._log(f"  LR (MP): {details['lr_mp']}")
            self._log(f"  ✓ TrainConfig compatible")

            self._record_test("train_config_compatibility", True, details)
            return True

        except Exception as e:
            self._record_test("train_config_compatibility", False, error=e)
            return False

    def run_all_tests(self):
        """Run all integration tests"""
        self._log("\n" + "=" * 70)
        self._log("🔧 nanoVLM Integration Test Suite")
        self._log("=" * 70)
        self._log(f"Device: {self.device}")
        self._log(f"PyTorch: {torch.__version__}")
        self._log("")

        # Test 1: Config compatibility
        success, model, cfg = self.test_config_compatibility()
        if not success:
            self._log("\n❌ Config compatibility failed - cannot proceed", "ERROR")
            return False

        # Run remaining tests
        tests = [
            (self.test_forward_signature, [model, cfg]),
            (self.test_forward_pass, [model, cfg]),
            (self.test_generate_method, [model, cfg]),
            (self.test_backward_pass, [model, cfg]),
            (self.test_save_load_methods, [model]),
            (self.test_tokenizer_compatibility, [model, cfg]),
            (self.test_train_config_compatibility, []),
        ]

        for test_func, args in tests:
            try:
                test_func(*args)
            except Exception as e:
                self._log(f"Test failed with exception: {e}", "ERROR")

        # Summary
        self._log("\n" + "=" * 70)
        self._log("📊 Test Summary")
        self._log("=" * 70)

        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"].values() if t["passed"])
        failed_tests = total_tests - passed_tests

        self._log(f"Total Tests: {total_tests}")
        self._log(f"Passed: {passed_tests} ✅")
        self._log(f"Failed: {failed_tests} ❌")

        if failed_tests > 0:
            self._log("\nFailed tests:", "ERROR")
            for name, result in self.results["tests"].items():
                if not result["passed"]:
                    self._log(f"  - {name}: {result.get('error', 'Unknown error')}", "ERROR")

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        self._log(f"\nSuccess Rate: {success_rate:.1f}%")

        return passed_tests == total_tests


def main():
    parser = argparse.ArgumentParser(description="nanoVLM Integration Testing")
    parser.add_argument("--config-only", action="store_true",
                       help="Only test config compatibility")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    # Create tester
    tester = VLMIntegrationTester(verbose=not args.quiet)

    if args.config_only:
        # Just test config
        success, _, _ = tester.test_config_compatibility()
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
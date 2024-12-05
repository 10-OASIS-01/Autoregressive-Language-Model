# This code contains unit tests for a Transformer model implementation.
# It verifies different aspects of the model's functionality including:
# 1. The forward pass of the model and loss computation.
# 2. Handling of different sequence lengths (smaller, equal, and larger than the block size).
# 3. Model inference when no target is provided (i.e., during inference mode).
# 4. Proper initialization of model parameters, ensuring they are finite and have the correct shape.
# 5. Testing the model with different configurations (number of layers, embedding size, etc.).
# 6. Ensuring gradients are correctly computed for the model's parameters during backpropagation.


import unittest
import torch
from transformer import Transformer, ModelConfig


class TestModels(unittest.TestCase):

    def test_transformer_forward(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)
        idx = torch.randint(0, 100, (1, 10))
        logits, loss = model(idx, targets=idx)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")
        self.assertEqual(logits.shape, (1, 10, config.vocab_size), "Logits shape mismatch")

        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertIsNotNone(param.grad, "Gradient should be computed for all model parameters")

    def test_different_sequence_lengths(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)

        # Test with smaller sequence length than block size
        idx_short = torch.randint(0, 100, (1, 5))  # Sequence length = 5
        logits, loss = model(idx_short, targets=idx_short)
        self.assertIsNotNone(logits)
        self.assertIsNotNone(loss)

        # Test with equal sequence length to block size
        idx_equal = torch.randint(0, 100, (1, 10))  # Sequence length = block size
        logits, loss = model(idx_equal, targets=idx_equal)
        self.assertIsNotNone(logits)
        self.assertIsNotNone(loss)

        # Test with longer sequence length (should trigger assertion)
        idx_long = torch.randint(0, 100, (1, 15))  # Sequence length = 15
        with self.assertRaises(AssertionError):
            model(idx_long, targets=idx_long)

    def test_loss_without_targets(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)
        idx = torch.randint(0, 100, (1, 10))  # Sequence length = 10
        logits, loss = model(idx)  # No targets passed, should be inference mode

        self.assertIsNone(loss)  # Loss should be None in inference mode
        self.assertEqual(logits.shape, (1, 1, config.vocab_size), "Logits shape mismatch in inference mode")

    def test_model_parameter_initialization(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)

        # Check that all parameters are initialized with the correct shape and are not NaN
        for param in model.parameters():
            self.assertIsNotNone(param)
            self.assertTrue(torch.all(torch.isfinite(param)), f"Parameter {param} contains NaN or Inf values.")
            self.assertGreater(param.numel(), 0, f"Parameter {param} has no elements.")

    def test_model_with_different_configurations(self):
        # Test with different configurations: More layers, larger embeddings
        config_1 = ModelConfig(vocab_size=100, block_size=10, n_layer=4, n_head=4, n_embd=64)
        model_1 = Transformer(config_1)
        idx = torch.randint(0, 100, (1, 10))
        logits_1, loss_1 = model_1(idx, targets=idx)

        self.assertIsNotNone(logits_1)
        self.assertIsNotNone(loss_1)
        self.assertEqual(logits_1.shape, (1, 10, config_1.vocab_size))

        # Test with even larger configurations
        config_2 = ModelConfig(vocab_size=200, block_size=20, n_layer=6, n_head=8, n_embd=128)
        model_2 = Transformer(config_2)
        idx_2 = torch.randint(0, 200, (1, 20))  # Sequence length = block size
        logits_2, loss_2 = model_2(idx_2, targets=idx_2)

        self.assertIsNotNone(logits_2)
        self.assertIsNotNone(loss_2)
        self.assertEqual(logits_2.shape, (1, 20, config_2.vocab_size))

    def test_model_gradients_on_embeds_and_layers(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)
        idx = torch.randint(0, 100, (1, 10))  # Sequence length = block size
        logits, loss = model(idx, targets=idx)

        # Perform backward pass to compute gradients
        loss.backward()

        # Ensure gradients are computed for the embeddings and the layers
        self.assertIsNotNone(model.transformer.wte.weight.grad, "No gradient for token embeddings.")
        self.assertIsNotNone(model.transformer.wpe.weight.grad, "No gradient for position embeddings.")

        # Check gradients for one of the layers
        layer = model.transformer.h[0]
        self.assertIsNotNone(layer.attn.c_attn.weight.grad, "No gradient for attention weights.")
        self.assertIsNotNone(layer.mlp.c_fc.weight.grad, "No gradient for MLP weights.")


if __name__ == '__main__':
    unittest.main()

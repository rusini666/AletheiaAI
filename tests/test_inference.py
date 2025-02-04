import unittest
import torch
from detector import CustomOPTClassifier
from transformers import AutoModelForCausalLM

class TestModelInference(unittest.TestCase):

    def setUp(self):
        base_model = "facebook/opt-1.3b"
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(base_model)
        self.classifier = CustomOPTClassifier(self.pretrained_model)

    def test_output_shape(self):
        dummy_input = torch.randint(0, 50272, (1, 150))  # Simulating a tokenized input
        dummy_mask = torch.ones((1, 150))  # Attention mask
        
        with torch.no_grad():
            output = self.classifier(dummy_input, dummy_mask)

        self.assertEqual(output.shape, torch.Size([1, 1]), "Output shape should be (1,1) for binary classification")

    def test_output_type(self):
        dummy_input = torch.randint(0, 50272, (1, 150))
        dummy_mask = torch.ones((1, 150))

        with torch.no_grad():
            output = self.classifier(dummy_input, dummy_mask)

        self.assertTrue(isinstance(output, torch.Tensor), "Output should be a PyTorch tensor")

if __name__ == '__main__':
    unittest.main()
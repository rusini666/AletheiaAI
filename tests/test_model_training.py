import pytest
import torch
from detector import CustomOPTClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
pretrained_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
model = CustomOPTClassifier(pretrained_model).to("cpu")  # Testing on CPU

def test_model_output():
    sample_text = "This is an AI-generated text."
    encoded_input = tokenizer(sample_text, return_tensors="pt")

    with torch.no_grad():
        logits = model(encoded_input["input_ids"], encoded_input["attention_mask"])

    assert logits.shape == (1, 1), "Output should have shape (1,1) for binary classification"

    print(f"✅ Model output shape: {logits.shape}")

if __name__ == "__main__":
    pytest.main()

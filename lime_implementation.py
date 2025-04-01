import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_text import LimeTextExplainer

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from tqdm import tqdm
import nltk
import re
import os
from datetime import datetime

nltk.download('punkt')
nltk.download('stopwords')

# Initialize configurations
app_configs = {}

# Custom Tokenization and Preprocessing
class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        print("Initializing PreprocessDataset...")
        texts = dataframe.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self.texts = [tokenizer(text, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
                      for text in texts]
        print("Texts after preprocessing and tokenization:", self.texts[:1])  # Debug the first tokenized text
        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _preprocess(self, text):
        print("Preprocessing text:", text)  # Debugging the text before preprocessing
        text = re.sub(r'(@.*?)[\s]', ' ', text)  # Remove mentions
        text = re.sub(r'\s+', ' ', text)         # Remove multiple spaces
        text = re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)  # Remove URLs
        text = ''.join(character for character in text if character not in string.punctuation)  # Remove punctuation
        tokens = nltk.word_tokenize(text)
        
        # Filter out stopwords and globally common words
        stop_words = nltk.corpus.stopwords.words('english')
        custom_stop_words = ['the', 'is', 'and', 'to', 'in', 'a', 'of']  # You can add more
        tokens = [token for token in tokens if token.lower() not in stop_words + custom_stop_words]

        processed_text = " ".join(tokens).strip()
        print("Processed text:", processed_text)  # Debugging the processed text
        return processed_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if hasattr(self, 'labels') else -1
        return text, label

# OPT-based Classifier
class CustomOPTClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomOPTClassifier, self).__init__()
        print("Initializing CustomOPTClassifier...")
        self.opt = pretrained_model
        self.fc1 = nn.Linear(pretrained_model.config.vocab_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        print("Running forward pass of the classifier...")  # Debugging forward pass
        opt_out = self.opt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        print("OPT model output:", opt_out)
        x = self.fc1(opt_out)
        x = self.relu(x)
        x = self.fc2(x)
        print("Final output of the classifier:", x)
        return x

def str_to_class(classname):
    return globals()[classname]

def target_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    app_configs['device'] = device
    print(f"Using device: {device}")
    return device

def get_pretrained_model():
    print("Loading tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    model_with_lora = get_peft_model(pretrained_model, lora_config)
    print("Model loaded with LoRA configuration.")
    return tokenizer, model_with_lora

# Classify a single text sample
def classify_single_text(text_sample, model_name=''):
    if model_name:
        app_configs['model_name'] = model_name

    # Load the model and tokenizer
    print("Classifying single text sample...")
    tokenizer, pretrained_model = get_pretrained_model()

    # Recreate the model architecture
    classifier = CustomOPTClassifier(pretrained_model)
    model_path = os.path.join(app_configs['models_path'], app_configs['model_name'] + ".pt")
    print("Loading model from path:", model_path)
    
    try:
        classifier.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    classifier = classifier.to(app_configs['device'])
    classifier.eval()

    # Preprocess the input text
    processed_sample = PreprocessDataset(pd.DataFrame({"text": [text_sample]}), tokenizer)[0]
    input_ids = processed_sample[0]['input_ids'].squeeze(1).to(app_configs['device'])
    attention_mask = processed_sample[0]['attention_mask'].to(app_configs['device'])

    # Get the prediction from the model
    with torch.no_grad():
        output = classifier(input_ids, attention_mask)
        print("Model output:", output)
        prediction = (output >= 0.5).int().item()  # 0.5 is the threshold
        print("Predicted label:", prediction)

    # Interpret the prediction
    return "AI-generated" if prediction == 1 else "Human-written"

# Embedding-Level LIME Explanation
def predict_proba(text_samples):
    print("Generating probabilities for LIME explanation...")
    tokenizer, pretrained_model = get_pretrained_model()
    model = str_to_class(app_configs['classifier'])(pretrained_model)
    model_path = os.path.join(app_configs['models_path'], app_configs['model_name'] + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.to(app_configs['device'])
    model.eval()

    all_probs = []
    batch_size = 4
    for i in range(0, len(text_samples), batch_size):
        batch_text_samples = text_samples[i:i+batch_size]

        # Tokenize the batch
        tokenized_inputs = tokenizer(batch_text_samples, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs['input_ids'].to(app_configs['device'])
        attention_mask = tokenized_inputs['attention_mask'].to(app_configs['device'])

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            print("Batch model output:", outputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            print("Probabilities:", probs)

        all_probs.append(np.concatenate([(1 - probs), probs], axis=1))

    return np.concatenate(all_probs, axis=0)

# LIME Explanation with Embedding-Level Importance
def explain_text_classification(text_sample):
    print("Generating LIME explanation...")
    explainer = LimeTextExplainer(class_names=["Human-written", "AI-generated"])

    # Generate explanation
    try:
        explanation = explainer.explain_instance(text_sample, predict_proba, num_features=10)
        # Option 1: Save explanation to HTML file
        explanation.save_to_file('lime_explanation.html')
        print("LIME explanation saved to 'lime_explanation.html'")
    except Exception as e:
        print(f"Error during LIME explanation generation: {e}")

def main():
    absolute_path = os.path.abspath('/Users/rusinitharagunarathne/Documents/AletheiaAI/data/models')
    app_configs['base_model'] = 'facebook/opt-1.3b'
    app_configs['classifier'] = 'CustomOPTClassifier'
    app_configs['models_path'] = absolute_path
    app_configs['device'] = target_device()

    # Example usage: classify a single input text
    sample_text = "How to Get Secured in the Internet'\n\nSecurity on the internet is essential..."
    model_for_classify = '202409230028_subtaskA_monolingual_facebook_opt-1.3b'

    print(f"Classifying the input text: '{sample_text}'")
    result = classify_single_text(sample_text, model_name=model_for_classify)
    if result:
        print(f"The text is classified as: {result}")

    # Explain the classification using LIME with custom tokenization and embeddings
    print("Explaining the classification...")
    explain_text_classification(sample_text)

if __name__ == "__main__":
    main()
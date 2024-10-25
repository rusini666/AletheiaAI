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
        texts = dataframe.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self.texts = [tokenizer(text, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
                      for text in texts]
        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _preprocess(self, text):
        text = re.sub(r'(@.*?)[\s]', ' ', text)  # Remove mentions
        text = re.sub(r'\s+', ' ', text)         # Remove multiple spaces
        text = re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)  # Remove URLs
        text = ''.join(character for character in text if character not in string.punctuation)  # Remove punctuation
        tokens = nltk.word_tokenize(text)
        
        # Filter out stopwords and globally common words
        stop_words = nltk.corpus.stopwords.words('english')
        custom_stop_words = ['the', 'is', 'and', 'to', 'in', 'a', 'of']  # You can add more
        tokens = [token for token in tokens if token.lower() not in stop_words + custom_stop_words]

        return " ".join(tokens).strip()

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
        self.opt = pretrained_model
        self.fc1 = nn.Linear(pretrained_model.config.vocab_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        opt_out = self.opt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        x = self.fc1(opt_out)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def str_to_class(classname):
    return globals()[classname]

def target_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    app_configs['device'] = device
    print(f"Using device: {device}")
    return device

def get_pretrained_model():
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    model_with_lora = get_peft_model(pretrained_model, lora_config)
    return tokenizer, model_with_lora

# Classify a single text sample
def classify_single_text(text_sample, model_name=''):
    if model_name:
        app_configs['model_name'] = model_name

    # Load the model and tokenizer
    tokenizer, pretrained_model = get_pretrained_model()

    # Recreate the model architecture
    classifier = CustomOPTClassifier(pretrained_model)
    model_path = os.path.join(app_configs['models_path'], app_configs['model_name'] + ".pt")
    classifier.load_state_dict(torch.load(model_path))

    classifier = classifier.to(app_configs['device'])
    classifier.eval()

    # Preprocess the input text
    processed_sample = PreprocessDataset(pd.DataFrame({"text": [text_sample]}), tokenizer)[0]
    input_ids = processed_sample[0]['input_ids'].squeeze(1).to(app_configs['device'])
    attention_mask = processed_sample[0]['attention_mask'].to(app_configs['device'])

    # Get the prediction from the model
    with torch.no_grad():
        output = classifier(input_ids, attention_mask)
        prediction = (output >= 0.5).int().item()  # 0.5 is the threshold

    # Interpret the prediction
    return "AI-generated" if prediction == 1 else "Human-written"

# Embedding-Level LIME Explanation
def predict_proba(text_samples):
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
            probs = torch.sigmoid(outputs).cpu().numpy()

        all_probs.append(np.concatenate([(1 - probs), probs], axis=1))

    return np.concatenate(all_probs, axis=0)

# LIME Explanation with Embedding-Level Importance
def explain_text_classification(text_sample):
    explainer = LimeTextExplainer(class_names=["Human-written", "AI-generated"])

    # Generate explanation
    explanation = explainer.explain_instance(text_sample, predict_proba, num_features=10)

    # Option 1: Save explanation to HTML file
    explanation.save_to_file('lime_explanation.html')
    print("LIME explanation saved to 'lime_explanation.html'")

def main():
    absolute_path = os.path.abspath('/Users/rusinitharagunarathne/Documents/AletheiaAI/data/models')
    app_configs['base_model'] = 'facebook/opt-1.3b'
    app_configs['classifier'] = 'CustomOPTClassifier'
    app_configs['models_path'] = absolute_path
    app_configs['device'] = target_device()

    # Example usage: classify a single input text
    sample_text = "How to Get Secured in the Internet'\n\nSecurity on the internet is essential in this day and age. It\u2019s not just important to protect yourself from viruses, identity thieves, hackers and malware, but also from data breaches, censorship, online tracking and even government surveillance. If you want to stay safe and secure on the internet, there are some key steps you can take to protect yourself. \n\nFirst and foremost, it\u2019s important to take steps to protect your computer from malicious software such as viruses, worms, Trojan horses and spyware. This can be accomplished by running regular scans for malware, as well as running an up-to-date antivirus software on your computer. You should also avoid downloading files from untrusted sources, as these can contain malware. Additionally, it\u2019s a good idea to keep your software and operating system up-to-date, as these also provide security patches against new threats. \n\nAnother important point to take care of would be defending your computer from viruses. Viruses can be spread in a variety of ways, such as through email attachments or downloads from untrusted websites. To protect your computer, make sure you are using reliable antivirus software and regularly run updates. Additionally, be sure to backup your files regularly so that you won't lose any of your data in the event of an infection.\n\nWhen your PC or laptop is protected, you can use one of anonymizing services such as Anonymous Proxy Server, Anonymize.net, or similar in order to secure your IP address and make it invisible to the outside world. Anonymizing services provide a good amount of protection for your internet activities, although most of them won\u2019t protect your data when sending e-mails. For this purpose, there are other services such as secure email services, virtual private networks (VPNs) and Tor -- a system that conceals your identity and location on the web by directing your traffic through a series of encrypted \u201cnodes.\u201d\n\nIt\u2019s also important to consider protecting your domain and email address. If you own a website, you may want to consider offshore hosting for your domain. This type of hosting often provides more security, freedom and privacy for users, but keep in mind that it may also be subject to different regulations. Additionally, if you use e-mail, you may want to consider using a \u201cdisposable\u201d email address to protect your personal information and identity. When sending emails, it\u2019s important to make sure the address you use is not traceable back to you.\n\nOverall, there are many steps you can take to protect yourself online. By following the steps outlined in this article, you can be sure that you\u2019re taking the necessary measures to keep yourself safe and secure while surfing the web. Whether you\u2019re using the internet for business or pleasure, it\u2019s important to take the time to protect yourself."
    model_for_classify = '202409230028_subtaskA_monolingual_facebook_opt-1.3b'

    print(f"Classifying the input text: '{sample_text}'")
    result = classify_single_text(sample_text, model_name=model_for_classify)
    print(f"The text is classified as: {result}")

    # Explain the classification using LIME with custom tokenization and embeddings
    print("Explaining the classification...")
    explain_text_classification(sample_text)

if __name__ == "__main__":
    main()
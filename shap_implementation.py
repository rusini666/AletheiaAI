# Import statements
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
from torch import nn
from torch.utils.data import Dataset 
from torch.optim import AdamW
from torch.utils.data import DataLoader 
import string 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
from peft import get_peft_model, LoraConfig, TaskType 
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup 
from tqdm import tqdm 
import nltk 
import re 
import os 
from datetime import datetime

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Initialize configurations
app_configs = {}

# Dataset Preprocessing
class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        texts = dataframe.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self.texts = [tokenizer(text, padding='max_length', max_length=150, truncation=True, return_tensors="pt") for text in texts]
        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _preprocess(self, text):
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)
        text = ''.join(character for character in text if character not in string.punctuation)
        tokens = nltk.word_tokenize(text)
        stop_words = nltk.corpus.stopwords.words('english')
        tokens = [token for token in tokens if token not in stop_words]
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

# Device Selection
def target_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    app_configs['device'] = device
    print(f"Using device: {device}")
    return device

# Load pretrained model with LoRA
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

# Main function
def main():
    app_configs['base_model'] = 'facebook/opt-1.3b'
    app_configs['classifier'] = 'CustomOPTClassifier'
    app_configs['models_path'] = '/Users/rusinitharagunarathne/Documents/AletheiaAI/data/models'
    app_configs['device'] = target_device()
    
    # Example usage: classify a single input text
    sample_text = "Repairing the Motor Coupler in a Kenmore Washer can be a hassle, but it doesn't have to be. With a few common household tools, you can have your washing machine up and running in no time. Here are the steps you should take when attempting a repair: \n\n1. Unplug washer or disconnect power. Make sure there is no source of electricity before attempting any repairs. \n\n2. Turn off the water to the washer. This will remove any possible sources of water and reduce the possibility of water damage.\n\n3. Pull the machine out far enough to tip it back against the wall. It's best to do this in an area that can be easily cleaned up if anything spills. \n\n4. Disconnect the wire harness from the motor. This will allow you to replace the motor more easily. \n\n5. Remove the two snap clamps from the water pump (Do NOT disconnect the water lines or you will have a bit of mess). These need to be removed because they may prevent the pump from working properly. \n\n6. Remove the cabinet from the base:\n\nLook at the console (the thing with the knobs). Unscrew any screws that hold the console to the machine. \n\n7. Remove the motor. This can be done by unscrewing any screws that hold it in place.\n\n8. Remove the bad coupler. Unscrew any screws and push the coupler off of the motor.\n\n9. Reinstall the motor. Make sure to replace all of the screws that you removed. \n\n10. Remove the pump:\n\nUsing a screwdriver or your hand, unsnap the bottom retainer from the pump. \n\n11. Remove the drive motor:\n\nUsing a screwdriver, remove the screws (if used) which hold the top and bottom retainers to the front of the drive motor. Pry or break the motor couplings off of the drive shaft and gear case shaft. \n\n12. Push one of the new plastic motor couplings on the shaft of the drive motor.\n\n13. Push the other new plastic motor coupling on the shaft of the gear case.\n\n14. Place the new rubber isolation coupling on the drive motor coupling by lining up the holes in the isolation with the studs in the coupling, and then push on.\n\n15. Turn the isolation a coupling on the drive motor lining up the holes with the studs on the other coupling on the gear case.\n\n16. Replace the drive motor.\n\n17. Replace the pump.\n\n18. Replace the cabinet.\n\n19. Reconnect the wire harness and extra wire to the motor.\n\n20. Put the washer back down on all fours.\n\n21. Plug in washer or reconnect power. \n\nWith these easy steps, you'll be able to repair the motor coupler in a Kenmore washer in no time. Just remember to be careful while working with the motor and disconnect all sources of power before beginning any repairs. Once the repairs are complete, you'll be able to get back to the business of doing laundry."
    model_for_classify = '202409230028_subtaskA_monolingual_facebook_opt-1.3b'
    print(f"Classifying the input text: '{sample_text}'")
    result = classify_single_text(sample_text, model_name=model_for_classify)
    print(f"The text is classified as: {result}")

if __name__ == "__main__":
    main()

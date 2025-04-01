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
from collections import Counter

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

# Dataset Preprocessing
class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        texts = dataframe.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self.texts = [tokenizer(text, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
                      for text in texts]
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

def predict_proba(text_samples):
    # Tokenize the input text samples
    tokenizer, pretrained_model = get_pretrained_model()
    model = str_to_class(app_configs['classifier'])(pretrained_model)
    model_path = os.path.join(app_configs['models_path'], app_configs['model_name'] + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.to(app_configs['device'])
    model.eval()

    # Reduce memory consumption by processing in smaller batches
    all_probs = []
    batch_size = 4  # You can adjust this value based on your system's memory
    for i in range(0, len(text_samples), batch_size):
        batch_text_samples = text_samples[i:i+batch_size]

        # Tokenize the batch
        tokenized_inputs = tokenizer(batch_text_samples, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs['input_ids'].to(app_configs['device'])
        attention_mask = tokenized_inputs['attention_mask'].to(app_configs['device'])

        # Get predictions for the batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()

        all_probs.append(np.concatenate([(1 - probs), probs], axis=1))

    # Return probabilities for both classes: 1 - prob for human-written, and prob for AI-generated
    return np.concatenate(all_probs, axis=0)

# Calculate Global Feature Importance
def calculate_global_feature_importance(text_samples, tokenizer):
    feature_counter = Counter()
    for text in text_samples:
        tokenized = tokenizer.tokenize(text)
        feature_counter.update(tokenized)
    
    # Return the most common tokens
    return feature_counter.most_common(10)

def explain_text_classification(text_sample):
    explainer = LimeTextExplainer(class_names=["Human-written", "AI-generated"])

    # Generate explanation
    explanation = explainer.explain_instance(text_sample, predict_proba, num_features=6)

    # Option 1: Save explanation to HTML file
    explanation.save_to_file('lime_explanation.html')
    print("LIME explanation saved to 'lime_explanation.html'")

    # Option 2: Print explanation to console
    # for label, weight in explanation.as_list():
    #     print(f"{label}: {weight:.4f}")

def main():
    absolute_path = os.path.abspath('/Users/rusinitharagunarathne/Documents/AletheiaAI/data/models')
    app_configs['base_model'] = 'facebook/opt-1.3b'
    app_configs['full_dataset_path'] = '/Users/rusinitharagunarathne/Documents/AletheiaAI/data/subtaskA_dataset_monolingual.jsonl'
    app_configs['classifier'] = 'CustomOPTClassifier'
    app_configs['models_path'] = absolute_path
    app_configs['device'] = target_device()

    # Load your dataset (assuming you have a full dataset path in app_configs)
    df = pd.read_json(app_configs['full_dataset_path'], lines=True)
    text_samples = df['text'].tolist()  # Extract text samples from the dataset
    
    # Example usage: classify a single input text
    sample_text = ("Prepare for the dungeon! Enter the Cave! Go through the door going right. Using your shield, bump into the spike turtle enemies to flip them over. Collect your prize. Head left. Travel through the new area until you reach a split path covered in tracks. Hop in the mine cart which will take you to room 6. Room 7 contains 4 masked enemies. Once you kill them all a portal will appear. Avoid the spike turtle enemies and make your way right. Enter the room on the downward side of the map. Don't forget your prize. Cross the molten lake. Ride the lift over to the other side which is covered in little armored enemies. Float through the sky. Fly a little more. Head through the upper door. Use your key on the door nearest to the mine cart. Ignore the cart. Follow the path outlined in stone and then jump down. Claim the dungeon item, the flaming sword. Flip the hazardous platform. Take to the sky. Ignore the teleporter for now. Use the cane to flip over the mine cart. Obtain another key. Hop back into the cart. Shrink once again. Get back to your original size. Open the new pathway. Fill some more holes. Hit the switch. Go up to the next floor then into the next room. Cross the river of lava. Clear the path. Like before, use your cane to flip over the spiky platforms and carefully make your way to the upper-right door. Float up and land on the upper level. Hop into the tornado. Finally reach the chest. Open the portal. Get prepped. Return to the start of the lava river room and catch the platform going up this time. Expose the boss's weakness. Go for his weak point. Dodge the projectiles. Link finds the Cave of Flames shortly after entering the Mino Forest. The Cave of Flames is one of the six available dungeons in the game, and it's a tough challenge, especially since it's guarded by a boss whose weakness can be exposed by Link's Cane. After clearing the room of hazards, Link reaches a chest filled with valuable items, including the Flaming Sword, an item that can defeat the boss. The chest is then able to be opened, allowing the player to defeat the boss and escape the Cave of Flames. Note that this guide discusses major plot points and some of the items that can be found in the Cave of Flames room. Be aware that some of these items may not be accessible if you haven't collected the required amount of Quartz or the correct sequence of switches has not been triggered. The Cave of Flames room contains several hazards, including spiked turtles and spiked platforms. Link's shield can be used to bump these out of the way, but Link will have to be careful and avoid them altogether to make it to the boss and the chest beyond it. Next, Link needs to reach the boss's weak point: A spinning platform covered in spikes. Once Link reaches this, he can activate the switch to deactivate the spikes, allowing him to safely make his way to the chest and defeat the boss. The chest that Link reaches at the end of the Cave of Flames contains the Flaming Sword, a powerful item that can be used to defeat the final boss of the dungeon. After beating the boss, Link needs to activate the switch to disable the rotating spikes at the end of the Cave of Flames room. After deactivating the spikes, he can safely reach the chest and defeat the final boss of the dungeon. Link returns to the start of the Cave of Flames room and spots a moving platform that leads to a chest, which contains a powerful sword. After acquiring the sword, Link is able to defeat the final boss of the dungeon. The Legend of Zelda: The Minish Cap is an exceptional spin-off game for the Legend of Zelda series. It takes a humorous approach to dungeon design, and the Cave of Flames room is no exception. It's a challenging puzzle to complete the dungeon, and it's almost always possible to get stuck in a loop, needing to reset all of the hazards multiple times. In addition to puzzle-solving, this minish Cap also highlights one of the major plot points of the Zelda series: the competition between the Seven Maidens of Mome City and the Goddesses of Hyrule. The seven Maidens, whose wish to become beautiful like the Goddesses of Hyrule caused the opening of the Door of Spirits, have been stealing Gods' powers in hopes of becoming as beautiful as the Goddesses. In the Cave of Flames room, one of the Maidens (who was disguised as the Goddess) tricks Link into opening a")
    model_for_classify = '202409230028_subtaskA_monolingual_facebook_opt-1.3b'

    print(f"Classifying the input text: '{sample_text}'")
    result = classify_single_text(sample_text, model_name=model_for_classify)
    print(f"The text is classified as: {result}")

    # Now, explain the classification using LIME
    print("Explaining the classification...")
    explain_text_classification(sample_text)

    # Get Global Feature Importance from the dataset (for global perspective)
    tokenizer, _ = get_pretrained_model()
    global_importance = calculate_global_feature_importance(text_samples, tokenizer)
    print("Global Feature Importance:", global_importance)

if __name__ == "__main__":
    main()
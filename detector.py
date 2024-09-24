import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from peft import get_peft_model, LoraConfig, TaskType
from transformers import XLMRobertaConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig, AutoModelForCausalLM, OPTForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

import sentencepiece

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

import re
import string
import json
from datetime import datetime
import types
import pickle
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
#from transformers import file_utils
#print(file_utils.default_cache_path)
#exit()

app_configs = {}

class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        texts = dataframe.text.values.tolist()

        texts = [self._preprocess(text) for text in texts]

        #self._print_random_samples(texts)

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        #text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        #text = self._lowercase(text)
        text = self._remove_punctuation(text)
        #text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        #text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()


    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    # Stemming (remove -ing, -ly, ...)
    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    # Lemmatisation (convert the word into root word)
    def _lemmatisation(self, text_tokens):
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        return [lem.lemmatize(token) for token in text_tokens]
    
    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label
    
    
#classifier for roberta base,bert, distilbert with 768 neurons on layer 1 and 32 on layer 2
class CustomClassifierBase(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierBase, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # Ensure the attention_mask has the correct shape
        attention_mask = attention_mask.squeeze(1)

        # Pass inputs through the BERT model
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        
        x = self.fc1(bert_out)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x

class CustomLlamaClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomLlamaClassifier, self).__init__()

        self.llama = pretrained_model
        self.fc1 = nn.Linear(pretrained_model.config.hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        # Ensure attention_mask is of the right dimension
        attention_mask = attention_mask[:, None, None, :]  # Add dimensions to match expected 4D input
        
        llama_out = self.llama(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        llama_out = llama_out[:, 0, :]  # Select the first token's output
        
        x = self.fc1(llama_out)
        x = self.relu(x)
        x = self.fc2(x)

        return x

#classifier for roberta large with 1024 neurons on layer 1 and 8 on layer 2
class CustomClassifierRobertaLarge(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierRobertaLarge, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(1024, 8)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x

class CustomClassifier2(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifier2, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(1024, 8)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x
        
class CustomClassifierAlbert(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierAlbert, self).__init__()

        model_configs = AutoConfig.from_pretrained(app_configs['base_model'])
        bert_model = model_configs._name_or_path
        print("Albert model:",bert_model)
        
        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768
            
        print("Albert hidden_size:", hidden_size)            
        self.bert = pretrained_model
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):   
        attention_mask = attention_mask.squeeze(1)     
        print("CustomClassifierAlbertLarge before bert:", input_ids.shape, attention_mask.shape)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        print("albert out");
        bert_out = bert_out[0][:, 0]
        
        print("CustomClassifierAlbertLarge forward:", bert_out)
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x
        
#classifier for roberta base,bert, distilbert with 768 neurons on layer 1 and 32 on layer 2 and 8 on layer 3
class CustomClassifierBase3Layers(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierBase3Layers, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        # x = self.sigmoid(x)

        return x

class CustomOPTClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomOPTClassifier, self).__init__()

        self.opt = pretrained_model
        self.fc1 = nn.Linear(pretrained_model.config.vocab_size, 32)  # Change 2048 to vocab_size (50272)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        # Ensure attention_mask is of the correct shape
        attention_mask = attention_mask.squeeze(1)

        # The output of OPTForCausalLM is a single tensor of logits
        opt_out = self.opt(input_ids=input_ids, attention_mask=attention_mask).logits

        # Select the last token's output
        opt_out = opt_out[:, -1, :]  # Select the last token's output

        x = self.fc1(opt_out)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class DistilbertCustomClassifier(nn.Module):
    def __init__(self,
                 bert_model,
                 num_labels = 1, 
                 bert_hidden_dim=768, 
                 classifier_hidden_dim=32, 
                 dropout=None):
        
        super().__init__()
        self.bert_model = bert_model
        # nn.Identity does nothing if the dropout is set to None
        self.head = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                  nn.Linear(classifier_hidden_dim, num_labels))
    
    def forward(self, input_ids, attention_mask):
        # feeding the input_ids and masks to the model. These are provided by our tokenizer
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state # shape: (batch_size, seq_length, bert_hidden_dim)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation 
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.head(CLS_token_state)
        return logits
            
class AlbertClassifierOld(nn.Module):

    def __init__(self, pretrained_model, freeze_bert=True):
        super(AlbertClassifierOld, self).__init__()
        self.bert_layer = pretrained_model
        
        model_configs = AutoConfig.from_pretrained(app_configs['base_model'])
        bert_model = model_configs._name_or_path
        print(bert_model)
        
        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attn_masks, token_type_ids = None):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits  
                
def str_to_class(s):
    #if s in globals() and isinstance(globals()[s], types.ClassType):
    return globals()[s]
    

def target_device():
    #gpu support for Mac
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()        
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")    
    app_configs['device'] = device
    print("Use device:", device)
    return device

def train(model, train_dataset, val_dataset, learning_rate, epochs, model_name):
    device = target_device()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    training_losses, validation_losses = [], []
    training_accuracies, validation_accuracies = [], []

    for epoch in range(epochs):
        total_loss_train, total_acc_train = 0, 0
        model.train()

        # Training loop with progress bar
        for train_input, train_label in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)
            loss = criterion(output, train_label.float().unsqueeze(1))
            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation loop
        total_loss_val, total_acc_val = 0, 0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)
                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, val_label.float().unsqueeze(1))
                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

        # Calculate average losses and accuracies
        avg_train_loss = total_loss_train / len(train_dataloader)
        avg_val_loss = total_loss_val / len(val_dataloader)
        avg_train_acc = total_acc_train / len(train_dataloader.dataset)
        avg_val_acc = total_acc_val / len(val_dataloader.dataset)

        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        training_accuracies.append(avg_train_acc)
        validation_accuracies.append(avg_val_acc)

        print(f'Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.3f} | Train Acc: {avg_train_acc:.3f} '
              f'| Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc:.3f}')

        # Save the best model based on validation loss
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(app_configs['models_path'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(app_configs['models_path'], f"{model_name}.pt"))
            print("Best model saved.")

    # Plot the training and validation losses and accuracies
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Train Loss')
    plt.plot(validation_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Train Accuracy')
    plt.plot(validation_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_text_predictions(model, loader):
    device = app_configs['device']
    model = model.to(device)
    
    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            output = model(input_ids, attention_mask)
            
            output = (output > 0.5).int()
            results_predictions.append(output)
    
    return torch.cat(results_predictions).cpu().detach().numpy()
    
# def get_pretrained_model():
#     global app_configs
    
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    
#     if (app_configs['base_model'] == 'xlm-roberta-base'):
#         print("load xlm")
#         exit()
#         pretrained_model = AutoModelForMaskedLM.from_pretrained(app_configs['base_model'])
#     else:
#         pretrained_model = AutoModel.from_pretrained(app_configs['base_model'])
        
#     app_configs['tokenizer'] = tokenizer
#     app_configs['pretrained_model'] = pretrained_model
#     return tokenizer, pretrained_model


# Define the LoRA model loader (without quantization)
def get_pretrained_model():
    # Load the tokenizer and add special tokens if needed
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Load the base model without quantization
    pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])

    # Apply LoRA configuration (LoRA without quantization)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Low-rank adaptation
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Get the model with LoRA applied
    model_with_lora = get_peft_model(pretrained_model, lora_config)
    
    return tokenizer, model_with_lora


# def get_pretrained_model():
#     global app_configs
    
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

#     pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])
        
#     app_configs['tokenizer'] = tokenizer
#     app_configs['pretrained_model'] = pretrained_model
#     return tokenizer, pretrained_model
    
  
def get_train_data(train_path, random_seed = 0):
    """
    Function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    

    percentOfData = app_configs['percent_of_data']
    train_df = train_df.sample(int(len(train_df)*percentOfData/100))
    print(len(train_df))
    return train_df

def get_train_val_test_data(full_dataset_path, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=42):
    """
    Function to load the full dataset and split it into train, validation, and test sets.
    """
    full_df = pd.read_json(full_dataset_path, lines=True)

    # Split into training and temp datasets
    train_df, temp_df = train_test_split(full_df, test_size=(val_size + test_size), random_state=random_seed, stratify=full_df['label'])
    
    # Split temp dataset into validation and test datasets
    val_df, test_df = train_test_split(temp_df, test_size=(test_size / (val_size + test_size)), random_state=random_seed, stratify=temp_df['label'])
    
    return train_df, val_df, test_df

def create_and_train():
    train_df, val_df, test_df = get_train_val_test_data(app_configs['full_dataset_path'])

    tokenizer, pretrained_model = get_pretrained_model()

    myModel = CustomOPTClassifier(pretrained_model)
    train_dataset = PreprocessDataset(train_df, tokenizer)
    val_dataset = PreprocessDataset(val_df, tokenizer)

    train(myModel, train_dataset, val_dataset, app_configs['learning_rate'], app_configs['epochs'], app_configs['model_name'])
    print("Training with LoRA completed.")


def creare_train_evaluate_vectorised():
    target_device()    
    tokenizer, pretrained_model = get_pretrained_model()
    
    # Construirea unui pipeline care include obținerea reprezentărilor de la BERT, extragerea de caracteristici cu TF-IDF și LinearSVC
    #classifier = Pipeline([
    #    ('bert_embeddings', FunctionTransformer(get_bert_embeddings)),
    #    ('clf', LinearSVC())
    #])
    # Construirea unui pipeline care include tokenizarea BERT, extragerea de caracteristici cu TF-IDF și LinearSVC
    classifier = Pipeline([
        ('tokenizer', tokenizer),
        ('vectorizer', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
        ('clf', LinearSVC())
    ])

    train_df = get_train_data(app_configs['train_path'])    
    
    train_preprocessed = PreprocessDataset(train_df, tokenizer)
    print(train_preprocessed)
    exit()
    # Antrenarea modelului pe datele de antrenare
    classifier.fit(X_train, y_train)

def load_and_evaluate(model_name=''):
    if model_name:
        app_configs['model_name'] = model_name

    # Load the full dataset and split into train, validation, and test sets
    _, _, test_df = get_train_val_test_data(app_configs['full_dataset_path'])

    # Drop unnecessary columns
    test_df = test_df.drop(["model", "source"], axis=1)

    # Initialize the model architecture first
    target_device()
    tokenizer, pretrained_model = get_pretrained_model()

    # Recreate the model architecture
    classifierClass = str_to_class(app_configs['classifier'])
    model = classifierClass(pretrained_model)

    # Load the saved state_dict into the model architecture
    model_path = app_configs['models_path'] + app_configs['model_name'] + ".pt"
    model.load_state_dict(torch.load(model_path))  # Loading the state dict properly
    print("Model loaded successfully from:", model_path)

    # Move the model to the correct device
    model = model.to(app_configs['device'])

    predictions_df = pd.DataFrame({'id': test_df['id']})
    test_dataloader = DataLoader(PreprocessDataset(test_df, tokenizer), batch_size=8, shuffle=False, num_workers=0)
    predictions_df['label'] = get_text_predictions(model, test_dataloader)

    predictions_df.to_json(app_configs['prediction_path'], lines=True, orient='records')
    merged_df = predictions_df.merge(test_df, on='id', suffixes=('_pred', '_gold'))
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    app_configs['accuracy'] = accuracy
    print("Accuracy:", accuracy)
    print(classification_report(merged_df['label_gold'], merged_df['label_pred']))

    # Compute and display the confusion matrix
    cm = confusion_matrix(merged_df['label_gold'], merged_df['label_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    if not model_name:
        save_app_options()

def load_app_options(model_name):
    options = pd.read_json(train_path, lines=True)
    
def save_app_options():
    configs = app_configs.copy()
    configs_keys = configs.keys()
    
    keys_2_del = {'tokenizer', 'pretrained_model', 'prediction_path', 'results_path', 'options_path', 'options_path', 'classifier', 'device'}
    for del_key in keys_2_del:
        configs.pop(del_key, None)
           
        
    # Writing to sample.json
    with open(app_configs['options_path'], "w") as outfile:
        json.dump(configs, outfile)
        
# datetime object containing current date and time
start_now = datetime.now()
start_time = start_now.strftime("%Y-%m-%d %H-%M")
timestamp_prefix = start_now.strftime("%Y%m%d%H%M")
print("process start at:", start_time)
torch.manual_seed(0)
np.random.seed(0)

absolute_path = os.path.abspath('/Users/rusinitharagunarathne/Documents/AletheiaAI/data')

opt_model_configs = {
    'base_model': 'facebook/opt-1.3b',  # Replace with the specific OPT model you want
    'classifier': 'CustomOPTClassifier',
    'learning_rate': 1e-5,
}

llama_model_configs1 = {
    'base_model': 'meta-llama/Llama-2-7b-hf',  # Adjust the model name as per your requirement
    'classifier': 'CustomLlamaClassifier',
    'learning_rate': 1e-5,
}

distilbert_model_configs1 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase',
}

distilbert_model_configs2 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase3Layers',
    'learning_rate': 2e-5,
}

robertabase_model_configs1 = {
    'base_model': 'roberta-base',
    'classifier': 'CustomClassifierBase', 
    'test_path': absolute_path + '/subtaskA_monolingual_gold.jsonl',
}

robertalarge_model_configs1 = {
    'base_model': 'roberta-large',    
    'classifier': 'CustomClassifierRobertaLarge', 
    # 'percent_of_data': 15,
    'learning_rate': 1e-5,
}

bert_model_configs1 = {
    'base_model': 'bert-base-uncased',
    'classifier': 'CustomClassifierBase',
}

albert_model_configs1 = {
    'base_model': 'albert-base-v2',
    'classifier': 'CustomClassifierAlbert',
}

default_configs = {
    'learning_rate': 1e-5,
    'epochs': 5,
    'task': 'subtaskA_monolingual',
    'timestamp_prefix': timestamp_prefix,
    'full_dataset_path': absolute_path + '/subtaskA_dataset_monolingual.jsonl',
    'percent_of_data': 100,
    'options_path': absolute_path + '/predictions/'  + 'tests.results.jsonl',
    'models_path':  absolute_path + '/models/',
}


app_configs = default_configs.copy()
app_configs.update(opt_model_configs)

app_configs['model_name'] = app_configs['timestamp_prefix'] + "_" + app_configs['task'] + "_" + app_configs['base_model'].replace("/", "_")
app_configs['prediction_path'] = absolute_path + '/predictions/' + app_configs['model_name'] + '.predictions.jsonl'
app_configs['options_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.options.jsonl'
app_configs['results_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.results.jsonl'

print("Working on pretrained-model:", app_configs['base_model'])

#model names that can be used for evaluation:
#model name roberta-large trained = 202401112145_subtaskA_monolingual_roberta-large
#model name roberta-base trained = 202402061024_subtaskA_monolingual_roberta
#model name for distilbert-base-uncased trained = 202401120919_subtaskA_monolingual_distilbert-base-uncased - 2 layers

#multilang tests
#model name for xlm_roberta_base = 202401201729_subtaskA_multilingual_FacebookAI_xlm-roberta-base
#model name for distilbert-base-multilingual-cased = 202401231736_subtaskA_monolingual_distilbert-base-multilingual-cased.options.jsonl
#creare_train_evaluate_vectorised()
#exit()
#model_for_evaluate='202402061024_subtaskA_monolingual_roberta-base'
model_for_evaluate='202409230028_subtaskA_monolingual_facebook_opt-1.3b'
# create_and_train()
# load_and_evaluate(model_for_evaluate)

# Conditional logic for training or evaluating
if model_for_evaluate:
    print(f"Evaluating model: {model_for_evaluate}")
    load_and_evaluate(model_for_evaluate)
else:
    print("Training a new model...")
    create_and_train()

end_now = datetime.now()
end_time = end_now.strftime("%Y-%m-%d %H-%M")
print("process finished at:", end_time)
running_time = (end_now - start_now).total_seconds()
app_configs['start_time'] = start_time
app_configs['end_time'] = end_time
app_configs['running_time'] = running_time
if (model_for_evaluate == ''): 
    save_app_options()
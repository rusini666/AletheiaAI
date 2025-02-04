import random
import string
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import custom modules
from detector import app_configs, PreprocessDataset, CustomOPTClassifier, get_train_val_test_data, load_and_evaluate

# Ensure device is set
app_configs['device'] = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {app_configs['device']}")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])
classifier = CustomOPTClassifier(pretrained_model).to(app_configs['device'])

# Load test dataset but do NOT evaluate it yet
_, _, df_test = get_train_val_test_data(app_configs['full_dataset_path'])
print(f"✅ Test dataset loaded with {len(df_test)} samples (Skipping initial evaluation).")

# Ensure dataset contains the required columns
if "text" not in df_test.columns or "label" not in df_test.columns:
    raise KeyError("Dataset must contain 'text' and 'label' columns.")

# Select a random subset of test samples for robustness testing
test_subset = df_test.sample(n=100, random_state=42)
test_texts = test_subset["text"].tolist()
true_labels = test_subset["label"].tolist()

# ✅ Run Robustness Testing First
print("🚀 Running **ROBUSTNESS TESTING FIRST**...")

def add_noise_to_text(text, noise_level=0.2):
    """ Introduces noise to text by adding typos, removing punctuation, and shuffling words. """
    words = text.split()
    num_words = len(words)
    num_noisy_words = max(1, int(num_words * noise_level))

    for _ in range(num_noisy_words):
        word_idx = random.randint(0, num_words - 1)
        noise_type = random.choice(["typo", "remove_punc", "shuffle"])

        if noise_type == "typo" and len(words[word_idx]) > 1:
            char_list = list(words[word_idx])
            swap_idx = random.randint(0, len(char_list) - 2)
            char_list[swap_idx], char_list[swap_idx + 1] = char_list[swap_idx + 1], char_list[swap_idx]
            words[word_idx] = "".join(char_list)

        elif noise_type == "remove_punc":
            words[word_idx] = words[word_idx].translate(str.maketrans('', '', string.punctuation))

        elif noise_type == "shuffle" and num_words > 1:
            swap_idx1, swap_idx2 = random.sample(range(num_words), 2)
            words[swap_idx1], words[swap_idx2] = words[swap_idx2], words[swap_idx1]

    return " ".join(words)

def evaluate_model_robustness(test_texts, true_labels, classifier, tokenizer):
    """ Evaluates model performance on clean and noisy data to measure robustness. """
    clean_preds, noisy_preds = [], []

    for text in test_texts:
        # Tokenize clean text
        encoded_text = tokenizer(text, padding="max_length", truncation=True, max_length=150, return_tensors="pt")
        input_ids = encoded_text["input_ids"].to(app_configs["device"])
        attention_mask = encoded_text["attention_mask"].to(app_configs["device"])

        if input_ids.dim() == 1:  # Ensure correct shape for OPT
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        with torch.no_grad():
            logits = classifier(input_ids, attention_mask)

        pred_score = torch.sigmoid(logits).item()
        clean_preds.append(1 if pred_score > 0.5 else 0)

        # Generate noisy text and preprocess it
        noisy_text = add_noise_to_text(text, noise_level=0.2)
        encoded_noisy = tokenizer(noisy_text, padding="max_length", truncation=True, max_length=150, return_tensors="pt")
        input_ids = encoded_noisy["input_ids"].to(app_configs["device"])
        attention_mask = encoded_noisy["attention_mask"].to(app_configs["device"])

        if input_ids.dim() == 1:  # Ensure correct shape for OPT
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        with torch.no_grad():
            logits = classifier(input_ids, attention_mask)

        pred_score = torch.sigmoid(logits).item()
        noisy_preds.append(1 if pred_score > 0.5 else 0)

    # Compute performance metrics
    clean_acc = accuracy_score(true_labels, clean_preds)
    noisy_acc = accuracy_score(true_labels, noisy_preds)
    clean_f1 = f1_score(true_labels, clean_preds)
    noisy_f1 = f1_score(true_labels, noisy_preds)

    print(f"✅ Clean Data - Accuracy: {clean_acc:.4f}, F1-score: {clean_f1:.4f}")
    print(f"⚠️ Noisy Data - Accuracy: {noisy_acc:.4f}, F1-score: {noisy_f1:.4f}")
    print(f"📉 Performance Drop - Accuracy: {clean_acc - noisy_acc:.4f}, F1-score: {clean_f1 - noisy_f1:.4f}")

    return {
        "Clean Accuracy": clean_acc,
        "Noisy Accuracy": noisy_acc,
        "Accuracy Drop": clean_acc - noisy_acc,
        "Clean F1": clean_f1,
        "Noisy F1": noisy_f1,
        "F1 Drop": clean_f1 - noisy_f1,
        "Clean Predictions": clean_preds,
        "Noisy Predictions": noisy_preds
    }

# Generate performance comparison bar chart
def plot_performance_comparison(results):
    categories = ["Accuracy", "F1-score"]
    clean_values = [results["Clean Accuracy"], results["Clean F1"]]
    noisy_values = [results["Noisy Accuracy"], results["Noisy F1"]]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, clean_values, width, label="Clean Data", color="blue")
    ax.bar(x + width/2, noisy_values, width, label="Noisy Data", color="red")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance on Clean vs Noisy Data")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.show()

# Plot confusion matrices
def plot_confusion_matrices(y_true, y_clean_pred, y_noisy_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix for clean data
    cm_clean = confusion_matrix(y_true, y_clean_pred)
    disp_clean = ConfusionMatrixDisplay(confusion_matrix=cm_clean, display_labels=["Human", "AI"])
    disp_clean.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix - Clean Data")

    # Confusion matrix for noisy data
    cm_noisy = confusion_matrix(y_true, y_noisy_pred)
    disp_noisy = ConfusionMatrixDisplay(confusion_matrix=cm_noisy, display_labels=["Human", "AI"])
    disp_noisy.plot(ax=axes[1], cmap=plt.cm.Reds)
    axes[1].set_title("Confusion Matrix - Noisy Data")

    plt.show()

# 🚀 Run Robustness Testing First
robustness_results = evaluate_model_robustness(test_texts, true_labels, classifier, tokenizer)

# ✅ **Generate Plots After Robustness Testing**
print("📊 Generating Performance Comparison Chart...")
plot_performance_comparison(robustness_results)

print("📊 Generating Confusion Matrices...")
plot_confusion_matrices(true_labels, robustness_results["Clean Predictions"], robustness_results["Noisy Predictions"])

# ✅ Now Run Standard Test Dataset Evaluation
print("\n🚀 Now running standard test dataset evaluation after robustness testing...\n")
load_and_evaluate(app_configs['model_name'])

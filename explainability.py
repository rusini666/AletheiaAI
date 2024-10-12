import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from detector import app_configs, CustomOPTClassifier, get_pretrained_model

# Set device to CPU to avoid memory overload on MPS
device = "cpu"

# Load tokenizer and model
tokenizer, model_with_lora = get_pretrained_model()

# Move model to CPU (or MPS if you think memory is sufficient)
model_with_lora.to(device)

# Define LIME explainer
explainer = LimeTextExplainer(class_names=['AI-generated', 'Human-written'])

# Preprocessing function
def preprocess_for_lime(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Prediction function for LIME
def predict_proba(texts):
    model_with_lora.eval()  # Set model to evaluation mode
    
    inputs = tokenizer(texts, padding='max_length', max_length=50, truncation=True, return_tensors="pt")  # Reduced max_length to reduce memory
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model_with_lora(input_ids, attention_mask)  # Get full model outputs
        logits = outputs.logits  # Extract the logits (the predictions)
        probs = torch.sigmoid(logits[:, -1, :]).cpu().numpy()  # Apply sigmoid to the logits and get probabilities

    return np.hstack([1 - probs, probs])  # Return both class probabilities

def generate_custom_explanation(text_instance, explanation):
    # Get the top influential words and their weights
    influential_words = explanation.as_list()
    
    # Separate words into positive and negative contributions
    positive_contributors = [word for word, weight in influential_words if weight > 0]
    negative_contributors = [word for word, weight in influential_words if weight < 0]
    
    # Get confidence score (local prediction) from LIME
    confidence = explanation.local_pred[0]

    # Create a dynamic explanation
    explanation_text = ""
    
    # Add a section for confidence
    if confidence > 0.75:
        explanation_text += "The model is highly confident that this text is AI-generated.\n"
    elif confidence < 0.25:
        explanation_text += "The model is highly confident that this text is human-written.\n"
    else:
        explanation_text += "The model's confidence is moderate for this prediction.\n"
    
    # Add a section for positive contributors
    if positive_contributors:
        explanation_text += f"Key words that supported this prediction include: {', '.join([word for word, _ in positive_contributors])}.\n"
    
    # Add a section for negative contributors
    if negative_contributors:
        explanation_text += f"However, words like {', '.join([word for word, _ in negative_contributors])} pushed the prediction towards the other class.\n"
    
    # Add dynamic reasoning based on which words had the strongest influence
    if len(positive_contributors) > len(negative_contributors):
        explanation_text += "Overall, the system found more evidence supporting the predicted class."
    else:
        explanation_text += "The system found some conflicting evidence, but the predicted class was stronger overall."
    
    return explanation_text

# Generate LIME explanation
def explain_text(text_instance):
    preprocessed_text = preprocess_for_lime(text_instance)
    explanation = explainer.explain_instance(preprocessed_text, predict_proba, num_features=6, num_samples=500)  # Reduced num_samples to save memory
    
    # Plot the explanation using matplotlib (this works in VS Code)
    fig = explanation.as_pyplot_figure()
    plt.show()

# Example text
if __name__ == "__main__":
    text_instance = "I hope you are doing well. I am writing to inquire about the timetable for the part-time Software Engineering (SE) students. We have not yet received any information regarding the schedule for this academic term, and I would appreciate it if you could provide an update on when we can expect the timetable to be shared."
    explain_text(text_instance)
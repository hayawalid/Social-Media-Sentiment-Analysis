import os
from pathlib import Path
import numpy as np
import torch  # Use PyTorch instead of TensorFlow
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_DIR = Path(__file__).resolve().parent / "best model"

def preprocess_social_text(text: str) -> str:
    clean_tokens = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        clean_tokens.append(token)
    return " ".join(clean_tokens)

def load_model_bundle():
    # Load strictly locally from your disk folder using PyTorch
    print("Loading compiled local assets completely offline via PyTorch...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    return tokenizer, model

def print_ranked_sentiment(text: str, tokenizer, model):
    clean_text = preprocess_social_text(text)
    
    # Tokenize and convert to PyTorch tensors ('pt')
    encoded_input = tokenizer(clean_text, return_tensors="pt", truncation=True)
    
    # Run inference without tracking gradients (saves memory/time on CPU)
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Extract logits from PyTorch output array
    logits = output.logits.numpy()[0]
    scores = softmax(logits)
    ranking = np.argsort(scores)[::-1]

    id2label = getattr(model.config, "id2label", {}) or {}
    for index in range(scores.shape[0]):
        label_index = ranking[index]
        label = id2label.get(str(label_index), id2label.get(label_index, f"LABEL_{label_index}"))
        score = scores[label_index]
        print(f"{index + 1}) {label} {np.round(float(score), 4)}")

def main():
    # Make sure you have downloaded pytorch_model.bin into the folder first!
    tokenizer, model = load_model_bundle()
    sample_text = "I love this product! It's amazing and works perfectly. Highly recommend it to everyone. #bestpurchaseever"
    print_ranked_sentiment(sample_text, tokenizer, model)

if __name__ == "__main__":
    main()

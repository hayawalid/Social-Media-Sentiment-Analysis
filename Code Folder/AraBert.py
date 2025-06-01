import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
import optuna
from datasets import Dataset as HFDataset, DatasetDict
import transformers
assert int(transformers.__version__.split('.')[0]) >= 4, "Transformers version too old. Please upgrade."



# Check if GPU is available
# import torch
# import os


# print(f"CUDA Available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# # At the top of your script
# with open("gpu_log.txt", "w") as f:
#     f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
#     if torch.cuda.is_available():
#         f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
#         f.write("Using GPU\n")
#     else:
#         f.write("No GPU available, using CPU\n")

# # Check if GPU is available
# print(f"CUDA Available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
#     # Force PyTorch to use GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     device = torch.device("cuda")
# else:
#     print("No GPU available, using CPU")
#     device = torch.device("cpu")

# print(f"Using device: {device}")

# print(torch.cuda.current_device())  # Should return 0
# print(torch.cuda.is_initialized())  # Should return True

# Load the data
df = pd.read_csv('stemmed_output.csv')

# Convert label: 1 -> 0 (negative), 2 -> 1 (positive)
df['label'] = df['label'].replace({1: 0, 2: 1})

# Load tokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Train/val/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['stemmedtext'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

train_dataset = HFDataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
val_dataset = HFDataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})
test_dataset = HFDataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})

tokenized_datasets = DatasetDict({
    "train": train_dataset.map(tokenize_function, batched=True),
    "validation": val_dataset.map(tokenize_function, batched=True),
    "test": test_dataset.map(tokenize_function, batched=True),
})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Optuna Objective
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def objective(trial):
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",  # Keep or change to save_steps if needed
        learning_rate=trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        num_train_epochs=10,
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.3),
        logging_dir="./logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Train final model with best parameters
best_params = study.best_trial.params

training_args = TrainingArguments(
    output_dir="./best_model",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",  # Keep or change to save_steps if needed
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    num_train_epochs=10,
    weight_decay=best_params["weight_decay"],
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model("best_arabert_model")

# Evaluate on test set
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Metrics
acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set")
plt.show()

# Plot training/validation loss
logs = trainer.state.log_history
train_loss = [log["loss"] for log in logs if "loss" in log and "eval_loss" not in log]
eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.show()
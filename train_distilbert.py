import os
import json
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Device
device = torch.device("cpu")
print("Training device:", device)

# Load dataset
df = pd.read_csv("data/intents.csv")

labels = sorted(df["intent"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["intent"].map(label2id)

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["query"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Load tokenizer
model_name = "distilbert/distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Tokenize data
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=64
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=64
)

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_encodings, train_labels)
test_dataset = IntentDataset(test_encodings, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
)

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

model.to(device)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

# Training
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Training Loss: {avg_loss:.4f}"
    )

# Evaluation
model.eval()

predictions = []
actual_labels = []

with torch.no_grad():
    for batch in test_loader:
        labels_batch = batch["labels"].to(device)

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(**inputs)

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels_batch.cpu().numpy())

accuracy = accuracy_score(
    actual_labels,
    predictions
)

print("\nDistilBERT Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        actual_labels,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    )
)

# Save model
os.makedirs("models/distilbert", exist_ok=True)

model.save_pretrained("models/distilbert")
tokenizer.save_pretrained("models/distilbert")

with open("models/distilbert/labels.json", "w") as f:
    json.dump(label2id, f)

print("\nModel saved successfully!")
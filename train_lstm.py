
import re
import pandas as pd
import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from models.lstm_model import LSTMIntentClassifier

torch.manual_seed(42)

# 1. Load dataset
df = pd.read_csv("data/intents_improved.csv")
df = df.dropna(subset=["text", "intent"])
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

# 2. Tokenize sentences
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_sentences = df["text"].apply(tokenize).tolist()

# 3. Encode intent labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["intent"])

# 4. Split dataset
X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
    tokenized_sentences,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Build vocabulary using training data only
word_counts = Counter(
    word
    for sentence in X_train_tokens
    for word in sentence
)

word_to_index = {"<PAD>": 0, "<UNK>": 1}

for word in word_counts:
    word_to_index[word] = len(word_to_index)

# 6. Convert words into numerical sequences
def encode_sentence(tokens):
    return [
        word_to_index.get(word, word_to_index["<UNK>"])
        for word in tokens
    ]

X_train = [encode_sentence(s) for s in X_train_tokens]
X_test = [encode_sentence(s) for s in X_test_tokens]

# 7. Pad sequences
max_length = max(len(s) for s in X_train)

def pad_sequences(sequences):
    padded = []

    for sequence in sequences:
        sequence = sequence[:max_length]
        sequence = sequence + [0] * (max_length - len(sequence))
        padded.append(sequence)

    return torch.tensor(padded, dtype=torch.long)

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 8. Create LSTM model
model = LSTMIntentClassifier(
    vocab_size=len(word_to_index),
    embedding_dim=64,
    hidden_size=64,
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Train model
epochs = 100

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

# 10. Evaluate model
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    predictions = outputs.argmax(dim=1).numpy()

accuracy = accuracy_score(y_test.numpy(), predictions)

print("\nLSTM EVALUATION RESULTS")
print("=======================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test.numpy(),
    predictions,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))

# 11. Save trained model
torch.save(
    {
        "model_state": model.state_dict(),
        "word_to_index": word_to_index,
        "encoder": encoder,
        "vocab_size": len(word_to_index),
        "embedding_dim": 64,
        "hidden_size": 64,
        "num_classes": len(encoder.classes_),
        "max_length": max_length
    },
    "models/lstm_checkpoint.pt"
)

print("\nLSTM model saved successfully!")

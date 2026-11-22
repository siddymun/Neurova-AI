
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from models.fnn_model import FNNIntentClassifier

# 1. Load dataset
df = pd.read_csv("data/intents_improved.csv")

print("CSV columns:", df.columns.tolist())
print(df.head())

df = df.dropna(subset=["text", "intent"])
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

# Remove empty rows
df = df[(df["text"] != "") & (df["intent"] != "")]

# 2. Encode intent labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["intent"])

# 3. Split raw text BEFORE fitting TF-IDF
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"],
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Convert text into numerical features
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# 5. Convert data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 6. Create FNN model
model = FNNIntentClassifier(
    input_size=X_train.shape[1],
    hidden_size=64,
    num_classes=len(encoder.classes_)
)

# 7. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 8. Train the model
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

# 9. Evaluate the model
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    predictions = outputs.argmax(dim=1).numpy()

accuracy = accuracy_score(y_test.numpy(), predictions)

print("\nFNN Evaluation Results")
print("----------------------")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test.numpy(),
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0
    )
)

# 10. Save model and preprocessing objects
torch.save(
    {
        "model_state": model.state_dict(),
        "vectorizer": vectorizer,
        "encoder": encoder,
        "input_size": X_train.shape[1]
    },
    "models/fnn_checkpoint_improved.pt"
)

print("\nFNN model saved successfully!")

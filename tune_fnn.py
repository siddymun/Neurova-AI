
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from models.fnn_model import FNNIntentClassifier

# Reproducibility
torch.manual_seed(42)

# Load improved dataset
df = pd.read_csv("data/intents_improved.csv")
df = df.dropna(subset=["text", "intent"])
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

# Encode intent labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["intent"])

# Use the same split for every experiment
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"],
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Hyperparameter combinations
hidden_sizes = [32, 64, 128]
learning_rates = [0.01, 0.001, 0.0001]

results = []

for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:

        torch.manual_seed(42)

        model = FNNIntentClassifier(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            num_classes=len(encoder.classes_)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # Train model
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).argmax(dim=1).numpy()
            accuracy = accuracy_score(y_test, predictions)

        results.append({
            "Hidden Neurons": hidden_size,
            "Learning Rate": learning_rate,
            "Accuracy": accuracy * 100
        })

        print(
            f"Hidden Neurons: {hidden_size}, "
            f"Learning Rate: {learning_rate}, "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy", ascending=False
)

print("\nHYPERPARAMETER TUNING RESULTS")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("fnn_tuning_results.csv", index=False)

# Create comparison chart
plt.figure(figsize=(10, 6))

labels = [
    f"Hidden={row['Hidden Neurons']}\nLR={row['Learning Rate']}"
    for _, row in results_df.iterrows()
]

plt.bar(labels, results_df["Accuracy"])
plt.title("FNN Hyperparameter Tuning")
plt.xlabel("Hyperparameter Configuration")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("fnn_tuning_comparison.png")
plt.show()

print("\nResults saved to fnn_tuning_results.csv")
print("Chart saved to fnn_tuning_comparison.png")

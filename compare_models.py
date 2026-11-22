
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from models.fnn_model import FNNIntentClassifier

# Reproducibility
torch.manual_seed(42)

# 1. Load improved dataset
df = pd.read_csv("data/intents_improved.csv")
df = df.dropna(subset=["text", "intent"])

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

# 2. Encode intent labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["intent"])

# 3. Split dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"],
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. TF-IDF feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

results = {}

# 5. Logistic Regression
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)
logistic_predictions = logistic.predict(X_test)

results["Logistic Regression"] = accuracy_score(
    y_test, logistic_predictions
)

# 6. Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
nb_predictions = naive_bayes.predict(X_test)

results["Naive Bayes"] = accuracy_score(
    y_test, nb_predictions
)

# 7. Support Vector Machine
svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

results["SVM"] = accuracy_score(
    y_test, svm_predictions
)

# 8. Prepare tensors for FNN
X_train_tensor = torch.tensor(
    X_train.toarray(), dtype=torch.float32
)
X_test_tensor = torch.tensor(
    X_test.toarray(), dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train, dtype=torch.long
)
y_test_tensor = torch.tensor(
    y_test, dtype=torch.long
)

# 9. Create and train FNN
fnn = FNNIntentClassifier(
    input_size=X_train_tensor.shape[1],
    hidden_size=64,
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    fnn.parameters(), lr=0.001
)

for epoch in range(100):
    fnn.train()

    optimizer.zero_grad()
    outputs = fnn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

# 10. Evaluate FNN
fnn.eval()

with torch.no_grad():
    outputs = fnn(X_test_tensor)
    fnn_predictions = outputs.argmax(dim=1).numpy()

results["FNN"] = accuracy_score(
    y_test, fnn_predictions
)

# 11. Display comparison
print("\nMODEL COMPARISON RESULTS")
print("========================")

for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")

# 12. Detailed FNN evaluation
print("\nFNN CLASSIFICATION REPORT")
print("=========================")

print(classification_report(
    y_test,
    fnn_predictions,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))

# 13. Plot comparison
plt.figure(figsize=(9, 5))

names = list(results.keys())
scores = [score * 100 for score in results.values()]

plt.bar(names, scores)
plt.title("Traditional ML vs FNN Accuracy")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()

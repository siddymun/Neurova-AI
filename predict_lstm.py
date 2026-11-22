
import re
import torch

from models.lstm_model import LSTMIntentClassifier


# Select CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the trained LSTM checkpoint
checkpoint = torch.load(
    "models/lstm_checkpoint.pt",
    map_location=device,
    weights_only=False
)


# Restore model settings
word_to_index = checkpoint["word_to_index"]
encoder = checkpoint["encoder"]
max_length = checkpoint["max_length"]

model = LSTMIntentClassifier(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_size=checkpoint["hidden_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()


# Convert text into word IDs
def preprocess_text(text):

    tokens = re.findall(r"\b\w+\b", text.lower())

    sequence = [
        word_to_index.get(word, word_to_index["<UNK>"])
        for word in tokens
    ]

    # Truncate long sentences
    sequence = sequence[:max_length]

    # Pad shorter sentences
    sequence += [word_to_index["<PAD>"]] * (
        max_length - len(sequence)
    )

    return torch.tensor([sequence], dtype=torch.long).to(device)


# Predict user intent
def predict_lstm(text):

    if not text or not text.strip():
        raise ValueError("Please enter a message.")

    input_tensor = preprocess_text(text)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    intent = encoder.inverse_transform(
        [predicted_class.item()]
    )[0]

    confidence = confidence.item()

    return intent, confidence

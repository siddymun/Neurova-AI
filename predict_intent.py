import torch
from models.fnn_model import FNNIntentClassifier

CHECKPOINT_PATH = "models/fnn_checkpoint_improved.pt"


def load_intent_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    vectorizer = checkpoint["vectorizer"]
    encoder = checkpoint["encoder"]

    model = FNNIntentClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=64,
        num_classes=len(encoder.classes_)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vectorizer, encoder


def predict_intent(text):
    model, vectorizer, encoder = load_intent_model()

    features = vectorizer.transform([text]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    return {
        "intent": encoder.inverse_transform(
            [predicted.item()]
        )[0],
        "confidence": round(confidence.item() * 100, 2)
    }


if __name__ == "__main__":
    message = input("Enter a message: ")
    result = predict_intent(message)

    print("Predicted intent:", result["intent"])
    print("Confidence:", f'{result["confidence"]}%')

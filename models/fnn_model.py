
import torch
import torch.nn as nn


class FNNIntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = FNNIntentClassifier(
        input_size=100,
        hidden_size=64,
        num_classes=5
    )

    sample = torch.randn(1, 100)
    output = model(sample)

    print("FNN model created successfully!")
    print("Output shape:", output.shape)

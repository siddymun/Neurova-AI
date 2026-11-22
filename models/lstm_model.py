
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMIntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes,
        padding_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        # Calculate actual sentence lengths
        lengths = (x != 0).sum(dim=1)

        # Convert word IDs into embeddings
        embedded = self.embedding(x)

        # Ignore padded positions
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Process sentence sequences
        _, (hidden, cell) = self.lstm(packed)

        # Get final hidden representation
        last_hidden = hidden[-1]

        # Apply dropout
        dropped = self.dropout(last_hidden)

        # Predict intent
        return self.fc(dropped)

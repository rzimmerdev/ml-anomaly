import torch
import torch.nn as nn

class CNNTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes, dropout_rate=0.1):
        super(CNNTransformerModel, self).__init__()

        # 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Transformer
        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Apply 1D CNN
        x = self.cnn(x)

        # Reshape for transformer
        x = x.permute(0, 2, 1)  # Change the shape to (batch_size, features, sequence_length)
        x = self.embedding(x)

        # Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate along the sequence dimension

        # Fully connected layer
        x = self.dropout(x)
        output = self.fc(x)

        return output

# Example usage
input_size = 6  # Number of features (gyroscope X, Y, Z, accelerometer X, Y, Z)
hidden_size = 64
num_heads = 4
num_layers = 2
num_classes = 5  # Multiclass classification task with 5 labels

# Create an instance of the model
model = CNNTransformerModel(input_size, hidden_size, num_heads, num_layers, num_classes)

# Sample input (batch size=1, sequence length=500, features=6)
sample_input = torch.randn(1, 6, 500)

# Forward pass
output = model(sample_input)
print(output)


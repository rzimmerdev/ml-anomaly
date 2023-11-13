import torch
import torch.nn as nn

class MultiClassAnomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes, dropout_rate=0.1):
        super(MultiClassAnomaly, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        output = self.fc(x)

        return output


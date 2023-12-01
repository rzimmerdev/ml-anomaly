import torch
import torchmetrics
from torch import nn
import lightning


class MultiClassAnomaly(lightning.LightningModule):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes, dropout_rate=0.5):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout_rate,
                activation='relu'
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = self.classifier(x)
        return nn.functional.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_accuracy', acc, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        y = torch.argmax(y, dim=1)
        acc = self.accuracy(pred, y)

        cm = torchmetrics.functional.confusion_matrix(pred, y, task='multiclass', num_classes=self.num_classes)
        # Can't log tensors, and cm is multiclass, so have to log each class separately
        self.log('loss', loss, on_epoch=True, on_step=False)
        self.log('accuracy', acc, on_epoch=True, on_step=False)

        for i in range(self.num_classes):
            false_positives = torch.sum(cm[:, i]) - cm[i, i]
            false_negatives = torch.sum(cm[i, :]) - cm[i, i]
            true_positives = cm[i, i]
            true_negatives = torch.sum(cm) - (false_positives + false_negatives + true_positives)

            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            self.log(f'precision_{i}', precision, on_epoch=True, on_step=False)
            self.log(f'recall_{i}', recall, on_epoch=True, on_step=False)
            self.log(f'f1_{i}', f1, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

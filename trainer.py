import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class Trainer(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, test_loader):
        super(CNNTransformerTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_accuracy', acc, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_accuracy', acc, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


### Using a pretrained backbone ref. https://huggingface.co/docs/timm/v0.9.16/en/reference/models#timm.create_model

import torch.nn as nn
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import timm
import lightning.pytorch as pl


class ImageClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for image classification using a model from timm library.
    """

    def __init__(self, config):
        super(ImageClassifier, self).__init__()
        # Using a pretrained backbone
        self.model = timm.create_model(
            config["model_name"], pretrained=True, num_classes=1
        )

        self.lr = config["lr"]
        self.optimizer_type = config["optimizer"]

        if config["optimizer"] == "adam":
            self.optimizer = Adam
        else:
            self.optimizer = SGD

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(num_classes=1, task="binary")

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y.float())
        self.log("loss", loss, on_step=True, prog_bar=True)

        acc = self.accuracy(outputs, y.long())
        self.log("acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y.float())
        self.log("val_loss", loss, on_step=True, prog_bar=True)

        acc = self.accuracy(outputs, y.long())
        self.log("val_acc", acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

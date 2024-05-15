import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L

from torch.optim import Adam
from torchmetrics import Accuracy
from torchvision import models

from einops import rearrange
from dataloader import PETADataset

import timm

from ImageClassifier import ImageClassifier


def show_originalimage(image):
    image = torch.clamp(image, -1, 1)
    img = image.cpu().numpy().copy()
    img *= np.array([0.229, 0.224, 0.225])[:, None, None]
    img += np.array([0.485, 0.456, 0.406])[:, None, None]

    img = rearrange(img, "c h w -> h w c")
    img = img * 255
    img = img.astype(np.uint8)
    return img


# class ImageClassifier(L.LightningModule):
#     def __init__(self, config):
#         super(ImageClassifier, self).__init__()
#         self.lf = 1e-4

#         self.criterion = torch.nn.BCEWithLogitsLoss()

#         self.optimizer = Adam
#         self.acc = Accuracy(task="binary", num_classes=1)

#         # Using a pretrained backbone
#         ## ref. https://huggingface.co/docs/timm/v0.9.16/en/reference/models#timm.create_model
#         self.model = timm.create_model(
#             config["model_name"], pretrained=True, num_classes=1
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x.squeeze(-1)  # Change the output dimension from [48, 1] to [48]

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self.forward(x)
#         loss = self.criterion(outputs, y.float())
#         self.log("loss", loss, on_step=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         self.val_origin_image = x
#         self.val_output = self.forward(x)
#         loss = self.criterion(self.val_output, y.float())
#         self.log("val_loss", loss, on_step=True, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return self.optimizer(self.model.parameters(), lr=self.lf)


if __name__ == "__main__":
    ######   wandb   ######
    wandb_logger = WandbLogger(project="spacevision_classification")

    ### config
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    ###### dataload  ######
    train_dataset = PETADataset(True, cfg["data"], augmentation=True)
    valid_dataset = PETADataset(False, cfg["data"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    ###########################

    model_save_path = cfg["model_save_path"]
    os.makedirs(model_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path)

    # callback for save model every 10000 steps donot overwrite
    setp_checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="step_{step}",
        every_n_train_steps=25_000,
        save_top_k=-1,
    )

    models = [
        # "resnet50",
        # "vgg16",
        # "convnext_base",
        "convnext_nano",
        # "vit_small_patch8_224_dino",
        # "deit_small_patch16_224",
    ]
    config = {
        "model_name": models[0],
        "lr": 1e-4,
        "optimizer": "adam",
    }
    imageClassifier = ImageClassifier(config)

    num_epoch = 10
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    trainer = L.Trainer(
        max_epochs=num_epoch,
        devices="auto",
        accelerator="gpu",
        precision="16-mixed",
        log_every_n_steps=16,
        limit_val_batches=50,
        # logger=wandb_logger,
        # val_check_interval=1/5,
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        model=imageClassifier,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    trainer

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import lightning.pytorch as pl

# import lightning as L
import mlflow.pytorch

# Required libraries
import ray
import ray.air as air
import ray.train.lightning
import torch
import torch.nn as nn
import torch.optim as optim

#  https://docs.ray.io/en/latest/tune/examples/tune-mlflow.html
import yaml
from ray import tune

# from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.train import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from dataloader import PETADataset
from torch.optim import SGD, Adam
from torchmetrics import Accuracy


import timm


class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super(ImageClassifier, self).__init__()
        self.lf = config["lr"]

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = Adam

        self.acc = Accuracy(task="binary", num_classes=1)

        # Using a pretrained backbone
        self.model = timm.create_model(
            config["model_name"], pretrained=True, num_classes=1
        )

    def forward(self, x):
        x = self.model(x)
        return x.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y.float())
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.val_origin_image = x
        self.val_output = self.forward(x)
        loss = self.criterion(self.val_output, y.float())
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lf)


def train_func(config):
    ###### dataload  ######
    train_dataset = PETADataset(True, cfg["data"])
    valid_dataset = PETADataset(False, cfg["data"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )

    # Training
    model = ImageClassifier(config)

    # Configure PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        enable_checkpointing=False,
        log_every_n_steps=16,
        limit_val_batches=50,
        val_check_interval=1 / 10,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    ### config
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    model_candidates = [
        "resnet50",
        "vgg16",
        "convnext_base",
        "vit_small_patch8_224_dino",
        "deit_small_patch16_224",
    ]
    optimizers = {"adam": Adam, "sgd": SGD}

    config = {
        "cfg": cfg,
    }
    search_space = {
        "model_name": tune.grid_search(model_candidates),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.grid_search([8, 16, 32]),  # tune.choice([8, 16]),
    }

    scheduler = ASHAScheduler(max_t=20, grace_period=1, reduction_factor=2)

    storage_path = (
        "C:\\Users\\wolve\\arsim\\spacevision_classification\\ray_tmp\\ray_results"
    )
    os.makedirs(storage_path, exist_ok=True)
    exp_name = "tune_analyzing_results"

    # Configure scaling and resource requirements.
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        trainer_resources={"GPU": 0.0},
    )

    max_failures = 2
    run_config = ray.train.RunConfig(
        failure_config=ray.train.FailureConfig(max_failures),
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_score_attribute="val_loss",
            num_to_keep=5,
        ),
        storage_path=storage_path,
    )
    torch_config = TorchConfig(backend="gloo")

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
        ),
    )

    result = tuner.fit()
    result

    best_result = result.get_best_result("val_loss", "min")
    print("Best trial config: {}".format(best_result.config))

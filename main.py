### ref. https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html
### MLFlow ref.  https://docs.ray.io/en/latest/tune/examples/includes/mlflow_ptl_example.html
###              https://docs.ray.io/en/latest/tune/examples/tune-mlflow.html

import os
import yaml

from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from torch.utils.data import DataLoader
from dataloader import PETADataset
from ImageClassifier import ImageClassifier

import mlflow
import lightning.pytorch as pl

# Configure environment for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_mlflow(config):
    """
    Configure MLflow for experiment tracking and autologging.
    """
    mlflow.set_tracking_uri(config["cfg"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["cfg"]["mlflow"]["experiment_name"])
    mlflow.log_params(config)
    mlflow.pytorch.autolog(log_every_n_step=config["cfg"]["mlflow"]["log_every_n_step"])


def create_dataloaders(config):
    """
    Create and return DataLoaders for training and validation datasets.
    """
    train_dataset = PETADataset(
        True, config["cfg"]["data"], augmentation=config["augment_data"]
    )
    valid_dataset = PETADataset(False, config["cfg"]["data"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config["cfg"]["train"]["num_workers"],
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config["cfg"]["valid"]["num_workers"],
        persistent_workers=True,
    )

    return train_loader, valid_loader


def train_model(config):
    """
    Function to initialize data loaders, model, and trainer to be used with Ray.
    """
    train_loader, valid_loader = create_dataloaders(config)
    setup_mlflow(config)

    callbacks = [
        TuneReportCheckpointCallback(
            metrics={
                "loss": "loss",
                "val_loss": "val_loss",
                "acc": "acc",
                "val_acc": "val_acc",
            },
            on="validation_end",
        )
    ]

    # Initialize the model
    model = ImageClassifier(config)

    # Configure PyTorch Lightning Trainer.
    cfg_train = config["cfg"]["train"]
    trainer = pl.Trainer(
        max_epochs=cfg_train["num_epochs"],
        devices="auto",
        accelerator="auto",
        log_every_n_steps=cfg_train["log_every_n_steps"],
        limit_val_batches=cfg_train["limit_val_batches"],
        # val_check_interval=0.1,
        check_val_every_n_epoch=cfg_train["check_val_every_n_epoch"],
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def load_yaml_config(config_path="config.yaml"):
    """
    Load the configuration file.
    """
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


def main():
    config = load_yaml_config()
    cfg_tune = config["ray"]["tune"]
    cfg_scheduler = config["ray"]["scheduler"]
    cfg_hyperparameters = config["hyperparameters"]

    # Runtime configuration for individual trials
    ray_storage_path = config["ray"]["storage_path"]
    os.makedirs(ray_storage_path, exist_ok=True)

    run_config = RunConfig(
        name=config["ray"]["run_name"],
        # failure_config=FailureConfig(config["ray"]["max_failures"]),
        storage_path=ray_storage_path,
    )

    tune_config = tune.TuneConfig(
        metric=cfg_tune["metric"],
        mode=cfg_tune["mode"],
        num_samples=cfg_tune["num_samples"],
        scheduler=ASHAScheduler(
            max_t=cfg_scheduler["max_t"],
            grace_period=cfg_scheduler["grace_period"],
            reduction_factor=cfg_scheduler["reduction_factor"],
        ),
        max_concurrent_trials=cfg_tune["max_concurrent_trials"],
        trial_dirname_creator=custom_trial_dirname_creator,
    )

    config_total = {
        "cfg": config,
        ##### hyperparameters #####
        "model_name": tune.grid_search(cfg_hyperparameters["model_name"]),
        "lr": tune.loguniform(
            cfg_hyperparameters["lr"]["min"],
            cfg_hyperparameters["lr"]["max"],
        ),
        "batch_size": tune.grid_search(cfg_hyperparameters["batch_size"]),
        "optimizer": tune.choice(cfg_hyperparameters["optimizer"]),
        "augment_data": (
            tune.grid_search([True, False])
            if cfg_hyperparameters["augment_data"] == "Both"
            else cfg_hyperparameters["augment_data"]
        ),
    }

    trainable = tune.with_parameters(train_model)

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": config["ray"]["resources"]["cpus_per_trial"],
                "gpu": config["ray"]["resources"]["gpus_per_trial"],
            },
        ),
        tune_config=tune_config,
        run_config=run_config,
        param_space=config_total,
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    print("Best trial config:", best_result.config)
    print("Best trial final validation loss:", best_result.metrics["val_loss"])


if __name__ == "__main__":
    main()

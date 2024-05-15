"""An example showing how to use Pytorch Lightning training, Ray Tune
HPO, and MLflow autologging all together."""

import os
import tempfile

import mlflow.pytorch
import lightning as pl

from ray import train, tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.examples.mnist_ptl_mini import LightningMNISTClassifier, MNISTDataModule
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


# config = {
#     "layer_1": 23,
#     "layer_2": 23,
#     "lr": 1e-4,
#     "batch_size": 32,
#     "experiment_name": "ptl_autologging_example",
#     # "tracking_uri": mlflow.get_tracking_uri(),
#     "data_dir": os.path.join(tempfile.gettempdir(), "mnist_data_"),
#     "num_epochs": 2,
# }

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("ptl_autologging_example")

# # mlf_out = setup_mlflow(
# #     config,
# #     experiment_name=config.get("experiment_name", None),
# #     # tracking_uri=config.get("tracking_uri", None),
# # )

# data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
# model = LightningMNISTClassifier(config, data_dir)
# dm = MNISTDataModule(data_dir=data_dir, batch_size=config["batch_size"])
# metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}

# # checkpoint_callback = ModelCheckpoint(dirpath=".")
# mlflow.pytorch.autolog(log_every_n_step=200)
# trainer = pl.Trainer(
#     log_every_n_steps=16,
#     # limit_val_batches=50,
#     val_check_interval=0.1,
#     callbacks=[TuneReportCheckpointCallback(metrics, on="validation_end")],
#     # logger=mlFlowLogger,
#     # callbacks=[checkpoint_callback],
# )
# # trainer.ml_out = mlf_out
# trainer.fit(model, dm)


def train_mnist_tune(config, data_dir=None, num_epochs=10, num_gpus=0):
    # setup_mlflow(
    #     config,
    #     experiment_name=config.get("experiment_name", None),
    #     tracking_uri="http://localhost:5000",
    # )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ptl_autologging_example")
    mlflow.log_params(config)

    model = LightningMNISTClassifier(config, data_dir)
    dm = MNISTDataModule(data_dir=data_dir, batch_size=config["batch_size"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}

    mlflow.pytorch.autolog(log_every_n_step=200)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=16,
        limit_val_batches=50,
        val_check_interval=0.1,
        callbacks=[TuneReportCheckpointCallback(metrics, on="validation_end")],
        # callbacks=[checkpoint_callback],
    )
    # trainer.ml_out = mlf_out
    trainer.fit(model, dm)


from ray.train import ScalingConfig


def tune_mnist(
    num_samples=10,
    num_epochs=10,
    gpus_per_trial=0,
    experiment_name="ptl_autologging_example",
):
    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    # Download data
    MNISTDataModule(data_dir=data_dir, batch_size=32).prepare_data()

    # Set the MLflow experiment, or create it if it does not exist.

    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "experiment_name": experiment_name,
        # "tracking_uri": mlflow.get_tracking_uri(),
        "data_dir": os.path.join(tempfile.gettempdir(), "mnist_data_"),
        "num_epochs": num_epochs,
        "scaling_config": ScalingConfig(
            num_workers=1,
            resources_per_worker={
                "CPU": 1,
            },
        ),
    }

    trainable = tune.with_parameters(
        train_mnist_tune,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_mnist",
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    tune_mnist(num_samples=2, num_epochs=2, gpus_per_trial=0)

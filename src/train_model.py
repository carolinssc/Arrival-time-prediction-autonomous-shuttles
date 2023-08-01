import datetime

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
import torch

import wandb
from data.datamodule import SHOWDataModule, Standardize
from models.gcn_model import NodeEncodedGCN_1l
from models.model_base_class import BaseModelClass
from pytorch_lightning.callbacks import ModelCheckpoint


def init_model(cfg: DictConfig, transform: Standardize) -> type[BaseModelClass]:
    if cfg.model.name == "gcn":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = GCN_segments(
            input_size=cfg.model.model_parameters.input_size,
            global_size=cfg.model.model_parameters.global_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
        )
    elif cfg.model.name == "node_encoded_gcn":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCN(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "node_encoded_gcn_tt":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCN_tt(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "node_encoded_gat":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGAT(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "mlp":
        model = MLP_segments()
    elif cfg.model.name == "classification_node_encoded_gcn":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCNClassifier(
            input_size=cfg.model.model_parameters.input_size,
            output_size=cfg.model.model_parameters.output_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "zil_node_encoded_gcn":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = ZILGM_NodeEncodedGCN(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            embed_dim=cfg.model.model_parameters.embed_dim,
            embed_hidden=cfg.model.model_parameters.embed_hidden,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "node_encoded_gcn_1l":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCN_1l(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "node_encoded_gcn_2l":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCN_2l(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    elif cfg.model.name == "node_encoded_gcn_3l":
        hidden_dims = list(cfg.model.model_parameters.hidden_dims.values())
        model = NodeEncodedGCN_3l(
            input_size=cfg.model.model_parameters.input_size,
            hidden_layers=hidden_dims,
            lr=cfg.hyperparameters.lr,
            drop_p=cfg.hyperparameters.drop_p,
            weight_decay=cfg.hyperparameters.weight_decay,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            aggregation_function=cfg.model.model_parameters.aggregation_function,
        )
    return model


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_model(cfg: DictConfig) -> None:
    # Set random seed
    if cfg.hyperparameters.seed is None:
        # sample seed if not given
        cfg.hyperparameters.seed = torch.randint(0, 10000, (1,)).item()

    print(f"Using seed: {cfg.hyperparameters.seed}")
    pl.seed_everything(cfg.hyperparameters.seed)

    # Init data module
    data_module = SHOWDataModule(
        site_name=cfg.data.site_name,
        transform=cfg.data.transform,
        num_lags=cfg.data.num_lags,
        train_frac=cfg.data.train_frac,
        batch_size=cfg.hyperparameters.batch_size,
        empty_graph=cfg.model.model_parameters.empty_graph,
        rf_remove_zero_obs=cfg.data.rf_remove_zero_obs,
        verbose=cfg.data.verbose_datamodule,
        time_kind=cfg.data.time_kind,
    )
    transform = data_module.setup()
    # Init model
    model = init_model(cfg, transform)

    # Init logger
    now = datetime.datetime.now()
    time_str = f"{now.day}-{now.month}-{now.hour}{now.minute}"
    run_name = f"{cfg.model.name}_{cfg.hyperparameters.seed}_{time_str}"
    wandb_logger = WandbLogger(
        project=cfg.logging_parameters.project_name,
        name=run_name,
        save_dir="wandb_dir",
        log_model=True,
    )
    wandb_logger.log_hyperparams(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'models/{cfg.model.name}/{cfg.saving_parameters.checkpoint_name}-{now.strftime("%m%d%H%M")}_best.ckpt',
        monitor="val/loss",
        save_top_k=1,
        mode="min",
    )
    # Init trainer
    trainer = pl.Trainer(
        max_epochs=cfg.hyperparameters.n_epochs,
        accelerator=cfg.hyperparameters.accelerator,
        devices=[cfg.hyperparameters.which_gpu],
        fast_dev_run=cfg.hyperparameters.dev_run,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.tune(
        model=model, datamodule=data_module
    )  # TODO remove this to check batch size manually to see if it affects training
    trainer.fit(model=model, datamodule=data_module)

    # Test model
    # trainer.test(model=model, datamodule=data_module, ckpt_path="best") # TODO fix after dev
    trainer.test(model=model, datamodule=data_module)

    if cfg.saving_parameters.save_model:
        trainer.save_checkpoint(
            f'models/{cfg.model.name}/{cfg.saving_parameters.checkpoint_name}-{cfg.hyperparameters.seed}-{now.strftime("%m%d%H%M")}.ckpt'
        )
        trainer.save_checkpoint(
            f'{wandb_logger.experiment.dir}/{cfg.saving_parameters.checkpoint_name}-{cfg.hyperparameters.seed}-{now.strftime("%m%d%H%M")}.ckpt'
        )
    wandb_logger.finalize(status="Success")
    wandb.finish()


if __name__ == "__main__":
    train_model()

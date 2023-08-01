from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import MeanAbsolutePercentageError

from data.datamodule import Standardize


import numpy as np


# TODO clean up and move somewhere nice
def simple_model_evaluation(y_test, y_pred_test):
    # Simple model evaluation that computes and prints MSE, RMSE and MAPE for the training and testing set

    # train_error_mse = np.square(y_train - y_pred_train).sum() / y_train.shape[0]
    test_error_mse = np.square(y_test - y_pred_test).sum() / y_test.shape[0]

    # train_error_mape = (100 / y_train.shape[0]) * (
    #    np.absolute(y_train - y_pred_train) / y_train
    # ).sum()  # y_train should never be 0 since the travel time in a segment cannot be 0
    test_error_mape = (100 / y_test.shape[0]) * (np.absolute(y_test - y_pred_test) / y_test).sum()

    test_error_mae = (1 / y_test.shape[0]) * (np.absolute(y_test - y_pred_test)).sum()
    print("-----------MSE----------")
    # print("Training error: {}".format(train_error_mse))
    print("Testing error: {}".format(test_error_mse))
    print("-----------RMSE----------")
    # print("Training error: {}".format(np.sqrt(train_error_mse)))
    print("Testing error: {}".format(np.sqrt(test_error_mse)))
    print("-----------MAPE----------")
    # print("Training error: {:.2f} %".format(train_error_mape))
    print("Testing error: {:.2f} %".format(test_error_mape))
    print("-----------MAE----------")
    print("Testing error: {}".format(test_error_mae))
    return test_error_mse, np.sqrt(test_error_mse), test_error_mape, test_error_mae


class BaseModelClass(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, batch_size: int, transform: Standardize) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.loss = self.l1_loss

        self.mape = MeanAbsolutePercentageError()

        self.lr = lr
        self.weight_decay = weight_decay
        self.transform = transform

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        step_type = "train"
        batch_size = len(batch.batch.unique())
        step_dict = self.standard_step(batch=batch, step_type=step_type)
        self.log(f"{step_type}/loss", step_dict["loss"], batch_size=batch_size, on_step=False, on_epoch=True)
        self.log(
            f"{step_type}/real_scale_mse",
            step_dict["real_scale_mse"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_type}/real_scale_mae",
            step_dict["real_scale_mae"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{step_type}/mape", step_dict["mape"], batch_size=64, on_step=False, on_epoch=True)
        return step_dict["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        step_type = "val"
        batch_size = len(batch.batch.unique())
        step_dict = self.standard_step(batch=batch, step_type=step_type)
        self.log(f"{step_type}/loss", step_dict["loss"], batch_size=batch_size, on_step=False, on_epoch=True)
        self.log(
            f"{step_type}/real_scale_mse",
            step_dict["real_scale_mse"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_type}/real_scale_mae",
            step_dict["real_scale_mae"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{step_type}/mape", step_dict["mape"], batch_size=64, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        step_type = "test"
        step_dict = self.standard_step(batch=batch, step_type="test")
        return {"y_hat": step_dict["y_hat"], "y_true": step_dict["y_true"]}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        step_type = "predict"
        step_dict = self.standard_step(batch=batch, step_type=step_type)

        rt_y_hat = self.transform.retransform_target_vals(step_dict["y_hat"])
        rt_y_true = self.transform.retransform_target_vals(step_dict["y_true"][:, 1])

        return rt_y_hat, rt_y_true, step_dict["y_true"][:, 0]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return super().predict_dataloader()

    def test_epoch_end(self, output_results: Tensor):
        step_type = "test"
        yn_hat = torch.cat([x["y_hat"] for x in output_results])
        yn_true = torch.cat([x["y_true"] for x in output_results])[:, 1]
        y_hat = self.transform.retransform_target_vals(yn_hat)
        y_true = self.transform.retransform_target_vals(yn_true)
        test_error_mse, test_error_rmse, test_error_mape, test_error_mae = simple_model_evaluation(
            y_pred_test=y_hat.cpu(), y_test=y_true.cpu()
        )
        real_scale_loss = self.loss(y_hat, y_true)
        self.log(f"{step_type}/error_mse", test_error_mse)
        self.log(f"{step_type}/error_rmse", test_error_rmse)
        self.log(f"{step_type}/error_mape", test_error_mape)
        self.log(f"{step_type}/error_mae", test_error_mae)
        self.log(f"{step_type}/real_scale_loss", real_scale_loss)

        return None

    def standard_step(self, batch: Tensor, step_type: int) -> Tensor:
        batch_size = len(batch.batch.unique())
        x = batch.x
        edge_index = batch.edge_index
        u = batch.global_feat.reshape(batch_size, -1)
        node_encoding = batch.node_encoding
        num_nodes = node_encoding.shape[-1]

        y_true = batch.y.reshape(batch_size, -1)
        true_obs_idxs = y_true[:, 0].long()
        true_obs_mask = F.one_hot(true_obs_idxs, num_nodes).reshape(-1).bool()

        true_obs_tt = y_true[:, 1]

        y_hat = self.forward(
            x=x, u=u, node_encoding=node_encoding, edge_index=edge_index, batch_size=batch_size
        ).squeeze()
        y_hat = y_hat[true_obs_mask]

        loss = self.loss(y_hat, true_obs_tt)

        mae = self.l1_loss(y_hat, true_obs_tt)
        mse = self.mse_loss(y_hat, true_obs_tt)
        mape = self.mape(y_hat, true_obs_tt)

        rtfed_y_hat = self.transform.retransform_target_vals(y_hat)
        rtfed_true_obs_tt = self.transform.retransform_target_vals(true_obs_tt)
        real_scale_mae = self.l1_loss(rtfed_y_hat, rtfed_true_obs_tt)
        real_scale_mse = self.mse_loss(rtfed_y_hat, rtfed_true_obs_tt)

        return {
            "loss": loss,
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "real_scale_mae": real_scale_mae,
            "real_scale_mse": real_scale_mse,
            "y_hat": y_hat.detach(),
            "y_true": y_true.detach(),
        }

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

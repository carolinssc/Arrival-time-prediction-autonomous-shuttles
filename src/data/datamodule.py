import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
import sklearn


class NoTransform:
    """Simple no transform class to make code run without transformation"""

    def setup(self, his_train: Tensor, weather_train: Tensor, target_train: Tensor) -> None:
        self.faux_setup = True

    def transform(self, his: Tensor, weather: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return his, weather, target

    def retransform(
        self,
        standardized_his: Tensor,
        standardized_weather: Tensor,
        standardized_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return standardized_his, standardized_weather, standardized_target

    def retransform_target_vals(self, standardized_target_vals: Tensor) -> Tensor:
        return standardized_target_vals


class Standardize:
    """Simple class for standardisation that saves params and can restandardize"""

    def setup(self, his_train: Tensor, weather_train: Tensor, target_train: Tensor) -> None:
        train_his_mean = his_train.mean((0, 1, 2))  # TODO fix which dims
        train_his_std = his_train.std((0, 1, 2))  # TODO fix which dims

        train_weather_mean = weather_train.mean((0))  # TODO fix which dims
        train_weather_std = weather_train.std((0))  # TODO fix which dims

        train_target_mean = target_train[..., 1].mean()  # TODO fix which dims
        train_target_std = target_train[..., 1].std()  # TODO fix which dims

        self.params = {
            "his": {"mean": train_his_mean, "std": train_his_std},
            "weather": {"mean": train_weather_mean, "std": train_weather_std},
            "target": {"mean": train_target_mean, "std": train_target_std},
        }

    def transform(self, his: Tensor, weather: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        standardized_his = (his - self.params["his"]["mean"]) / self.params["his"]["std"]
        standardized_weather = (weather - self.params["weather"]["mean"]) / self.params["weather"]["std"]
        standardized_target = (target[..., 1] - self.params["target"]["mean"]) / self.params["target"]["std"]
        standardized_target = torch.stack([target[..., 0], standardized_target]).T
        return standardized_his, standardized_weather, standardized_target

    def retransform(
        self,
        standardized_his: Tensor,
        standardized_weather: Tensor,
        standardized_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        his = standardized_his * self.params["his"]["std"] + self.params["his"]["mean"]
        weather = standardized_weather * self.params["weather"]["std"] + self.params["weather"]["mean"]
        target = standardized_target[..., 1] * self.params["target"]["std"] + self.params["target"]["mean"]
        target = torch.stack([standardized_target[..., 0], target]).T
        return his, weather, target

    def retransform_target_vals(self, standardized_target_vals: Tensor) -> Tensor:
        assert standardized_target_vals.dim() == 1
        target_vals = standardized_target_vals * self.params["target"]["std"] + self.params["target"]["mean"]
        return target_vals


class MaxMin:
    """Simple class for standardisation that saves params and can restandardize"""

    def setup(self, his_train: Tensor, weather_train: Tensor, target_train: Tensor) -> None:
        train_his_min, _ = his_train.view(-1, 2).min(0)  # TODO fix which dims
        train_his_max, _ = his_train.view(-1, 2).max(0)  # TODO fix which dims

        train_weather_min, _ = weather_train.view(-1, 3).min(0)  # TODO fix which dims
        train_weather_max, _ = weather_train.view(-1, 3).max(0)  # TODO fix which dims

        train_target_min, _ = target_train[..., 1].min(0)  # TODO fix which dims
        train_target_max, _ = target_train[..., 1].max(0)  # TODO fix which dims

        self.params = {
            "his": {"min": train_his_min, "max": train_his_max},
            "weather": {"min": train_weather_min, "max": train_weather_max},
            "target": {"min": train_target_min, "max": train_target_max},
        }

    def transform(self, his: Tensor, weather: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scaled_his = (his - self.params["his"]["min"]) / (self.params["his"]["max"] - self.params["his"]["min"])
        scaled_weather = (weather - self.params["weather"]["min"]) / (
            self.params["weather"]["max"] - self.params["weather"]["min"]
        )
        scaled_target = (target[..., 1] - self.params["target"]["min"]) / (
            self.params["target"]["max"] - self.params["target"]["min"]
        )

        scaled_target = torch.stack([target[..., 0], scaled_target]).T
        return scaled_his, scaled_weather, scaled_target

    def retransform(
        self,
        standardized_his: Tensor,
        standardized_weather: Tensor,
        standardized_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        his = standardized_his * (self.params["his"]["max"] - self.params["his"]["min"]) + self.params["his"]["min"]
        weather = (
            standardized_weather * (self.params["weather"]["max"] - self.params["weather"]["min"])
            + self.params["weather"]["min"]
        )
        target = (
            standardized_target[..., 1] * (self.params["target"]["max"] - self.params["target"]["min"])
            + self.params["target"]["min"]
        )
        target = torch.stack([standardized_target[..., 0], target]).T
        return his, weather, target

    def retransform_target_vals(self, standardized_target_vals: Tensor) -> Tensor:
        assert standardized_target_vals.dim() == 1
        target_vals = (
            standardized_target_vals * (self.params["target"]["max"] - self.params["target"]["min"])
            + self.params["target"]["min"]
        )
        return target_vals


class SHOWDataset(Data):
    """Torch Geometric dataset extended with graph level feature
    x:
    edge_index:
    edge_attr:
    y:
    global_feat:
    node_encoding:

    global feat:
    node encoding:
    """

    def __init__(
        self,
        x: Tensor = None,
        edge_index: Tensor = None,
        edge_attr: Tensor = None,
        y: Tensor = None,
        global_feat: Tensor = None,
        node_encoding: Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, **kwargs)
        self.global_feat = global_feat
        self.node_encoding = node_encoding


class SHOWDataModule(pl.LightningDataModule):
    """Lightning DataModule for SHOW data that loads historical, weather and time data.
    The data is transformed and split into train, val and test sets.
    """

    def __init__(
        self,
        site_name: str,
        transform: str,
        train_frac: float,
        num_lags: int,
        batch_size: int,
        empty_graph: bool,
        time_kind: str,
        verbose=True,
        rf_remove_zero_obs: bool = False,
    ):
        super().__init__()
        self.site_name = site_name
        self.time_kind = time_kind
        self.folder_path = (f"../../{site_name}",)

        self.empty_graph = empty_graph

        self.rf_remove_zero_obs = rf_remove_zero_obs

        self.batch_size = batch_size
        self.train_frac = train_frac

        self.num_lags = num_lags

        if transform == "standardize":
            self.transform = Standardize()
        elif transform == "maxmin":
            self.transform = MaxMin()
        elif transform == "none":
            self.transform = NoTransform()
        else:
            raise NotImplementedError

        self.verbose = verbose

    def setup(self, stage=None) -> Standardize:
        base_path = os.path.dirname(__file__)
        # Load predefined segment names, order and edge index
        constants = yaml.safe_load(
            Path(f"{base_path}/../../data/processed/{self.site_name}/constants.yaml").read_text()
        )
        if self.time_kind == "travel_times":
            self.segment_lookup = {k: v for v, k in enumerate(constants["SEGMENT_IDS_FILTERED"])}
            self.edge_index = torch.Tensor(constants["EDGE_INDEX_SEGMENTS"]).long()
        else:
            self.segment_lookup = {k: v for v, k in enumerate(constants["STOP_IDS_FILTERED"])}
            self.edge_index = torch.Tensor(constants["EDGE_INDEX_STOPS"]).long()

        self.n_vehicles = constants["N_VEHICLES"]
        self.lag_kind = constants["LAG_KIND"]

        if self.empty_graph:
            # Remove graph and add self loops between all idx in the segment lookup keys
            self.edge_index = torch.stack([torch.arange(len(self.segment_lookup))] * 2)

        self.n_nodes = len(self.segment_lookup)  # we need to change that in the future
        # Load data and add node ordering
        # Remember to not sort vehicles but just concat them so precalculated lags match
        # TODO add system for loading different amount of vehicles

        if self.lag_kind == "per_veh":
            tt_tv_arr = []
            for vehicle_num in range(1, self.n_vehicles + 1):
                tt_tv_arr.append(
                    pd.read_csv(
                        f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_vehicle_{vehicle_num}_{self.time_kind}_train.csv",
                        parse_dates=["date"],
                    )
                )
            tt_tv = pd.concat(tt_tv_arr).reset_index(drop=True)
        elif self.lag_kind == "all_veh":
            tt_tv = pd.read_csv(
                f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_{self.time_kind}_train.csv",
                parse_dates=["date"],
            )

        tt_tv["segment_idx"] = tt_tv.segment_id.map(self.segment_lookup)

        vehicle_tv = F.one_hot(torch.LongTensor(tt_tv[["vehicle_id"]].to_numpy()) - 1).squeeze()
        if self.n_vehicles == 1:
            vehicle_tv = vehicle_tv.unsqueeze(-1)

        if self.lag_kind == "per_veh":
            tt_test_arr = []
            for i in range(1, self.n_vehicles + 1):
                tt_test_arr.append(
                    pd.read_csv(
                        f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_vehicle_{i}_{self.time_kind}_test.csv",
                        parse_dates=["date"],
                    )
                )
            tt_test = pd.concat(tt_test_arr).reset_index(drop=True)
        elif self.lag_kind == "all_veh":
            tt_test = pd.read_csv(
                f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_{self.time_kind}_test.csv",
                parse_dates=["date"],
            )
        tt_test["segment_idx"] = tt_test.segment_id.map(self.segment_lookup)

        # TODO this can be cleaned up
        if self.time_kind == "travel_times":
            target_name = "travel_time"
            if self.lag_kind == "per_veh":
                his_data_arr = []
                for i in range(1, self.n_vehicles + 1):
                    his_data_arr.append(
                        np.load(
                            f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_tt_vehicle_{i}_train.npy",
                            allow_pickle=True,
                        )
                    )
                his_data = np.concatenate(his_data_arr)
                his_test_arr = []
                for i in range(1, self.n_vehicles + 1):
                    his_test_arr.append(
                        np.load(
                            f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_tt_vehicle_{i}_test.npy",
                            allow_pickle=True,
                        )
                    )
                his_test = np.concatenate(his_test_arr)
            elif self.lag_kind == "all_veh":
                his_data = np.load(
                    f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_tt_train.npy",
                    allow_pickle=True,
                )
                his_test = np.load(
                    f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_tt_test.npy",
                    allow_pickle=True,
                )

        elif self.time_kind == "dwell_times":
            target_name = "dwell_time"
            if self.lag_kind == "per_veh":
                his_data_arr = []
                for i in range(1, self.n_vehicles + 1):
                    his_data_arr.append(
                        np.load(
                            f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_dt_vehicle_{i}_train.npy",
                            allow_pickle=True,
                        )
                    )
                his_data = np.concatenate(his_data_arr)
                his_test_arr = []
                for i in range(1, self.n_vehicles + 1):
                    his_test_arr.append(
                        np.load(
                            f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_dt_vehicle_{i}_test.npy",
                            allow_pickle=True,
                        )
                    )
                his_test = np.concatenate(his_test_arr)
            elif self.lag_kind == "all_veh":
                his_data = np.load(
                    f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_dt_train.npy",
                    allow_pickle=True,
                )
                his_test = np.load(
                    f"{base_path}/../../data/processed/{self.site_name}/{self.site_name}_lags_dt_test.npy",
                    allow_pickle=True,
                )

        his_data = np.stack(his_data)
        his_segment_order = his_data[-1, :, 0, 0]
        his_data = torch.Tensor(his_data[..., -self.num_lags :, 1:].astype("float64"))

        his_test = np.stack(his_test)
        his_test_segment_order = his_test[-1, :, 0, 0]
        his_test = torch.Tensor(his_test[..., -self.num_lags :, 1:].astype("float64"))

        # Node ordering sanity check
        for i in range(len(self.segment_lookup)):
            assert his_test_segment_order[i] == his_segment_order[i]
            assert self.segment_lookup[his_segment_order[i]] == i

        # setup transform before potentially removing data based on RF
        weather_temp = torch.Tensor(tt_tv[["temp", "prcp", "wspd"]].to_numpy())
        target_temp = torch.Tensor(tt_tv[["segment_idx", target_name]].to_numpy())
        self.transform.setup(his_train=his_data, weather_train=weather_temp, target_train=target_temp)

        # Use pretrained RF to remove zero obs
        if self.rf_remove_zero_obs:
            print("** Removing zero obs based on RF classifier ***")
            self.non_zero_indices, self.non_zero_indices_test = self.random_forest_remove_zero_obs(
                tt_tv, tt_test, his_data, his_test
            )
            his_data = his_data[self.non_zero_indices]
            tt_tv = tt_tv.iloc[self.non_zero_indices]
            his_test = his_test[self.non_zero_indices_test]
            tt_test = tt_test.iloc[self.non_zero_indices_test]
        # Create val and train split
        tv_obs_full = len(his_data)
        val_len = int(tv_obs_full * (1 - self.train_frac))
        train_len = tv_obs_full - val_len
        # Frozen seed for reproducibility
        train_set, val_set = random_split(
            range(tv_obs_full),
            [train_len, val_len],
            generator=torch.Generator().manual_seed(1),
        )

        his_train = his_data[train_set.indices]
        his_val = his_data[val_set.indices]

        time_tv = torch.Tensor(tt_tv[["dow_sin", "dow_cos", "tod_sin", "tod_cos"]].to_numpy())
        time_train = time_tv[train_set.indices]
        time_val = time_tv[val_set.indices]
        time_test = torch.Tensor(tt_test[["dow_sin", "dow_cos", "tod_sin", "tod_cos"]].to_numpy())

        weather_tv = torch.Tensor(tt_tv[["temp", "prcp", "wspd"]].to_numpy())
        weather_train = weather_tv[train_set.indices]
        weather_val = weather_tv[val_set.indices]
        weather_test = torch.Tensor(tt_test[["temp", "prcp", "wspd"]].to_numpy())

        vehicle_train = vehicle_tv[train_set.indices]
        vehicle_val = vehicle_tv[val_set.indices]
        vehicle_test = F.one_hot(torch.LongTensor(tt_test[["vehicle_id"]].to_numpy()) - 1).squeeze()
        if self.n_vehicles == 1:
            vehicle_test = vehicle_test.unsqueeze(-1)

        self.train_dates = tt_tv.iloc[train_set.indices].date
        self.val_dates = tt_tv.iloc[val_set.indices].date
        self.test_dates = tt_test.date
        tt_tv = torch.Tensor(tt_tv[["segment_idx", target_name]].to_numpy())
        tt_train = tt_tv[train_set.indices]
        tt_val = tt_tv[val_set.indices]
        tt_test = torch.Tensor(tt_test[["segment_idx", target_name]].to_numpy())

        # Transform the data w. the given tranformation
        tfed_his_train, tfed_weather_train, tfed_tt_train = self.transform.transform(
            his=his_train, weather=weather_train, target=tt_train
        )
        tfed_his_val, tfed_weather_val, tfed_tt_val = self.transform.transform(
            his=his_val, weather=weather_val, target=tt_val
        )
        tfed_his_test, tfed_weather_test, tfed_tt_test = self.transform.transform(
            his=his_test, weather=weather_test, target=tt_test
        )
        # Create graph level feature
        global_train = torch.cat([vehicle_train, time_train, tfed_weather_train], dim=1)
        global_val = torch.cat([vehicle_val, time_val, tfed_weather_val], dim=1)
        global_test = torch.cat([vehicle_test, time_test, tfed_weather_test], dim=1)

        node_one_hot_encoding = F.one_hot(torch.tensor(list(range(self.n_nodes))))
        # Create datasets that can be used w. torch geometric
        self.train_data = [
            SHOWDataset(
                x=x,
                edge_index=self.edge_index,
                y=y,
                global_feat=global_feat,
                node_encoding=node_one_hot_encoding,
            )
            for x, y, global_feat in zip(tfed_his_train, tfed_tt_train, global_train)
        ]

        self.val_data = [
            SHOWDataset(
                x=x,
                edge_index=self.edge_index,
                y=y,
                global_feat=global_feat,
                node_encoding=node_one_hot_encoding,
            )
            for x, y, global_feat in zip(tfed_his_val, tfed_tt_val, global_val)
        ]

        self.test_data = [
            SHOWDataset(
                x=x,
                edge_index=self.edge_index,
                y=y,
                global_feat=global_feat,
                node_encoding=node_one_hot_encoding,
            )
            for x, y, global_feat in zip(tfed_his_test, tfed_tt_test, global_test)
        ]
        return self.transform

    def random_forest_remove_zero_obs(
        self, tt_tv: pd.DataFrame, tt_test: pd.DataFrame, his_data: np.ndarray, his_test: np.ndarray
    ) -> tuple[np.ndarray, pd.DataFrame]:
        # Load training data for random forest
        target = tt_tv["dwell_time"]
        segment_idx = tt_tv["segment_idx"]
        lags_data = [his_data[idx, segment_id] for idx, segment_id in enumerate(segment_idx)]
        lags_data = torch.stack(lags_data)
        unravel_lags = lags_data.reshape(-1, 4)
        x = tt_tv[
            [
                "lat",
                "lon",
                "dow_sin",
                "dow_cos",
                "tod_sin",
                "tod_cos",
                "temp",
                "prcp",
                "wspd",
                "segment_id",
                "vehicle_id",
            ]
        ]
        x = pd.get_dummies(x, columns=["segment_id", "vehicle_id"])
        x["lags_1"] = unravel_lags[:, 0]
        x["lags_2"] = unravel_lags[:, 2]
        x_train_vals = x.values

        # Load test data for random forest
        target_test = tt_test["dwell_time"]
        segment_idx_test = tt_test["segment_idx"]
        lags_test = [his_test[idx, segment_id] for idx, segment_id in enumerate(segment_idx_test)]
        lags_test = torch.stack(lags_test)
        unravel_lags_test = lags_test.reshape(-1, 4)
        x_test = tt_test[
            [
                "lat",
                "lon",
                "dow_sin",
                "dow_cos",
                "tod_sin",
                "tod_cos",
                "temp",
                "prcp",
                "wspd",
                "segment_id",
                "vehicle_id",
            ]
        ]
        x_test = pd.get_dummies(x_test, columns=["segment_id", "vehicle_id"])
        x_test["lags_1"] = unravel_lags_test[:, 0]
        x_test["lags_2"] = unravel_lags_test[:, 2]
        x_test_vals = x_test.values

        # Train random forest
        cl = RandomForestClassifier(random_state=0)
        y_class_train_true = target != 0
        y_class_test_true = target_test != 0
        class_weight = {
            1: 1 - sum(y_class_train_true) / len(y_class_train_true),
            0: sum(y_class_train_true) / len(y_class_train_true),
        }
        sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight, y_class_train_true)
        cl.fit(x_train_vals, target != 0, sample_weight=sample_weight)
        y_class_train_pred = cl.predict(x_train_vals)
        y_class_test_pred = cl.predict(x_test_vals)

        conf_train = sklearn.metrics.confusion_matrix(y_class_train_true, y_class_train_pred).ravel()
        conf_test = sklearn.metrics.confusion_matrix(y_class_test_true, y_class_test_pred).ravel()
        print(f"Confusion matrix train: {conf_train}")
        print(f"Confusion matrix test: {conf_test}")

        # Save indices of non zero obs
        non_zero_indices = np.where(y_class_train_pred == 1)[0]
        non_zero_indices_test = np.where(y_class_test_pred == 1)[0]
        return non_zero_indices, non_zero_indices_test

    def train_dataloader(self) -> type[DataLoader]:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self) -> type[DataLoader]:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self) -> type[DataLoader]:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=1)

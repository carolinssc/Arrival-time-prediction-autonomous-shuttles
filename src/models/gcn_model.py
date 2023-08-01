import torch
import torch.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GAT, GCNConv
from torch_geometric.nn.norm import BatchNorm

from data.datamodule import Standardize

from .model_base_class import BaseModelClass


class GCN_segments_msc(BaseModelClass):
    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        lr: float,
        weight_decay: float,
        drop_p: float = 0.5,
    ) -> None:
        super().__init__(lr, weight_decay)

        self.conv1 = GCNConv(input_size, hidden_layers[0])
        self.conv2 = GCNConv(hidden_layers[0], hidden_layers[1])
        self.conv3 = GCNConv(hidden_layers[1], hidden_layers[2])
        self.linear_out = nn.Linear(hidden_layers[2], 1)
        self.dropout = nn.Dropout(p=drop_p)

    # TODO add in the U to all nodes before conv
    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.linear_out(x)

        return x


class GCN_segments(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        input_size: int,
        global_size: int,
        hidden_layers: list[int],
        lr: float,
        weight_decay: float,
        batch_size: int,
        drop_p: float,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)

        self.conv1 = GCNConv(input_size, hidden_layers[0])
        self.fc_merge = nn.Linear(hidden_layers[0] + global_size, hidden_layers[1])
        self.conv2 = GCNConv(hidden_layers[1], hidden_layers[2])
        self.fc_out = nn.Linear(hidden_layers[2], 1)
        self.dropout = nn.Dropout(p=drop_p)

        self.act_func = nn.ReLU()

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        # Reshape node features to 2d
        x = x.reshape(x.shape[0], -1)
        # Pad global feature to be able to run each node in parallel. OBS does not work for varying graphs
        u = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)

        x = self.act_func(self.conv1(x, edge_index))
        x = self.dropout(x)

        # TODO replicate u to match size of nodes
        x = self.act_func(self.fc_merge(torch.cat([x, u], dim=1)))
        x = self.dropout(x)

        x = self.act_func(self.conv2(x, edge_index))
        x = self.fc_out(x)

        return x


class GCN_segments2(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        input_size: int,
        global_size: int,
        hidden_layers: list[int],
        lr: float,
        weight_decay: float,
        batch_size: int,
        drop_p: float = 0.2,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)

        self.conv1 = GCNConv(input_size + global_size, hidden_layers[1], aggr="max")
        # self.conv1 = GAT(input_size + global_size, hidden_layers[1], num_layers=1)
        # self.fc_merge = nn.Linear(hidden_layers[1] + global_size, hidden_layers[1])
        self.conv2 = GCNConv(hidden_layers[1], hidden_layers[2])
        self.fc = nn.Linear(hidden_layers[2], hidden_layers[2])
        self.fc_out = nn.Linear(hidden_layers[2], 1)
        # self.dropout = nn.Dropout(p=drop_p)

        self.act_func = nn.ReLU()

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        # Reshape node features to 2d
        x = x.reshape(x.shape[0], -1)
        # Pad global feature to be able to run each node in parallel. OBS does not work for varying graphs
        u = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)

        u = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        x = self.act_func(self.conv1(torch.cat([x, u], dim=1), edge_index))
        # x = self.dropout(x)
        # TODO replicate u to match size of nodes
        # x = self.act_func(self.fc_merge(torch.cat([x, u], dim=1)))
        # x = self.dropout(x)

        # x = self.act_func(self.conv2(x, edge_index))
        x = self.act_func(self.fc(x))
        x = self.fc_out(x)

        return x


class MLP_segments(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        input_size: int,
        global_size: int,
        hidden_layers: list[int],
        lr: float,
        weight_decay: float,
        batch_size: int,
        drop_p: float = 0.2,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)

        self.linear1 = nn.Linear(input_size + global_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        self.act_func = nn.ReLU()

    def forward(self, x: Tensor, u: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        x = x.reshape(x.shape[0], -1)

        u = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)

        x = self.act_func(self.linear1(torch.cat([x, u], dim=1)))
        x = self.act_func(self.linear2(x))
        x = self.linear3(x)
        return x


class NodeEncodedGCN(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_in2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_in3 = nn.Linear(hidden_layers[0], hidden_layers[0])

        self.conv1 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv2 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv3 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_out1 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.act_func(self.fc_in2(h))  # comment out in tt
        h = self.dropout(h)
        h = self.fc_in3(h)  # comment out in tt
        h = self.batch_norm1(h)  # comment out in tt
        h = self.act_func(h)  # comment out in tt

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))
        h = self.act_func(self.conv2(x=h, edge_index=edge_index))
        h = self.act_func(self.conv3(x=h, edge_index=edge_index))  # comment out in tt

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.fc_merge(h)
        h = self.batch_norm2(h)
        h = self.act_func(h)

        # last linear module
        h = self.act_func(self.fc_out1(h))  # comment out in tt
        h = self.dropout(h)  # comment out in tt
        h = self.fc_last(h)
        return h


class NodeEncodedGCN_tt(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_in2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_in3 = nn.Linear(hidden_layers[0], hidden_layers[0])

        self.conv1 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv2 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv3 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_out1 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.dropout(h)

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))
        h = self.act_func(self.conv2(x=h, edge_index=edge_index))

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.fc_merge(h)
        h = self.batch_norm2(h)
        h = self.act_func(h)

        # last linear module
        h = self.fc_last(h)
        return h


class NodeEncodedGAT(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_in2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_in3 = nn.Linear(hidden_layers[0], hidden_layers[0])

        self.conv1 = GAT(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv2 = GAT(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv3 = GAT(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_out1 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.act_func(self.fc_in2(h))
        h = self.dropout(h)
        h = self.fc_in3(h)
        h = self.batch_norm1(h)
        h = self.act_func(h)

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))
        h = self.act_func(self.conv2(x=h, edge_index=edge_index))
        h = self.act_func(self.conv3(x=h, edge_index=edge_index))

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.fc_merge(h)
        h = self.batch_norm2(h)
        h = self.act_func(h)

        # last linear module
        h = self.act_func(self.fc_out1(h))
        h = self.dropout(h)
        h = self.fc_last(h)
        return h


class NodeEncodedGCN_3l(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_in2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_in3 = nn.Linear(hidden_layers[0], hidden_layers[0])

        self.conv1 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv2 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv3 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_out1 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.act_func(self.fc_in2(h))
        h = self.dropout(h)
        h = self.fc_in3(h)
        h = self.batch_norm1(h)
        h = self.act_func(h)

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))
        h = self.act_func(self.conv2(x=h, edge_index=edge_index))
        h = self.act_func(self.conv3(x=h, edge_index=edge_index))

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.fc_merge(h)
        h = self.batch_norm2(h)
        h = self.act_func(h)

        # last linear module
        h = self.act_func(self.fc_out1(h))
        h = self.dropout(h)
        h = self.fc_last(h)
        return h


class NodeEncodedGCN_2l(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_in2 = nn.Linear(hidden_layers[0], hidden_layers[0])

        self.conv1 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)
        self.conv2 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.act_func(self.fc_in2(h))
        h = self.dropout(h)

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))
        h = self.act_func(self.conv2(x=h, edge_index=edge_index))

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.act_func(self.fc_merge(h))
        h = self.dropout(h)

        # last linear module
        h = self.fc_last(h)
        return h


class NodeEncodedGCN_1l(BaseModelClass):
    def __init__(
        self,
        transform: Standardize,
        weight_decay: float,
        batch_size: int,
        lr: float,
        drop_p: float,
        input_size: int,
        hidden_layers: list[int],
        aggregation_function: str,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_size=batch_size, transform=transform)
        self.save_hyperparameters()

        self.fc_in1 = nn.Linear(input_size, hidden_layers[0])

        self.conv1 = GCNConv(hidden_layers[0], hidden_layers[0], aggr=aggregation_function)

        self.fc_merge = nn.Linear(hidden_layers[0] + input_size, hidden_layers[0])
        self.fc_last = nn.Linear(hidden_layers[0], 1)

        self.dropout = nn.Dropout(p=drop_p)
        self.act_func = nn.ReLU()
        self.batch_norm1 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding
        self.batch_norm2 = BatchNorm(hidden_layers[0])  # TODO does this mess with the onehotencoding

    def forward(self, x: Tensor, u: Tensor, node_encoding: Tensor, edge_index: Tensor, batch_size: int) -> Tensor:
        lags = x[..., 0]
        global_feat = u.unsqueeze(1).repeat(1, int(x.shape[0] / batch_size), 1).reshape(x.shape[0], -1)
        merged_features = torch.cat([lags, global_feat, node_encoding], dim=1)

        # First linear module
        h = self.act_func(self.fc_in1(merged_features))
        h = self.dropout(h)

        # GCN modules
        h = self.act_func(self.conv1(x=h, edge_index=edge_index))

        # skip connection and merge module
        h = torch.cat([h, merged_features], dim=1)
        h = self.act_func(self.fc_merge(h))
        h = self.dropout(h)

        # last linear module
        h = self.fc_last(h)
        return h

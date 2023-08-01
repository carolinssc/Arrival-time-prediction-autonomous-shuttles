import torch
import torch.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GAT, GCNConv
from torch_geometric.nn.norm import BatchNorm

from data.datamodule import Standardize

from .model_base_class import BaseModelClass


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

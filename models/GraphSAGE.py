import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GSAGE(torch.nn.Module):
    def __init__(
        self,
        input_feat_dim,
        conv_shapes,
        cls_shapes,
        dropout_rate=None,
        last_activation="sigmoid",
    ):
        super(GSAGE, self).__init__()

        self.num_layers = len(conv_shapes)
        self.last_activation = last_activation
        self.dropout_rate = dropout_rate
        self.linear = None
        if self.last_activation == "sigmoid":
            self.num_class = 1
        else:
            self.num_class = 2

        assert (
            self.num_layers >= 1
        ), "Number of layers should be more than or equal to 1"

        if input_feat_dim != conv_shapes[0][0]:
            self.linear = nn.Linear(input_feat_dim, conv_shapes[0][0])

        self.convs = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList()
        for l in range(self.num_layers):
            self.convs.append(
                SAGEConv(
                    conv_shapes[l][0],
                    conv_shapes[l][1],
                )
            )

        self.pooling = global_mean_pool

        self.classifier = nn.Sequential()
        for idx, dim in enumerate(cls_shapes):
            if idx == 0:
                self.classifier.append(nn.Linear(conv_shapes[-1][1], dim))
                self.classifier.append(nn.ReLU())
            else:
                self.classifier.append(nn.Linear(cls_shapes[idx - 1], dim))
                self.classifier.append(nn.ReLU())

        self.classifier.append(nn.Linear(cls_shapes[-1], self.num_class))

        # for i in range(num_layers - 1):
        #     self.bns.append(torch.nn.BatchNorm1d(num_features=hidden_dim))

        # Probability of an element getting zeroed
        # self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, batched_data):

        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )

        if self.linear is not None:
            x = self.linear(x.float())

        for l in range(self.num_layers):
            x = F.relu(self.convs[l](x.float(), edge_index))
            if self.dropout_rate is not None:
                x = F.dropout(x, p=self.dropout_rate)

        x = self.pooling(x, batch)
        x = self.classifier(x)

        return x

        # # for conv, bn in zip(self.convs[:-1], self.bns):
        # #     x = F.relu(bn(conv(x, edge_index)))
        # #     if self.training:
        # #         x = F.dropout(x, p=self.dropout)

        # x = self.convs[-1](x, edge_index)
        # out = torch.sigmoid(x)

        # return out

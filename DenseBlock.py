import torch
import torch.nn as nn
import torch.nn.functional as F
import node_lib
import quantum_forest
from node_lib.odst import ODST

class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, config, num_layers,  max_features=None,
                 input_dropout=0.0, flatten_output=True, Module=ODST,feat_info=None, **kwargs):
        layers = []
        tree_dim=config.tree_dim
        num_trees = config.nTree
        for i in range(num_layers):
            oddt = Module(input_dim, num_trees, config, flatten_output=True,feat_info=feat_info, **kwargs)
            input_dim = min(input_dim + num_trees * tree_dim, max_features or float('inf'))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, num_trees, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs

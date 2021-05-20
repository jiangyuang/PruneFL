import torch
from torch import nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential
from .utils import is_conv, is_fc

__all__ = ["Conv2", "Conv4"]


class Conv2(BaseModel):
    def __init__(self, dict_module: dict = None):
        if dict_module is None:
            dict_module = dict()
            features = nn.Sequential(DenseConv2d(1, 32, kernel_size=5, padding=2),  # 32x28x28
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2),  # 32x14x14
                                     DenseConv2d(32, 64, kernel_size=5, padding=2),  # 64x14x14
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2))  # 64x7x7

            classifier = DenseSequential(DenseLinear(64 * 7 * 7, 2048, mode="fan_out"),
                                         nn.ReLU(inplace=True),
                                         DenseLinear(2048, 62, mode="fan_out"))

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(Conv2, self).__init__(binary_cross_entropy_with_logits, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        self.prunable_layers = self.param_layers
        self.prunable_layer_prefixes = self.param_layer_prefixes

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def loss(self, inputs, labels) -> torch.Tensor:
        return self.loss_func(self(inputs), labels)

    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
        return self.__class__(new_module_dict)

    def remove_empty_channels(self):
        list_in_out = []
        is_transition = False
        prev_is_transition = False
        for idx, (layer, next_layer) in enumerate(zip(self.prunable_layers, self.prunable_layers[1:] + [None])):
            # works for both conv and fc
            if is_conv(layer) and is_fc(next_layer):
                is_transition = True

            num_out, num_in = layer.weight.size()[:2]

            if idx == 0 or prev_is_transition:
                list_remain_in = "all"
            else:
                list_remain_in = set()
                for in_id in range(num_in):
                    mask_slice = layer.mask.index_select(dim=1, index=torch.tensor([in_id]))
                    if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
                        list_remain_in.add(in_id)
                if len(list_remain_in) == layer.weight.size()[1]:
                    list_remain_in = "all"

            if next_layer is None or is_transition:
                list_remain_out = "all"
            else:
                list_remain_out = set()
                for out_id in range(num_out):
                    mask_slice = layer.mask.index_select(dim=0, index=torch.tensor([out_id]))
                    if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
                        list_remain_out.add(out_id)
                if len(list_remain_out) == layer.weight.size()[0]:
                    list_remain_out = "all"

            list_in_out.append((list_remain_in, list_remain_out))

            if prev_is_transition:
                prev_is_transition = False
            if is_transition:
                prev_is_transition = True
                is_transition = False

        for ((in_indices, out_indices),
             (in_indices_next, out_indices_next),
             layer,
             next_layer) in zip(list_in_out[:-1], list_in_out[1:], self.prunable_layers[:-1],
                                self.prunable_layers[1:]):

            if out_indices == "all" or in_indices_next == "all":
                merged_indices = "all"
            else:
                merged_indices = list(out_indices.intersection(in_indices_next))

            if merged_indices != "all":
                layer.weight = nn.Parameter(layer.weight.index_select(dim=0, index=torch.tensor(merged_indices)))
                layer.mask = layer.mask.index_select(dim=0, index=torch.tensor(merged_indices))
                len_merged_indices = len(merged_indices)
                if layer.bias is not None:
                    layer.bias = nn.Parameter(layer.bias[merged_indices])
                if is_conv(layer):
                    layer.out_channels = len_merged_indices
                elif is_fc(layer):
                    layer.out_features = len_merged_indices

                next_layer.weight = nn.Parameter(
                    next_layer.weight.index_select(dim=1, index=torch.tensor(merged_indices)))
                next_layer.mask = next_layer.mask.index_select(dim=1, index=torch.tensor(merged_indices))
                if is_conv(next_layer):
                    next_layer.in_channels = len_merged_indices
                elif is_fc(next_layer):
                    next_layer.in_features = len_merged_indices


# class FEMNISTModel(BaseModel):
#     def __init__(self, dict_module: dict = None):
#         if dict_module is None:
#             dict_module = dict()
#             features = nn.Sequential(DenseConv2d(1, 32, kernel_size=5, padding=2),  # 32x28x28
#                                      nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(2, stride=2),  # 32x14x14
#                                      DenseConv2d(32, 64, kernel_size=5, padding=2),  # 64x14x14
#                                      nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(2, stride=2))  # 64x7x7
#
#             classifier = DenseSequential(DenseLinear(64 * 7 * 7, 2048, init_mode="fan_out"),
#                                          nn.ReLU(inplace=True),
#                                          DenseLinear(2048, 62, init_mode="fan_out"))
#
#             dict_module["features"] = features
#             dict_module["classifier"] = classifier
#
#         super(FEMNISTModel, self).__init__(binary_cross_entropy_with_logits, dict_module)
#
#     def collect_layers(self):
#         self.get_param_layers(self.param_layers, self.param_layer_prefixes)
#         self.prunable_layers = self.param_layers
#         self.prunable_layer_prefixes = self.param_layer_prefixes
#
#     def forward(self, inputs):
#         outputs = self.features(inputs)
#         outputs = outputs.view(outputs.size(0), -1)
#         outputs = self.classifier(outputs)
#         return outputs
#
#     def loss(self, inputs, labels) -> torch.Tensor:
#         return self.loss_func(self(inputs), labels)
#
#     def to_sparse(self):
#         new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
#         new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
#         return self.__class__(new_module_dict)
#
#     def remove_empty_channels(self):
#         list_in_out = []
#         is_transition = False
#         prev_is_transition = False
#         for idx, (layer, next_layer) in enumerate(zip(self.prunable_layers, self.prunable_layers[1:] + [None])):
#             # works for both conv and fc
#             if is_conv(layer) and is_fc(next_layer):
#                 is_transition = True
#
#             num_out, num_in = layer.weight.size()[:2]
#
#             if idx == 0 or prev_is_transition:
#                 list_remain_in = "all"
#             else:
#                 list_remain_in = set()
#                 for in_id in range(num_in):
#                     mask_slice = layer.mask.index_select(dim=1, index=torch.tensor([in_id]))
#                     if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
#                         list_remain_in.add(in_id)
#                 if len(list_remain_in) == layer.weight.size()[1]:
#                     list_remain_in = "all"
#
#             if next_layer is None or is_transition:
#                 list_remain_out = "all"
#             else:
#                 list_remain_out = set()
#                 for out_id in range(num_out):
#                     mask_slice = layer.mask.index_select(dim=0, index=torch.tensor([out_id]))
#                     if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
#                         list_remain_out.add(out_id)
#                 if len(list_remain_out) == layer.weight.size()[0]:
#                     list_remain_out = "all"
#
#             list_in_out.append((list_remain_in, list_remain_out))
#
#             if prev_is_transition:
#                 prev_is_transition = False
#             if is_transition:
#                 prev_is_transition = True
#                 is_transition = False
#
#         for ((in_indices, out_indices),
#              (in_indices_next, out_indices_next),
#              layer,
#              next_layer) in zip(list_in_out[:-1], list_in_out[1:], self.prunable_layers[:-1],
#                                 self.prunable_layers[1:]):
#
#             if out_indices == "all" or in_indices_next == "all":
#                 merged_indices = "all"
#             else:
#                 merged_indices = list(out_indices.intersection(in_indices_next))
#
#             if merged_indices != "all":
#                 layer.weight = nn.Parameter(layer.weight.index_select(dim=0, index=torch.tensor(merged_indices)))
#                 layer.mask = layer.mask.index_select(dim=0, index=torch.tensor(merged_indices))
#                 len_merged_indices = len(merged_indices)
#                 if layer.bias is not None:
#                     layer.bias = nn.Parameter(layer.bias[merged_indices])
#                 if is_conv(layer):
#                     layer.out_channels = len_merged_indices
#                 elif is_fc(layer):
#                     layer.out_features = len_merged_indices
#
#                 next_layer.weight = nn.Parameter(
#                     next_layer.weight.index_select(dim=1, index=torch.tensor(merged_indices)))
#                 next_layer.mask = next_layer.mask.index_select(dim=1, index=torch.tensor(merged_indices))
#                 if is_conv(next_layer):
#                     next_layer.in_channels = len_merged_indices
#                 elif is_fc(next_layer):
#                     next_layer.in_features = len_merged_indices

class Conv4(BaseModel):
    def __init__(self, dict_module: dict = None):
        if dict_module is None:
            dict_module = dict()
            features = nn.Sequential(DenseConv2d(3, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2),
                                     DenseConv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2),
                                     DenseConv2d(32, 32, kernel_size=3, padding=2),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2),
                                     DenseConv2d(32, 32, kernel_size=3, padding=2),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2))

            classifier = DenseLinear(in_features=32 * 6 * 6, out_features=2)

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(Conv4, self).__init__(cross_entropy, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_ids = [idx for idx, layer in enumerate(self.param_layers) if not isinstance(layer, nn.BatchNorm2d)]
        self.prunable_layers = list(self.param_layers[i] for i in prunable_ids)
        self.prunable_layer_prefixes = list(self.param_layer_prefixes[i] for i in prunable_ids)

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def loss(self, inputs, labels) -> torch.Tensor:
        return self.loss_func(self(inputs), labels)

    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features),
                           "classifier": self.classifier.to_sparse(transpose=True)}
        return self.__class__(new_module_dict)

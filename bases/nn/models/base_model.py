from abc import ABC, abstractmethod
from typing import Union, Sized

import torch
from torch import nn as nn

from .utils import traverse_module


class BaseModel(nn.Module, ABC):
    def __init__(self, loss_func, dict_module: dict):
        super(BaseModel, self).__init__()

        for module_name, module in dict_module.items():
            self.add_module(module_name, module)

        self.loss_func = loss_func
        self.param_layers: list = []
        self.param_layer_prefixes: list = []
        self.prunable_layers: list = []
        self.prunable_layer_prefixes: list = []

        self.collect_layers()

    # def load_state_dict(self, state_dict, strict=True):
    #     _IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])
    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []
    #
    #     # copy state_dict so _load_from_state_dict can modify it
    #     metadata = getattr(state_dict, '_metadata', None)
    #     state_dict = state_dict.copy()
    #     if metadata is not None:
    #         state_dict._metadata = metadata
    #
    #     def load(module, prefix=''):
    #         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    #         module._load_from_state_dict(
    #             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')
    #
    #     load(self)
    #
    #     if strict:
    #         if len(unexpected_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Unexpected key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    #         if len(missing_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Missing key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in missing_keys)))
    #
    #     if len(error_msgs) > 0:
    #         raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #             self.__class__.__name__, "\n\t".join(error_msgs)))
    #     return _IncompatibleKeys(missing_keys, unexpected_keys)

    # parameterized layers
    # @abstractmethod
    # def param_layers(self) -> list:
    #     pass
    #
    # @abstractmethod
    # def param_layer_prefixes(self) -> list:
    #     pass
    def traverse(self, criterion, layers: list, names: list):
        traverse_module(self, criterion, layers, names)

    def get_param_layers(self, layers: list, names: list, criterion=None):
        self.traverse(lambda x: len(list(x.parameters())) != 0, layers, names)

    @abstractmethod
    def collect_layers(self):
        pass

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    def loss(self, inputs, labels: torch.IntTensor) -> torch.FloatTensor:
        return self.loss_func(self(inputs), labels)

    @torch.no_grad()
    def evaluate(self, test_loader, mode="sum"):
        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        device = next(self.parameters()).device

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            batch_loss = self.loss_func(outputs, labels)
            test_loss += batch_loss.item()

            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)

            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        if mode == "mean":
            test_loss /= n_total
        self.train()
        return test_loss, n_correct / n_total

    # @torch.no_grad()
    # def apply_grad(self):
    #     for param in self.parameters():
    #         param.add_(param.grad, alpha=-self.lr)  # includes both sparse and dense

    # def step(self, inputs, labels):
    #     self.zero_grad()
    #     loss = self.loss(inputs, labels)
    #     loss.backward()
    #     self.apply_grad()

    def prune_by_threshold(self, thr_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(thr_arg, Sized):
            assert len(prunable_layers) == len(thr_arg)
        else:
            thr_arg = [thr_arg] * len(prunable_layers)
        for thr, layer in zip(thr_arg, prunable_layers):
            if thr is not None:
                layer.prune_by_threshold(thr)

        return self

    def prune_by_rank(self, rank_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.prune_by_rank(rank)

        return self

    def retain_by_rank(self, rank_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.retain_by_rank(rank)

        return self

    def prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.prune_by_pct(pct)

        return self

    def random_prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.random_prune_by_pct(pct)

        return self

    @torch.no_grad()
    def reinit_from_model(self, final_model):
        assert isinstance(final_model, self.__class__)
        for self_layer, layer in zip(self.prunable_layers, final_model.prunable_layers):
            self_layer.mask = layer.mask.clone().to(self_layer.mask.device)

    def calc_num_prunable_params(self, count_bias):
        total_param_in_use = 0
        total_param = 0
        for layer in self.prunable_layers:
            num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight
            num_params_in_use = num_weight + num_bias
            num_params = layer.weight.nelement() + num_bias
            total_param_in_use += num_params_in_use
            total_param += num_params

        return total_param_in_use, total_param

    def calc_num_all_active_params(self, count_bias):
        total_param = 0
        for layer in self.param_layers:
            num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight if hasattr(layer, "num_weight") else layer.weight.nelement()
            num_params = num_weight + num_bias
            total_param += num_params

        return total_param

    def nnz(self, count_bias=False):
        # number of parameters in use in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[0]

    def nelement(self, count_bias=False):
        # number of all parameters in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[1]

    def density(self, count_bias=False):
        total_param_in_use, total_param = self.calc_num_prunable_params(count_bias=count_bias)
        return total_param_in_use / total_param

    def _get_module_by_name_list(self, module_names: list):
        module = self
        for name in module_names:
            module = getattr(module, name)
        return module

    def get_module_by_name(self, module_name: str):
        return self._get_module_by_name_list(module_name.split('.'))

    def get_mask_by_name(self, param_name: str):
        if param_name.endswith("bias"):
            return None
        module = self._get_module_by_name_list(param_name.split('.')[:-1])
        return module.mask if hasattr(module, "mask") else None

    @abstractmethod
    def to_sparse(self):
        pass

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            # move data to device
            for m in self.prunable_layers:
                m.move_data(device)
        return super(BaseModel, self).to(*args, **kwargs)

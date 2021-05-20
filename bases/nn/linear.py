import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.init as init
from torch.nn.parameter import Parameter
from bases.autograd.functions import AddmmFunction

import math


class SparseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, weight: sparse.FloatTensor, bias, mask, transpose=False):
        super(SparseLinear, self).__init__()
        if not weight.is_sparse:
            raise ValueError("Weight must be sparse")
        elif weight._nnz() > 0 and not weight.is_coalesced():
            raise ValueError("Weight must be coalesced")

        self.transpose = transpose

        self.in_features = weight.size(1)
        self.out_features = weight.size(0)
        self.mask = mask.clone()

        # in order to add to optimizer
        self.weight = Parameter(weight.data.clone(), requires_grad=False)
        # Don't move after creation to make it a leaf
        self.dense_weight_placeholder = Parameter(torch.empty(size=self.weight.size(), device=self.weight.device))
        self.dense_weight_placeholder.is_placeholder = True

        # create links
        self.weight.dense = self.dense_weight_placeholder
        self.weight.mask = self.mask
        self.weight.is_sparse_param = True

        if bias is None:
            self.register_parameter('bias', None)
        else:
            assert bias.size() == torch.Size((weight.size(0), 1))
            self.bias = Parameter(bias.data.clone())

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(SparseLinear, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
    #                                                     missing_keys, unexpected_keys, error_msgs)
    #
    #     assert hasattr(state_dict[prefix + "weight"], "mask")
    #     self.mask = state_dict[prefix + "weight"].mask

    def _sparse_masked_select_abs(self, sparse_tensor: sparse.FloatTensor, thr):
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
        prune_mask = torch.abs(values) >= thr
        return torch.sparse_coo_tensor(indices=indices.masked_select(prune_mask).reshape(2, -1),
                                       values=values.masked_select(prune_mask),
                                       size=[self.out_features, self.in_features]).coalesce()

    def prune_by_threshold(self, thr):
        self.weight = Parameter(self._sparse_masked_select_abs(self.weight, thr))

    def prune_by_rank(self, rank):
        weight_val = self.weight._values()
        sorted_abs_weight = torch.sort(torch.abs(weight_val))[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.weight._nnz() * pct)
        self.prune_by_rank(prune_idx)

    def move_data(self, device: torch.device):
        self.weight = self.weight.to(device)

    def forward(self, inp: torch.Tensor):
        if self.transpose:
            return AddmmFunction.apply(self.bias, self.weight, self.dense_weight_placeholder, inp.t()).t()
        else:
            return AddmmFunction.apply(self.bias, self.weight, self.dense_weight_placeholder, inp)

    @property
    def num_weight(self) -> int:
        return self.weight._nnz()

    def __repr__(self):
        return "SparseLinear(in_features={}, out_features={}, bias={}, transpose = {})".format(self.in_features,
                                                                                               self.out_features,
                                                                                               self.bias is not None,
                                                                                               self.transpose)

    def __str__(self):
        return self.__repr__()


class DenseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, use_bias=True, use_mask=True, **kwargs):
        super(DenseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if use_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(**kwargs)

        # self._initial_weight = self.weight.data.clone()
        # self._initial_bias = self.bias.data.clone() if use_bias else None
        self.use_mask = use_mask
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)

    def reset_parameters(self, **kwargs):
        if len(kwargs.keys()) == 0:
            # default init, see https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            init.kaiming_uniform_(self.weight, **kwargs)

        if self.bias is not None:
            # default init, see https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp: torch.Tensor):
        masked_weight = self.weight * self.mask if self.use_mask else self.weight
        return nn.functional.linear(inp, masked_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def prune_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weight_val = self.weight[self.mask == 1.]
        sorted_abs_weight = weight_val.abs().sort()[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def retain_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def retain_by_rank(self, rank):
        weights_val = self.weight[self.mask == 1.]
        sorted_abs_weights = weights_val.abs().sort(descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand(size=self.mask.size(), device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = rand_val.sort()[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    # def reinitialize(self):
    #     self.weight = Parameter(self._initial_weight)
    #     if self._initial_bias is not None:
    #         self.bias = Parameter(self._initial_bias)

    def to_sparse(self, transpose=False) -> SparseLinear:
        """
        by chance, some entries with mask = 1 can have a 0 value. Thus, the to_sparse methods give a different size
        there's no efficient way to solve it yet
        """
        sparse_bias = None if self.bias is None else self.bias.reshape((-1, 1))
        sparse_linear = SparseLinear((self.weight * self.mask).to_sparse(), sparse_bias, self.mask)
        if transpose:
            sparse_linear.transpose = True
        return sparse_linear

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]

        if device is not None:
            self.move_data(device)

        return super(DenseLinear, self).to(*args, **kwargs)

    @property
    def num_weight(self) -> int:
        return self.mask.sum().item()

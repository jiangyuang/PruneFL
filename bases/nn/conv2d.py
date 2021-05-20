import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Parameter
from bases.autograd.functions import SparseConv2dFunction, DenseConv2dFunction


class SparseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight, bias, mask):
        super(SparseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mask = mask.clone()

        self.weight = Parameter(weight.clone(), requires_grad=False)
        self.dense_weight_placeholder = Parameter(torch.empty(size=self.weight.size()))
        self.dense_weight_placeholder.is_placeholder = True

        self.weight.dense = self.dense_weight_placeholder
        self.weight.mask = self.mask
        self.weight.is_sparse_param = True

        if bias is None:
            self.bias = torch.zeros(size=(out_channels,))
        else:
            self.bias = Parameter(bias.clone())

    def forward(self, inp):
        return SparseConv2dFunction.apply(inp, self.weight, self.dense_weight_placeholder, self.kernel_size,
                                          self.bias, self.stride, self.padding)

    # @classmethod
    # def from_conv2d(cls, conv: torch.nn.Conv2d):
    #     weight = conv.weight.data.view(conv.out_channels, -1).to_sparse()
    #     return cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, None,
    #                weight, conv.bias)
    #
    # @classmethod
    # def from_masked_conv2d(cls, conv: DenseConv2d):
    #     weight = conv.weight.data * conv.mask
    #     weight = weight.view(conv.out_channels, -1).to_sparse()
    #     return cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, None,
    #                weight, conv.bias)

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(SparseConv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
    #                                                     missing_keys, unexpected_keys, error_msgs)
    #
    #     assert hasattr(state_dict[prefix + "weight"], "mask")
    #     self.mask = state_dict[prefix + "weight"].mask

    @property
    def num_weight(self):
        return self.weight._nnz()

    def __repr__(self):
        return "SparseConv2d({}, {}, kernel_size={}, stride={}, padding={})".format(self.in_channels, self.out_channels,
                                                                                    self.kernel_size, self.stride,
                                                                                    self.padding)

    def __str__(self):
        return self.__repr__()


class DenseConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True,
                 padding_mode='zeros', mask: torch.FloatTensor = None, use_mask=True):
        super(DenseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                          dilation, groups, use_bias, padding_mode)
        if mask is None:
            self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        else:
            self.mask = mask
            assert self.mask.size() == self.weight.size()

        self.use_mask = use_mask

        # self._initial_weight = self.weight.clone()
        # self._initial_bias = self.bias.clone() if isinstance(self.bias, torch.Tensor) else None

    # def conv2d_forward(self, inp, weight):
    #     if self.padding_mode == 'circular':
    #         expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
    #                             (self.padding[0] + 1) // 2, self.padding[0] // 2)
    #         inp = F.pad(inp, expanded_padding, mode='circular')
    #         padding = _pair(0)
    #     else:
    #         padding = self.padding
    #
    #     return DenseConv2dFunction.apply(inp, weight, self.kernel_size, self.bias, self.stride, padding)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, inp):
        masked_weight = self.weight * self.mask if self.use_mask else self.weight

        return self.conv2d_forward(inp, masked_weight)

    def prune_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def retain_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def retain_by_rank(self, rank):
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val), descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand_like(self.mask, device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    #  by chance, some entries with mask = 1 can have a 0 value. Thus, the to_sparse methods give a different size
    #  there's no efficient way to solve it yet
    def to_sparse(self):
        weight = (self.weight * self.mask).view(self.out_channels, -1).to_sparse()
        return SparseConv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, weight,
                            self.bias, self.mask.view(self.out_channels, -1))

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    @property
    def num_weight(self):
        return torch.sum(self.mask).int().item()

import torch
from bases.nn.linear import DenseLinear


class SparseSequential(torch.nn.Sequential):
    def forward(self, inp: torch.Tensor):
        inp = inp.t().contiguous()
        for module in self._modules.values():
            inp = module(inp)
        inp = inp.t().contiguous()
        return inp


class DenseSequential(torch.nn.Sequential):
    def to_sparse(self):
        sparse_layers = [layer.to_sparse() if isinstance(layer, DenseLinear) else layer for layer in self]
        return SparseSequential(*sparse_layers)

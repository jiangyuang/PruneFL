from bases.nn.conv2d import DenseConv2d, SparseConv2d
from bases.nn.linear import DenseLinear, SparseLinear


def is_fc(layer):
    return isinstance(layer, DenseLinear) or isinstance(layer, SparseLinear)


def is_conv(layer):
    return isinstance(layer, DenseConv2d) or isinstance(layer, SparseConv2d)


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")

import pickle
import copyreg
import io
import os
import torch
import numpy as np
from PIL import Image

import warnings

bytes_types = (bytes, bytearray)


# mode 0: store indices
# mode 1: store bitmap
def get_mode(x) -> bool:
    return False if x._nnz() / x.nelement() < 1 / 32 else True


def get_int_type(max_val: int):
    assert max_val >= 0
    max_uint8 = 1 << 8
    max_int16 = 1 << 15
    max_int32 = 1 << 31
    if max_val < max_uint8:
        return torch.uint8
    elif max_val < max_int16:
        return torch.int16
    elif max_val < max_int32:
        return torch.int32
    else:
        return torch.int64


def sparse_coo_from_indices(indices, values, size):
    mask = torch.zeros(size=size, dtype=torch.bool)
    mask[indices.tolist()] = True
    tensor = torch.sparse_coo_tensor(indices.to(torch.long), values, size).coalesce()
    tensor.mask = mask
    return tensor


def sparse_coo_from_values_bitmap(bitmap, values, size):
    mask = torch.from_numpy(np.array(bitmap, np.uint8, copy=False))
    indices = mask.nonzero().t()
    tensor = torch.sparse_coo_tensor(indices.to(torch.long), values, size).coalesce()
    tensor.mask = mask
    return tensor


def rebuild_dispatcher(mode, arg0, arg1, arg2):
    if mode is False:
        return sparse_coo_from_indices(arg0, arg1, arg2)
    else:
        return sparse_coo_from_values_bitmap(arg0, arg1, arg2)


def args_dispatcher(mode, x) -> tuple:
    # supports only 2 dimensional tensors
    if mode is False:
        int_type = get_int_type(torch.max(x._indices()).item())
        return mode, x._indices().to(int_type), x._values(), x.size()
    else:
        bitmap = torch.zeros(size=x.size(), dtype=torch.bool)
        bitmap[x._indices().tolist()] = True
        bitmap = Image.fromarray(bitmap.numpy())
        assert bitmap.mode == "1"
        return mode, bitmap, x._values(), x.size()


def reduce(x: torch.Tensor):
    if x.is_sparse:
        assert x.ndim == 2, "Only 2-dimensional tensors are supported"
        mode = get_mode(x)
        return rebuild_dispatcher, args_dispatcher(mode, x)
    else:
        return x.__reduce_ex__(pickle.DEFAULT_PROTOCOL)


# register custom reduce function for sparse tensors
copyreg.pickle(torch.Tensor, reduce)


def dumps(obj):
    f = io.BytesIO()
    pickle.dump(obj, f)
    res = f.getvalue()
    assert isinstance(res, bytes_types)
    return res


def loads(res):
    return pickle.loads(res)


def save(obj, f):
    # disabling warnings from torch.Tensor's reduce function. See issue: https://github.com/pytorch/pytorch/issues/38597
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(f, "wb") as opened_f:
            pickle.dump(obj, opened_f)


def mkdir_save(obj, f):
    dir_name = os.path.dirname(f)
    if dir_name == "":
        save(obj, f)
    else:
        os.makedirs(dir_name, exist_ok=True)
        save(obj, f)


def load(f):
    with open(f, 'rb') as opened_f:
        return pickle.load(opened_f)

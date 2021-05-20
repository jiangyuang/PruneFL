import torch
from torch.nn.functional import one_hot


class Flatten:
    def __call__(self, img: torch.FloatTensor):
        return img.reshape((-1))


class OneHot:
    def __init__(self, n_classes, to_float: bool = False):
        self.n_classes = n_classes
        self.to_float = to_float

    def __call__(self, label: torch.Tensor):
        return one_hot(label, self.n_classes).float() if self.to_float else one_hot(label, self.n_classes)


class DataToTensor:
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = torch.float
        self.dtype = dtype

    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)

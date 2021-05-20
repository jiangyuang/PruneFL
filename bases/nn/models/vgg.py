from torch import nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential

__all__ = ["VGG11"]


class VGG11(BaseModel):
    def __init__(self, dict_module: dict = None):
        if dict_module is None:
            dict_module = dict()
            self.batch_norm = False
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

            features = self._make_feature_layers()
            classifier = DenseSequential(DenseLinear(512, 512, a=0),
                                         nn.ReLU(inplace=True),
                                         DenseLinear(512, 512, a=0),
                                         nn.ReLU(inplace=True),
                                         DenseLinear(512, 10, a=1.5, mode="fan_out"))

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(VGG11, self).__init__(binary_cross_entropy_with_logits, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers) if not isinstance(ly, nn.BatchNorm2d)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]

    def _make_feature_layers(self):
        layers = []
        in_channels = 3
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if self.batch_norm:
                    layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(param),
                                   nn.ReLU(inplace=True)])
                else:
                    layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
        return self.__class__(new_module_dict)

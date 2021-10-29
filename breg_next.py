# Copyright (c) 2021 Regents of the University of California
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Code taken from https://github.com/behzadhsni/BReG-NeXt

import torch

class BRegNextShortcutModifier(torch.nn.Module):

    def __init__(self,):
        super(BRegNextShortcutModifier, self).__init__()

        self._a = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self._c = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, inputs):
        numl = torch.atan((self._a * inputs) / torch.sqrt(self._c ** 2 + 1))
        denom = self._a * torch.sqrt(self._c ** 2 + 1)
        return  (numl / denom)


class BReGNeXtResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample_stride = 1):
        super(BReGNeXtResidualLayer, self).__init__()

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._downsample_stride = downsample_stride

        self._conv0 = torch.nn.Conv2d(in_channels, out_channels, 3, downsample_stride)
        torch.nn.init.kaiming_uniform_(self._conv0.weight)
        self._conv1 = torch.nn.Conv2d(out_channels, out_channels, 3, 1)
        torch.nn.init.kaiming_uniform_(self._conv1.weight)
        self._shortcut = BRegNextShortcutModifier()
        self._batchnorm_conv0 = torch.nn.BatchNorm2d(self._in_channels)
        self._batchnorm_conv1 = torch.nn.BatchNorm2d(self._out_channels)

    def forward(self, inputs):
        # First convolution
        normed_inputs = inputs if self._batchnorm_conv0 is None else self._batchnorm_conv0(inputs)
        normed_inputs = torch.nn.functional.elu(normed_inputs)
        normed_inputs = torch.nn.functional.pad(normed_inputs, (1,1,1,1,0,0))
        conv0_outputs = self._conv0(normed_inputs)

        # Second convolution
        normed_conv0_outputs = conv0_outputs if self._batchnorm_conv1 is None else self._batchnorm_conv1(conv0_outputs)
        normed_conv0_outputs = torch.nn.functional.elu(normed_conv0_outputs)
        normed_conv0_outputs = torch.nn.functional.pad(normed_conv0_outputs, (1,1,1,1,0,0))
        conv1_outputs = self._conv1(normed_conv0_outputs)


        shortcut_modifier = self._shortcut(inputs)
        if self._downsample_stride > 1:
            shortcut_modifier = torch.nn.functional.avg_pool2d(shortcut_modifier, self._downsample_stride, self._downsample_stride)

        # Upsample the shortcut in the channel dimension if necessary
        if self._out_channels > self._in_channels:
            pad_dimension = (self._out_channels - self._in_channels) // 2
            shortcut_modifier = torch.nn.functional.pad(shortcut_modifier, [0,0,0,0,pad_dimension, pad_dimension])
            # NOTE: This code doesn't handle the case if _out_channels < _in_channels

        return conv1_outputs + shortcut_modifier


class BRegNextResidualBlock(torch.nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, downsample_stride=1):
        super(BRegNextResidualBlock, self).__init__()
        layers = [BReGNeXtResidualLayer(in_channels, out_channels, downsample_stride)] + [
            BReGNeXtResidualLayer(out_channels, out_channels, downsample_stride) for _ in range(n_blocks - 1)
        ]
        self._layer_stack = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._layer_stack(inputs)


class BReGNeXt(torch.nn.Module):

    def __init__(self, n_classes: int = 8) -> None:

        super(BReGNeXt, self).__init__()

        self._conv0 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        self._model = torch.nn.Sequential(
            # NOTE: The original BReGNeXt code uses a truncated normal initialization for this convolution, however
            # that is not implemented in PyTorch 1.7 - This defaults to a uniform initializer in PyTorch.

            BRegNextResidualBlock(n_blocks=7, in_channels=32, out_channels=32),
            BRegNextResidualBlock(n_blocks=1, in_channels=32, out_channels=64, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=8, in_channels=64, out_channels=64),
            BRegNextResidualBlock(n_blocks=1, in_channels=64, out_channels=128, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=7, in_channels=128, out_channels=128),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
        )
        self._fc0 = torch.nn.Linear(128, n_classes)

    def forward(self, x):
        # These two blocks simulate "SAME" padding from TensorFlow
        net = torch.nn.functional.pad(x, (1,1,1,1,0,0))
        net = self._conv0(net)
        # Handle the rest of the net
        return self._fc0(self._model(net).reshape(-1, 128))



class multi_BReGNeXt(torch.nn.Module):

    def __init__(self) -> None:

        super(multi_BReGNeXt, self).__init__()

        self._conv0 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        self._model = torch.nn.Sequential(
            # NOTE: The original BReGNeXt code uses a truncated normal initialization for this convolution, however
            # that is not implemented in PyTorch 1.7 - This defaults to a uniform initializer in PyTorch.

            BRegNextResidualBlock(n_blocks=7, in_channels=32, out_channels=32),
            BRegNextResidualBlock(n_blocks=1, in_channels=32, out_channels=64, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=8, in_channels=64, out_channels=64),
            BRegNextResidualBlock(n_blocks=1, in_channels=64, out_channels=128, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=7, in_channels=128, out_channels=128),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
        )

        self.fc_cont = torch.nn.Linear(128, 2)
        self.fc_cat = torch.nn.Linear(128, 7)

    def forward(self, x):
        # These two blocks simulate "SAME" padding from TensorFlow
        net = torch.nn.functional.pad(x, (1,1,1,1,0,0))
        net = self._conv0(net)
        # Handle the rest of the net
        feat = self._model(net).reshape(-1, 128)
        x_cat = self.fc_cat(feat)
        x_cont = self.fc_cont(feat)

        return x_cat, x_cont
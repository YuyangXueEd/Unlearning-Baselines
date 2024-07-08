import torch
from torch import nn

from torch.hub import load_state_dict_from_url

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution with padding."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"

    def normalize_fn(self, tensor, mean, std):
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        imagenet=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        print("The normalize layer is contained in the network")
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )

        if not imagenet:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet18(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=progress)
            self.load_state_dict(state_dict)


class ResNet34(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet34"], progress=progress)
            self.load_state_dict(state_dict)


class ResNet50(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=progress)
            self.load_state_dict(state_dict)


class ResNet101(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet101"], progress=progress)
            self.load_state_dict(state_dict)


class ResNet152(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet152"], progress=progress)
            self.load_state_dict(state_dict)


class ResNext50_32x4d(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnext50_32x4d"], progress=progress)
            self.load_state_dict(state_dict)


class ResNext101_32x8d(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnext101_32x8d"], progress=progress)
            self.load_state_dict(state_dict)

class WideResNet50_2(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2, num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["wide_resnet50_2"], progress=progress)
            self.load_state_dict(state_dict)


class WideResNet101_2(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, progress=True, **kwargs):
        super().__init__(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2, num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["wide_resnet101_2"], progress=progress)
            self.load_state_dict(state_dict)

if __name__ == "__main__":
    _ = ResNet(Bottleneck, [3, 4, 6, 3])

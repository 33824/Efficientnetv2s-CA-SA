import torch
import torch.nn as nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, se_ratio=0.25):
        super().__init__()
        self.se_reduce = nn.Conv2d(in_channels, int(out_channels * se_ratio), kernel_size=1)
        self.se_expand = nn.Conv2d(int(out_channels * se_ratio), out_channels, kernel_size=1)

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.se_expand(F.relu(self.se_reduce(x_se)))
        return torch.sigmoid(x_se) * x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se)
        return x_se * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.fc1(x)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se)
        return x_se * x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, reduction_ratio=16, drop_connect_rate=0.2):
        super().__init__()

        self.expand_channels = int(in_channels * expand_ratio)
        self.use_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.use_sa = True
        self.id_skip = (stride == 1 and in_channels == out_channels)

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, self.expand_channels, kernel_size=1)
            self.bn0 = nn.BatchNorm2d(self.expand_channels)
        else:
            self.expand_conv = None
            self.bn0 = None

        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv2d(
            self.expand_channels, self.expand_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=self.expand_channels
        )
        self.bn1 = nn.BatchNorm2d(self.expand_channels)

        # Squeeze and Excitation phase
        if self.use_se:
            self.se = SqueezeExcitation(self.expand_channels, self.expand_channels, se_ratio=se_ratio)

        # Spatial Attention phase
        if self.use_sa:
            self.sa = SpatialAttention(self.expand_channels, reduction_ratio=reduction_ratio)

        # Channel Attention phase
        self.ca = ChannelAttention(self.expand_channels, reduction_ratio=reduction_ratio)

        # Output phase
        self.project_conv = nn.Conv2d(self.expand_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Dropout
        self.drop_connect_rate = drop_connect_rate

    def _drop_connect(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return binary_tensor * x / keep_prob

    def forward(self, inputs):
        x = inputs

        # Expansion phase
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = F.relu(x)

        # Depthwise convolution phase
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Squeeze and Excitation phase
        if self.use_se:
            x = self.se(x)

        # Spatial Attention phase
        if self.use_sa:
            x_sa = self.sa(x)
            x = x * x_sa

        # Channel Attention phase
        x_ca = self.ca(x)
        x = x * x_ca

        # Output phase
        x = self.project_conv(x)
        x = self.bn2(x)

        # Skip connection and drop connect
        if self.id_skip:
            if self.drop_connect_rate > 0:
                x = self._drop_connect(x)
            x = x + inputs

        return x


class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, reduction_ratio=16):
        super().__init__()

        # Settings for each block of MBConv blocks
        settings = [
            # c, k, s, e, r
            [24, 3, 1, 1, None],
            [24, 3, 2, 4, None],
            [40, 5, 2, 4, None],
            [80, 3, 2, 4, 0.25],
            [80, 3, 1, 6, 0.25],
            [120, 5, 2, 6, 0.25],
            [200, 3, 1, 6, 0.25],
            [200, 5, 2, 6, 0.25],
            [480, 3, 1, 3, 0.25],
            [480, 3, 1, 6, 0.25],
            [480, 5, 2, 6, 0.25],
        ]

        # Parameters for scaling each block of MBConv blocks
        scale_params = [
            # width, depth, res
            [1.0, 1.0, 1.0],
            [1.0, 1.1, 1.2],
            [1.1, 1.2, 1.4],
            [1.2, 1.4, 1.8],
            [1.4, 1.8, 2.2],
            [1.6, 2.2, 2.6],
            [1.8, 2.6, 3.1],
            [2.0, 3.1, 3.6],
            [2.2, 3.6, 4.1],
            [2.4, 4.1, 4.6],
            [2.6, 4.6, 5.1],
        ]

        # Calculate the number of channels for each block of MBConv blocks
        channels = [int(24 * width_mult)]
        for width, depth, res in scale_params:
            channels.append(int(width * width_mult))

        # Construct the layers
        layers = []
        in_channels = 3
        out_channels = channels[0]
        for i, (c, k, s, e, r) in enumerate(settings):
            out_channels = channels[i + 1]
            layers.append(MBConvBlock(in_channels, out_channels, k, s, e, se_ratio=r, reduction_ratio=reduction_ratio))
            in_channels = out_channels

        # Construct the classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, num_classes),
        )

        # Construct the network
        self.layers = nn.Sequential(*layers)

        # Initialize the weights
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                nn.init.normal_(m.weight, mean=0, std=1 / fan_out)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

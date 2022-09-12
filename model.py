import torch
from torch_geometric.nn import NNConv
import torch
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    Sequential,
    ReLU,
    Linear,
    Module,
    ConvTranspose2d,
    Upsample,
    MaxPool2d,
)


class DoubleConv(Module):
    """
    (convolution => [BN] => ReLU) * 2
    Taken from https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(Module):
    """
    Downscaling with maxpool then double conv
    Taken from https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = Sequential(
            MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(Module):
    """
    Upscaling then double conv
    Taken from https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(Module):
    """
    Taken from https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class UNet(Module):
    """
    Taken from https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class DGN(Module):
    """
    Taken from https://github.com/basiralab/DGN
    """

    def __init__(self, n_views, conv1, conv2, conv3):
        super().__init__()
        nn = Sequential(Linear(n_views, conv1), ReLU())
        self.conv1 = NNConv(1, conv1, nn, aggr="mean")

        nn = Sequential(Linear(n_views, conv1 * conv2), ReLU())
        self.conv2 = NNConv(conv1, conv2, nn, aggr="mean")

        nn = Sequential(Linear(n_views, conv2 * conv3), ReLU())
        self.conv3 = NNConv(conv2, conv3, nn, aggr="mean")

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        repeated_out = x.repeat(x.shape[0], 1, 1)
        repeated_t = torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)

        return cbt

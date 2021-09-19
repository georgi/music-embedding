from typing import Tuple
import torch
from torch import nn
from torch.nn.functional import mse_loss
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def init_weights(mod: nn.Module):
    for m in mod.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Tuple[int, int],
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, kernel_size: Tuple[int, int], stride: int):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        if stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

        init_weights(self)

    def forward(self, x: torch.Tensor):
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


class Encoder(nn.Module):
    def __init__(self, planes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, planes, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicBlock(planes, planes, kernel_size=(3, 3), stride=2),
            BasicBlock(planes, planes, kernel_size=(3, 3), stride=2),
            BasicBlock(planes, planes, kernel_size=(3, 3), stride=2),
        )
        init_weights(self.encoder)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, planes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(planes, planes, kernel_size=(5, 5), stride=(2, 2), padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, kernel_size=(5, 5), stride=(2, 2), padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, kernel_size=(5, 3), stride=(2, 2), padding=(0, 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, kernel_size=(5, 3), stride=(1, 2), padding=(0, 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, kernel_size=(5, 2), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, 1, kernel_size=(4, 1), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(True),
        )
        init_weights(self.decoder)

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class AutoEncoder(LightningModule):
    encoder: Encoder
    decoder: Decoder
    lr: float

    def __init__(self, planes: int = 64, lr : float = 0.1):
        super().__init__()
        self.encoder = Encoder(planes)
        self.decoder = Decoder(planes)
        self.lr = lr

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch: torch.Tensor, _: int):
        y_hat = self.forward(batch)
        loss = mse_loss(y_hat, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: torch.Tensor, _: int):
        y_hat = self.forward(batch)
        loss = mse_loss(y_hat, batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }




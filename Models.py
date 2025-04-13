import torch.nn as nn
import math

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.prelu = nn.PReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.prelu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    return out + residual

class SRResNet(nn.Module):
  def __init__(self, scale_factor=4, num_channels=3, num_filters=64, large_kernel_size=9, small_kernel_size=3, num_res_blocks=16):
    super(SRResNet, self).__init__()

    self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=large_kernel_size, stride=1, padding=large_kernel_size//2)
    self.prelu = nn.PReLU()

    self.res_blocks = nn.Sequential(
        *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
    )

    self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=small_kernel_size, stride=1, padding=small_kernel_size//2)
    self.bn2 = nn.BatchNorm2d(num_filters)

    upsample_layers = []
    n_subpixel_blocks = int(math.log2(scale_factor))
    for _ in range(n_subpixel_blocks):
      upsample_layers.append(nn.Conv2d(num_filters, num_filters*4, kernel_size=small_kernel_size, stride=1, padding=small_kernel_size//2))
      upsample_layers.append(nn.PixelShuffle(2))
      upsample_layers.append(nn.PReLU())
    self.upsample = nn.Sequential(*upsample_layers)

    self.conv3 = nn.Conv2d(num_filters, num_channels, kernel_size=large_kernel_size, stride=1, padding=large_kernel_size//2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.prelu(x)
    residual = x

    y = self.res_blocks(x)

    y = self.conv2(y)
    y = self.bn2(y)
    y = y + residual

    y = self.upsample(y)

    y = self.conv3(y)
    
    return y
  
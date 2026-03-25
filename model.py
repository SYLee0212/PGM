from polars import groups
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,groups=1,relu=True,bn=True,bias=False,):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ASAM(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.ConvBlock = BasicConv(
            2,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
        )
    def forward(self, x, p):
        x = torch.cat((torch.max(x, dim=1)[0].unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)),dim=1,)
        x_out = self.ConvBlock(x)
        return torch.sigmoid(x_out) * p
    
class PGMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_l1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fusion_l2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x, p):
        x = F.interpolate(x, size=p.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, p], dim=1)
        x = self.fusion_l1(x)
        x = self.fusion_l2(x)
        return x

class PGM(nn.Module):
    """
    Input:
      x:    [B, 1, H, W]
      brpi: [B, 1, H2, W2]
    Output embedding:
      [B, D]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

        self.img_l2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.img_l3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.brpi_l1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.brpi_l2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.Module_ASAM = ASAM()
        self.Module_PGMF = PGMF()
    def forward(self, x, p):
        p = p.to(x.device, dtype=x.dtype)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.img_l2(x)
        p = self.brpi_l1(p)
        p = self.brpi_l2(p)
        p = self.Module_ASAM(x, p)
        x = self.img_l3(x)
        x = self.Module_PGMF(x, p)
        return x.flatten(1)

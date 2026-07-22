"""Modules used by literature-comparison model configurations."""

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import DeformConv2d

from .block import C3k2
from .conv import Conv, DWConv
from .conv import autopad
from .transformer import AIFI

__all__ = (
    "AdaptiveSparseAIFI",
    "C3k2NAM",
    "CoordAtt",
    "DCNv2",
    "ECA",
    "ECASpatial",
    "EMA",
    "FastEMA",
    "FrequencyGuidedAttention",
    "IncepMix",
    "ParallelBackboneBlock",
    "WaveletFeatureRefinement",
)


class C3k2NAM(C3k2):
    """YOLO11 C3k2 block followed by normalization-based channel attention."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.nam = nn.BatchNorm2d(c2, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        normalized = self.nam(y)
        weights = self.nam.weight.abs()
        weights = weights / weights.sum().clamp_min(1e-6)
        return y * torch.sigmoid(normalized * weights.view(1, -1, 1, 1))


class IncepMix(nn.Module):
    """Dynamic multi-receptive-field depthwise mixing used by the SFA-DETR proxy."""

    def __init__(self, c1: int, c2: int, kernels=(3, 5, 7), reduction: int = 4):
        super().__init__()
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.branches = nn.ModuleList(DWConv(c2, c2, k) for k in kernels)
        hidden = max(c2 // reduction, 8)
        self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, len(kernels), 1))
        self.fuse = Conv(c2, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        gates = self.selector(x).softmax(dim=1)
        mixed = sum(branch(x) * gates[:, i : i + 1] for i, branch in enumerate(self.branches))
        return x + self.fuse(mixed)


class FrequencyGuidedAttention(nn.Module):
    """Split features into low/high-frequency terms and recalibrate them channel-wise."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        self.out = Conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, 3, 1, 1)
        high = x - low
        gates = self.selector(torch.cat((low, high.abs()), dim=1)).view(x.shape[0], 2, x.shape[1], 1, 1)
        gates = gates.softmax(dim=1)
        return x + self.out(low * gates[:, 0] + high * gates[:, 1])


class AdaptiveSparseAIFI(AIFI):
    """AIFI with a learned sparse spatial residual gate for semantically important tokens."""

    def __init__(self, c1: int, cm: int = 1024, num_heads: int = 8, keep_ratio: float = 0.5):
        super().__init__(c1, cm, num_heads)
        self.keep_ratio = keep_ratio
        self.score = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).flatten(1)
        keep = max(1, int(scores.shape[1] * self.keep_ratio))
        threshold = scores.topk(keep, dim=1).values[:, -1:]
        hard = (scores >= threshold).to(x.dtype)
        soft = scores.sigmoid()
        gate = (hard + soft - soft.detach()).view(x.shape[0], 1, x.shape[2], x.shape[3])
        return x + gate * (super().forward(x) - x)


class ParallelBackboneBlock(nn.Module):
    """Parallel local/global feature extractor used by the FUR-DETR proxy."""

    def __init__(self, c1: int, c2: int, stride: int = 1):
        super().__init__()
        self.local = nn.Sequential(Conv(c1, c2, 3, stride), DWConv(c2, c2, 3))
        self.global_branch = nn.Sequential(Conv(c1, c2, 1, stride), DWConv(c2, c2, 7))
        self.fuse = Conv(c2 * 2, c2, 1)
        self.shortcut = Conv(c1, c2, 1, stride, act=False) if stride != 1 or c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.fuse(torch.cat((self.local(x), self.global_branch(x)), dim=1))


class WaveletFeatureRefinement(nn.Module):
    """Haar-inspired low/high-frequency refinement without changing feature resolution."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels * 2, 1)
        )
        self.out = Conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_small = F.avg_pool2d(x, 2, 2)
        low = F.interpolate(low_small, size=x.shape[-2:], mode="nearest")
        high = x - low
        gates = self.gate(torch.cat((low, high.abs()), dim=1)).view(x.shape[0], 2, x.shape[1], 1, 1).softmax(dim=1)
        return x + self.out(low * gates[:, 0] + high * gates[:, 1])


class CoordAtt(nn.Module):
    """Coordinate attention with separate horizontal and vertical channel gates."""

    def __init__(self, channels: int, reduction: int = 32):
        """Initialize shared coordinate embedding and directional projections."""
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode long-range position along each axis and apply directional gates."""
        h, w = x.shape[2:]
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.act(self.bn1(self.conv1(torch.cat((x_h, x_w), dim=2))))
        encoded_h, encoded_w = torch.split(encoded, [h, w], dim=2)
        gate_h = self.conv_h(encoded_h).sigmoid()
        gate_w = self.conv_w(encoded_w).transpose(2, 3).sigmoid()
        return x * gate_h * gate_w


class ECA(nn.Module):
    """Efficient channel attention using local cross-channel interaction."""

    def __init__(self, channels: int, kernel_size: int = 3):
        """Initialize global pooling and a lightweight 1D channel interaction."""
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight channels while preserving the input spatial resolution."""
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.act(self.conv(weights)).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ECASpatial(ECA):
    """Efficient channel attention followed by a lightweight spatial gate."""

    def __init__(self, channels: int, kernel_size: int = 3, spatial_kernel: int = 7):
        """Initialize ECA and a two-map spatial projection."""
        super().__init__(channels, kernel_size)
        if spatial_kernel not in (3, 7):
            raise ValueError("spatial_kernel must be 3 or 7")
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel calibration and spatial foreground selection."""
        x = super().forward(x)
        descriptor = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * self.spatial(descriptor).sigmoid()


class EMA(nn.Module):
    """Efficient multi-scale attention with grouped spatial interactions."""

    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        self.groups = max(math.gcd(channels, factor), 1)
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).transpose(2, 3)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.transpose(2, 3).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, 1, -1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, 1, -1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class FastEMA(EMA):
    """Mathematically equivalent EMA using direct reductions and batched matrix multiplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EMA with lower-overhead tensor operations."""
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = group_x.mean(dim=3, keepdim=True)
        x_w = group_x.mean(dim=2, keepdim=True).transpose(2, 3)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.transpose(2, 3).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(x1.mean(dim=(2, 3)).unsqueeze(1))
        x12 = x2.flatten(2)
        x21 = self.softmax(x2.mean(dim=(2, 3)).unsqueeze(1))
        x22 = x1.flatten(2)
        weights = (torch.bmm(x11, x12) + torch.bmm(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class DCNv2(nn.Module):
    """Modulated deformable convolution v2 followed by batch norm and SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p=None, g: int = 1, act: bool = True):
        super().__init__()
        p = autopad(k, p)
        self.offset_mask = nn.Conv2d(c1, 3 * k * k, k, s, p)
        self.conv = DeformConv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_x, offset_y, mask = torch.chunk(self.offset_mask(x), 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        return self.act(self.bn(self.conv(x, offset, mask.sigmoid())))

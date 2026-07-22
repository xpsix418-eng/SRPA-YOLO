# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Partial_conv3, Proto, Residual, SwiGLUFFN
from .conv import AsymRepConv, ContrastRepConv, Conv, DiverseRepConv, DWConv, RepConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init




__all__ = (
    "OBB",
    "AuxDirectDetect",
    "Classify",
    "BoxSpatialDetect",
    "Detect",
    "DetectDGQP",
    "DetectLogitDGQP",
    "DirectDetect",
    "HybridLiteDecoupledDetect",
    "LiteDecoupledDetect",
    "LiteSharedDetect",
    "P2LiteDecoupledDetect",
    "P3LiteDecoupledDetect",
    "P3P4LiteDecoupledDetect",
    "P4LiteDecoupledDetect",
    "Pose",
    "RepBoxDetect",
    "RepSharedClsAdapterDetect",
    "RepSharedChannelScaleDetect",
    "RepSharedDetect",
    "RepSharedTaskResidualDetect",
    "RepSharedP2TaskResidualDetect",
    "RepSharedP2DualTaskResidualDetect",
    "RepSharedP2GlobalSemanticClsDetect",
    "RepSharedP2GlobalScoreGateDetect",
    "RepSharedP2SemanticClsDetect",
    "RepSharedP5LiteDetect",
    "RepSharedMergedDetect",
    "RepSharedMergedBoxAdapterDetect",
    "RepSharedMergedDGQPDetect",
    "RepSharedMergedP3P4DGQPDetect",
    "RepSharedDetachClsDetect",
    "RepSharedDGQPDetect",
    "RepSharedGatedClsDetect",
    "RepSharedIsolatedGatedClsDetect",
    "RepSharedLeadAuxDetect",
    "RepSharedLeadPyramidAuxDetect",
    "RTDETRDecoder",
    "Segment",
    "SharedDetect",
    "SlimDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: list[torch.Tensor]) -> dict | tuple:
        """Perform forward pass of the v10Detect module.

        Args:
            x (list[torch.Tensor]): Input feature maps from different levels.

        Returns:
            outputs (dict | tuple): Training mode returns dict with one2many and one2one outputs. Inference mode returns
                processed detections or tuple with detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class DetectDGQP(Detect):
    """Detect head with DFL-distribution-guided localization quality ranking."""

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Add a five-parameter GFLv2-style quality predictor to the standard Detect head."""
        super().__init__(nc, ch)
        self.reg_conf = nn.Sequential(nn.Conv2d(4, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.reg_conf[0].weight)
        nn.init.constant_(self.reg_conf[0].bias, math.log(9.0))

    def _localization_quality(self, regression: torch.Tensor) -> torch.Tensor:
        """Estimate box quality from the four edge-distribution peak probabilities."""
        batch, _, height, width = regression.shape
        distribution = regression.view(batch, 4, self.reg_max, height, width).softmax(2)
        return self.reg_conf(distribution.amax(dim=2))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Fuse classification confidence with DFL-derived localization quality."""
        if self.end2end:
            return super().forward(x)
        for i in range(self.nl):
            regression = self.cv2[i](x[i])
            classification = self.cv3[i](x[i])
            if self.training:
                quality = self._localization_quality(regression)
                joint_probability = classification.sigmoid() * quality
                classification = torch.logit(joint_probability.clamp(1e-4, 1.0 - 1e-4))
            x[i] = torch.cat((regression, classification), 1)
        if self.training:
            return x
        y = self._inference_with_quality(x)
        return y if self.export else (y, x)

    def _inference_with_quality(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode DFL boxes once and multiply class scores by predicted localization quality."""
        shape = x[0].shape
        x_cat = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (anchor.transpose(0, 1) for anchor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, classification = x_cat.split((self.reg_max * 4, self.nc), 1)
        distribution = box.view(shape[0], 4, self.reg_max, -1).transpose(2, 1).softmax(1)
        distances = self.dfl.conv(distribution).view(shape[0], 4, -1)
        quality = self.reg_conf(distribution.amax(dim=1).unsqueeze(2)).squeeze(2)
        boxes = self.decode_bboxes(distances, self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((boxes, classification.sigmoid() * quality), 1)


class DetectLogitDGQP(Detect):
    """Detect head with low-cost DFL-logit-margin localization quality ranking."""

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Add a quality predictor over top-1/top-2 DFL logit margins."""
        super().__init__(nc, ch)
        self.reg_conf = nn.Sequential(nn.Conv2d(4, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.reg_conf[0].weight)
        nn.init.constant_(self.reg_conf[0].bias, math.log(9.0))

    def _localization_quality(self, regression: torch.Tensor) -> torch.Tensor:
        """Estimate quality from per-edge logit margins without an extra softmax."""
        batch, _, height, width = regression.shape
        edge_logits = regression.view(batch, 4, self.reg_max, height, width)
        top2 = edge_logits.topk(2, dim=2).values
        return self.reg_conf(top2[:, :, 0] - top2[:, :, 1])

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Fuse classification confidence with margin-derived localization quality."""
        if self.end2end:
            return super().forward(x)
        for i in range(self.nl):
            regression = self.cv2[i](x[i])
            classification = self.cv3[i](x[i])
            if self.training:
                quality = self._localization_quality(regression)
                joint_probability = classification.sigmoid() * quality
                classification = torch.logit(joint_probability.clamp(1e-4, 1.0 - 1e-4))
            x[i] = torch.cat((regression, classification), 1)
        if self.training:
            return x
        y = self._inference_with_quality(x)
        return y if self.export else (y, x)

    def _inference_with_quality(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode boxes normally and apply a cheap margin-derived quality factor."""
        shape = x[0].shape
        x_cat = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (anchor.transpose(0, 1) for anchor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, classification = x_cat.split((self.reg_max * 4, self.nc), 1)
        boxes = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        edge_logits = box.view(shape[0], 4, self.reg_max, -1)
        top2 = edge_logits.topk(2, dim=2).values
        quality = self.reg_conf((top2[:, :, 0] - top2[:, :, 1]).unsqueeze(2)).squeeze(2)
        return torch.cat((boxes, classification.sigmoid() * quality), 1)


class SlimDetect(Detect):
    """Latency-oriented Detect head with one standard 3x3 block per branch and scale."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize a shallower decoupled head while preserving DFL and Detect outputs."""
        super().__init__(nc, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)


class AuxDirectDetect(Detect):
    """Direct deployment head with a standard convolutional auxiliary head used only during training."""

    legacy = True
    auxiliary_train = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Keep rich Detect branches as auxiliary supervision and install direct deployment predictors."""
        super().__init__(nc, ch)
        self.aux_cv2 = self.cv2
        self.aux_cv3 = self.cv3
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | dict | tuple:
        """Return both heads during training and only the direct deployment head during inference."""
        main = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        if self.training:
            if not self.stride.any():
                return main
            auxiliary = [
                torch.cat((self.aux_cv2[i](x[i]), self.aux_cv3[i](x[i])), 1) for i in range(self.nl)
            ]
            return {"main": main, "auxiliary": auxiliary}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize direct and auxiliary box/classification predictors."""
        for box, cls, aux_box, aux_cls, stride in zip(
            self.cv2, self.cv3, self.aux_cv2, self.aux_cv3, self.stride
        ):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
            aux_box[-1].bias.data[:] = 1.0
            aux_cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def fuse(self):
        """Remove training-only auxiliary predictors from the deployment module."""
        self.aux_cv2 = nn.ModuleList([nn.Identity()] * self.nl)
        self.aux_cv3 = nn.ModuleList([nn.Identity()] * self.nl)


class BoxSpatialDetect(Detect):
    """Detect head with a narrow spatial regression branch and a direct classification branch."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Add full 3x3 channel mixing only where localization needs spatial context."""
        super().__init__(nc, ch)
        c2 = max(16, min(32, ch[0]))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

    def bias_init(self):
        """Initialize regression and direct classification predictor biases."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class DirectDetect(Detect):
    """Minimal decoupled Detect head with direct 1x1 box and class predictors."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize direct predictors while retaining DFL decoding and all detection scales."""
        super().__init__(nc, ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

    def bias_init(self):
        """Initialize direct box and class predictor biases."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class LiteDecoupledDetect(Detect):
    """Efficient Detect head with independent depthwise-separable box and class branches."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Restore task-specific spatial features while retaining lightweight branch transforms."""
        super().__init__(nc, ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )


class HybridLiteDecoupledDetect(Detect):
    """Task-specific lightweight branches on fine scales with a shared coarse-scale head."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Decouple spatial operators where small objects dominate and share the coarse-scale transform."""
        super().__init__(nc, ch)
        if len(ch) != 3:
            raise ValueError(f"HybridLiteDecoupledDetect expects three scales, received {len(ch)}")
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.fine_box = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:2]
        )
        self.fine_cls = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch[:2]
        )
        coarse_width = 48
        self.coarse_stem = RepConv(ch[2], coarse_width, 3, 1, bn=(ch[2] == coarse_width))
        self.coarse_pred = nn.Conv2d(coarse_width, self.no, 1)
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Predict with independent fine-scale spatial transforms and one shared coarse transform."""
        for i in range(2):
            x[i] = torch.cat((self.fine_box[i](x[i]), self.fine_cls[i](x[i])), 1)
        x[2] = self.coarse_pred(self.coarse_stem(x[2]))
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize output biases for fine decoupled and coarse merged predictions."""
        for box, cls, stride in zip(self.fine_box, self.fine_cls, self.stride[:2]):
            box[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
        self.coarse_pred.bias.data[: 4 * self.reg_max] = 1.0
        self.coarse_pred.bias.data[4 * self.reg_max :] = math.log(
            5 / self.nc / (640 / self.stride[2]) ** 2
        )


class P2LiteDecoupledDetect(Detect):
    """Nonlinear task decoupling at the finest scale with shared merged coarse heads."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Reserve independent spatial operators for the UAV-dominant finest detection scale."""
        super().__init__(nc, ch)
        if len(ch) != 3:
            raise ValueError(f"P2LiteDecoupledDetect expects three scales, received {len(ch)}")
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.fine_box = nn.Sequential(DWConv(ch[0], ch[0], 3), Conv(ch[0], c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1))
        self.fine_cls = nn.Sequential(DWConv(ch[0], ch[0], 3), Conv(ch[0], c3, 1), nn.Conv2d(c3, self.nc, 1))
        coarse_width = 48
        self.coarse_stem = nn.ModuleList(
            RepConv(x, coarse_width, 3, 1, bn=(x == coarse_width)) for x in ch[1:]
        )
        self.coarse_pred = nn.ModuleList(nn.Conv2d(coarse_width, self.no, 1) for _ in ch[1:])
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use independent finest-scale tasks and shared transforms on the two coarse scales."""
        x[0] = torch.cat((self.fine_box(x[0]), self.fine_cls(x[0])), 1)
        for i in range(1, self.nl):
            x[i] = self.coarse_pred[i - 1](self.coarse_stem[i - 1](x[i]))
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize fine task-specific and coarse merged output biases."""
        self.fine_box[-1].bias.data[:] = 1.0
        self.fine_cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[0]) ** 2)
        for predictor, stride in zip(self.coarse_pred, self.stride[1:]):
            predictor.bias.data[: 4 * self.reg_max] = 1.0
            predictor.bias.data[4 * self.reg_max :] = math.log(5 / self.nc / (640 / stride) ** 2)


class P3LiteDecoupledDetect(Detect):
    """Nonlinear task decoupling on P3 with shared merged P2/P4 heads."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Reserve independent spatial operators for the small-object middle fine scale."""
        super().__init__(nc, ch)
        if len(ch) != 3:
            raise ValueError(f"P3LiteDecoupledDetect expects three scales, received {len(ch)}")
        c2 = max((16, ch[1] // 4, self.reg_max * 4))
        c3 = max(ch[1], min(self.nc, 100))
        coarse_width = 48
        self.shared_stem = nn.ModuleList(
            RepConv(x, coarse_width, 3, 1, bn=(x == coarse_width)) if i != 1 else nn.Identity()
            for i, x in enumerate(ch)
        )
        self.shared_pred = nn.ModuleList(
            nn.Conv2d(coarse_width, self.no, 1) if i != 1 else nn.Identity() for i in range(3)
        )
        self.p3_box = nn.Sequential(DWConv(ch[1], ch[1], 3), Conv(ch[1], c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1))
        self.p3_cls = nn.Sequential(DWConv(ch[1], ch[1], 3), Conv(ch[1], c3, 1), nn.Conv2d(c3, self.nc, 1))
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use independent P3 tasks and shared transforms on P2/P4."""
        for i in range(self.nl):
            if i == 1:
                x[i] = torch.cat((self.p3_box(x[i]), self.p3_cls(x[i])), 1)
            else:
                x[i] = self.shared_pred[i](self.shared_stem[i](x[i]))
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize P3 task-specific and shared-scale output biases."""
        for i, stride in enumerate(self.stride):
            if i == 1:
                self.p3_box[-1].bias.data[:] = 1.0
                self.p3_cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
            else:
                self.shared_pred[i].bias.data[: 4 * self.reg_max] = 1.0
                self.shared_pred[i].bias.data[4 * self.reg_max :] = math.log(5 / self.nc / (640 / stride) ** 2)


class P3P4LiteDecoupledDetect(Detect):
    """Task-specific lightweight branches on P3/P4 with the original shared P2 head."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Decouple mid/coarse spatial operators while keeping the dense P2 path shared."""
        super().__init__(nc, ch)
        if len(ch) != 3:
            raise ValueError(f"P3P4LiteDecoupledDetect expects three scales, received {len(ch)}")
        coarse_width = 48
        self.p2_stem = RepConv(ch[0], coarse_width, 3, 1, bn=(ch[0] == coarse_width))
        self.p2_pred = nn.Conv2d(coarse_width, self.no, 1)
        self.task_box = nn.ModuleList()
        self.task_cls = nn.ModuleList()
        for channels in ch[1:]:
            c2 = max((16, channels // 4, self.reg_max * 4))
            c3 = max(channels, min(self.nc, 100))
            self.task_box.append(
                nn.Sequential(DWConv(channels, channels, 3), Conv(channels, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1))
            )
            self.task_cls.append(
                nn.Sequential(DWConv(channels, channels, 3), Conv(channels, c3, 1), nn.Conv2d(c3, self.nc, 1))
            )
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Keep P2 merged and use independent task transforms on P3/P4."""
        x[0] = self.p2_pred(self.p2_stem(x[0]))
        for i in range(1, self.nl):
            x[i] = torch.cat((self.task_box[i - 1](x[i]), self.task_cls[i - 1](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize shared P2 and decoupled P3/P4 output biases."""
        self.p2_pred.bias.data[: 4 * self.reg_max] = 1.0
        self.p2_pred.bias.data[4 * self.reg_max :] = math.log(5 / self.nc / (640 / self.stride[0]) ** 2)
        for box, cls, stride in zip(self.task_box, self.task_cls, self.stride[1:]):
            box[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class P4LiteDecoupledDetect(Detect):
    """Task-specific lightweight branch only on P4 with shared P2/P3 heads."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Reserve nonlinear task-specific spatial capacity for the coarse 40x40 scale."""
        super().__init__(nc, ch)
        if len(ch) != 3:
            raise ValueError(f"P4LiteDecoupledDetect expects three scales, received {len(ch)}")
        width = 48
        self.shared_stem = nn.ModuleList(
            RepConv(x, width, 3, 1, bn=(x == width)) if i != 2 else nn.Identity()
            for i, x in enumerate(ch)
        )
        self.shared_pred = nn.ModuleList(
            nn.Conv2d(width, self.no, 1) if i != 2 else nn.Identity() for i in range(3)
        )
        c2 = max((16, ch[2] // 4, self.reg_max * 4))
        c3 = max(ch[2], min(self.nc, 100))
        self.p4_box = nn.Sequential(DWConv(ch[2], ch[2], 3), Conv(ch[2], c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1))
        self.p4_cls = nn.Sequential(DWConv(ch[2], ch[2], 3), Conv(ch[2], c3, 1), nn.Conv2d(c3, self.nc, 1))
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use shared transforms on P2/P3 and task-specific transforms on P4."""
        for i in range(self.nl):
            if i == 2:
                x[i] = torch.cat((self.p4_box(x[i]), self.p4_cls(x[i])), 1)
            else:
                x[i] = self.shared_pred[i](self.shared_stem[i](x[i]))
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize shared P2/P3 and decoupled P4 output biases."""
        for i, stride in enumerate(self.stride):
            if i == 2:
                self.p4_box[-1].bias.data[:] = 1.0
                self.p4_cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
            else:
                self.shared_pred[i].bias.data[: 4 * self.reg_max] = 1.0
                self.shared_pred[i].bias.data[4 * self.reg_max :] = math.log(5 / self.nc / (640 / stride) ** 2)


class LiteSharedDetect(Detect):
    """Efficient Detect head with a depthwise-separable spatial stem shared by box and class outputs."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Add inexpensive 3x3 spatial modeling before separate 1x1 box and class predictors."""
        super().__init__(nc, ch)
        c = max((16, ch[0] // 4, self.reg_max * 4))
        self.stem = nn.ModuleList(nn.Sequential(DWConv(x, x, 3), Conv(x, c, 1)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply the shared spatial stem once per scale before predicting box and class logits."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each feature-map stride."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class SharedDetect(Detect):
    """Latency-oriented Detect head sharing one 3x3 feature transform between box and class outputs."""

    legacy = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize one shared feature block and separate 1x1 predictors per scale."""
        super().__init__(nc, ch)
        c = max((16, ch[0] // 4, self.reg_max * 4))
        self.stem = nn.ModuleList(Conv(x, c, 3) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply each shared stem once before producing regression and classification logits."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each feature-map stride."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class RepSharedDetect(Detect):
    """Narrow shared Detect head with a training-time reparameterizable spatial stem."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Use a configurable-width RepConv stem shared by box and class predictors."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(RepConv(x, c, 3, 1, bn=(x == c)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply the reparameterizable stem once per scale before task-specific output projections."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each feature-map stride."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class RepSharedSplitTaskDetect(Detect):
    """Single-stem head that reserves a private classification subspace within a fixed channel budget."""

    legacy = True

    def __init__(
        self,
        nc: int = 80,
        shared_width: int | tuple = 48,
        private_width: int = 16,
        ch: tuple = (),
    ):
        """Use shared channels for both tasks and private channels only for classification."""
        if isinstance(shared_width, (list, tuple)):
            ch, shared, private = tuple(shared_width), 48, 16
        else:
            shared, private = int(shared_width), int(private_width)
        super().__init__(nc, ch)
        total = shared + private
        self.shared_width = shared
        self.private_width = private
        # Keep the V92 two-branch RepConv topology so the first 48 channels can be copied exactly.
        self.stem = nn.ModuleList(RepConv(x, total, 3, 1, bn=False) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(total, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(total, self.nc, 1) for _ in ch)
        self.merged_cv = nn.ModuleList()
        for box in self.cv2:
            box.weight.register_hook(self._mask_private_box_gradient)

    def _mask_private_box_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Keep private classification columns structurally absent from the regression projection."""
        gradient = gradient.clone()
        gradient[:, self.shared_width :] = 0
        return gradient

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Keep regression in the shared basis while classification can use the full basis."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = self.merged_cv[i](feature) if len(self.merged_cv) else torch.cat(
                (self.cv2[i](feature), self.cv3[i](feature)), 1
            )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each feature-map stride."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def fuse(self):
        """Merge box and class projections into one exactly equivalent deployment convolution."""
        if len(self.merged_cv):
            return
        merged = nn.ModuleList()
        with torch.no_grad():
            for box, cls in zip(self.cv2, self.cv3):
                projection = nn.Conv2d(
                    box.in_channels,
                    box.out_channels + cls.out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ).to(device=box.weight.device, dtype=box.weight.dtype)
                projection.weight.copy_(torch.cat((box.weight, cls.weight), dim=0))
                projection.bias.copy_(torch.cat((box.bias, cls.bias), dim=0))
                merged.append(projection)
        self.merged_cv = merged
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()


class RepSharedPrivateResidualDetect(Detect):
    """Frozen V92-compatible base plus a class residual that folds into one deployment stem."""

    legacy = True
    private_residual_training = True

    def __init__(self, nc: int = 80, shared_width: int | tuple = 48, private_width: int = 16, ch: tuple = ()):
        """Keep base and private parameters separate during training for exact constrained optimization."""
        if isinstance(shared_width, (list, tuple)):
            ch, shared, private = tuple(shared_width), 48, 16
        else:
            shared, private = int(shared_width), int(private_width)
        super().__init__(nc, ch)
        self.shared_width, self.private_width = shared, private
        self.shared_stem = nn.ModuleList(RepConv(x, shared, 3, 1, bn=(x == shared)) for x in ch)
        self.private_stem = nn.ModuleList(RepConv(x, private, 3, 1, bn=(x == private)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(shared, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList()
        self.cv3_shared = nn.ModuleList(nn.Conv2d(shared, self.nc, 1) for _ in ch)
        self.cv3_private = nn.ModuleList(nn.Conv2d(private, self.nc, 1, bias=False) for _ in ch)
        for projection in self.cv3_private:
            nn.init.zeros_(projection.weight)
        self.deployment_stem = nn.ModuleList()
        self.merged_cv = nn.ModuleList()
        self.deployment_act = nn.SiLU()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Add the private class residual before fusion and use one path after fusion."""
        for i in range(self.nl):
            if len(self.deployment_stem):
                x[i] = self.merged_cv[i](self.deployment_act(self.deployment_stem[i](x[i])))
            else:
                shared = self.shared_stem[i](x[i])
                private = self.private_stem[i](x[i])
                x[i] = torch.cat(
                    (self.cv2[i](shared), self.cv3_shared[i](shared) + self.cv3_private[i](private)), 1
                )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize only the frozen base biases; the private residual has no bias."""
        for box, cls, stride in zip(self.cv2, self.cv3_shared, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def fuse(self):
        """Concatenate parallel training stems and projections into one deployment path."""
        if len(self.deployment_stem):
            return
        stems, projections = nn.ModuleList(), nn.ModuleList()
        with torch.no_grad():
            for shared, private, box, cls_s, cls_p in zip(
                self.shared_stem, self.private_stem, self.cv2, self.cv3_shared, self.cv3_private
            ):
                if hasattr(shared, "conv1"):
                    shared.fuse_convs()
                if hasattr(private, "conv1"):
                    private.fuse_convs()
                stem = nn.Conv2d(shared.conv.in_channels, self.shared_width + self.private_width, 3, 1, 1, bias=True).to(
                    device=shared.conv.weight.device, dtype=shared.conv.weight.dtype
                )
                stem.weight.copy_(torch.cat((shared.conv.weight, private.conv.weight), 0))
                stem.bias.copy_(torch.cat((shared.conv.bias, private.conv.bias), 0))
                stems.append(stem)
                projection = nn.Conv2d(
                    self.shared_width + self.private_width, box.out_channels + self.nc, 1, bias=True
                ).to(device=box.weight.device, dtype=box.weight.dtype)
                box_weight = F.pad(box.weight, (0, 0, 0, 0, 0, self.private_width))
                cls_weight = torch.cat((cls_s.weight, cls_p.weight), 1)
                projection.weight.copy_(torch.cat((box_weight, cls_weight), 0))
                projection.bias.copy_(torch.cat((box.bias, cls_s.bias), 0))
                projections.append(projection)
        self.deployment_stem, self.merged_cv = stems, projections
        self.shared_stem = nn.ModuleList()
        self.private_stem = nn.ModuleList()
        self.cv2 = nn.ModuleList()
        self.cv3_shared = nn.ModuleList()
        self.cv3_private = nn.ModuleList()


class RepSharedDiversePrivateDetect(RepSharedPrivateResidualDetect):
    """V101 head with a foldable low-frequency branch in each private classification stem."""

    private_residual_training = False
    diverse_private_training = True

    def __init__(self, nc: int = 80, shared_width: int | tuple = 48, private_width: int = 16, ch: tuple = ()):
        """Replace only the private stems while preserving V101-compatible base parameter names."""
        super().__init__(nc, shared_width, private_width, ch)
        self.private_stem = nn.ModuleList(
            DiverseRepConv(x, self.private_width, 3, 1, bn=(x == self.private_width)) for x in ch
        )


class RepSharedContrastPrivateDetect(RepSharedPrivateResidualDetect):
    """V101 head with a foldable center-surround branch in each private classification stem."""

    private_residual_training = False
    contrast_private_training = True

    def __init__(self, nc: int = 80, shared_width: int | tuple = 48, private_width: int = 16, ch: tuple = ()):
        """Replace only the private stems while preserving V101-compatible base parameter names."""
        super().__init__(nc, shared_width, private_width, ch)
        self.private_stem = nn.ModuleList(
            ContrastRepConv(x, self.private_width, 3, 1, bn=(x == self.private_width)) for x in ch
        )


class RepSharedQualityResidualDetect(RepSharedPrivateResidualDetect):
    """Frozen object-presence logits plus a foldable localization-quality correction."""

    private_residual_training = False
    quality_residual_training = True

    def __init__(self, nc: int = 80, shared_width: int | tuple = 48, private_width: int = 16, ch: tuple = ()):
        """Add a zero-initialized quality projection on the existing private features."""
        super().__init__(nc, shared_width, private_width, ch)
        self.cv3_quality = nn.ModuleList(nn.Conv2d(self.private_width, self.nc, 1, bias=False) for _ in ch)
        for projection in self.cv3_quality:
            nn.init.zeros_(projection.weight)
        self.cache_quality_logits = False
        self.base_cls_maps: list[torch.Tensor] = []
        self.quality_residual_maps: list[torch.Tensor] = []

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Predict a quality correction that is separately supervised and exactly foldable."""
        if self.cache_quality_logits:
            self.base_cls_maps = []
            self.quality_residual_maps = []
        for i in range(self.nl):
            if len(self.deployment_stem):
                x[i] = self.merged_cv[i](self.deployment_act(self.deployment_stem[i](x[i])))
            else:
                shared = self.shared_stem[i](x[i])
                private = self.private_stem[i](x[i])
                base_cls = self.cv3_shared[i](shared) + self.cv3_private[i](private)
                quality = self.cv3_quality[i](private)
                if self.cache_quality_logits:
                    self.base_cls_maps.append(base_cls)
                    self.quality_residual_maps.append(quality)
                quality_for_output = quality.detach() if self.training else quality
                x[i] = torch.cat((self.cv2[i](shared), base_cls + quality_for_output), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Fold the quality correction into the private class projection."""
        if len(self.deployment_stem):
            return
        with torch.no_grad():
            for private, quality in zip(self.cv3_private, self.cv3_quality):
                private.weight.add_(quality.weight)
        self.cv3_quality = nn.ModuleList()
        super().fuse()


class RepSharedTaskResidualDetect(RepSharedDetect):
    """Shared head with one low-cost grouped spatial residual for both tasks."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add zero-initialized box/class residuals in a single grouped convolution."""
        super().__init__(nc, width_or_ch, ch)
        channels = self.cv2[0].in_channels
        self.task_residual = nn.ModuleList(
            Conv(channels, 2 * channels, 3, 1, g=channels, act=False) for _ in range(self.nl)
        )
        for residual in self.task_residual:
            nn.init.zeros_(residual.bn.weight)
            nn.init.zeros_(residual.bn.bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Refine shared features with separate box and class spatial residuals."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            box_delta, cls_delta = self.task_residual[i](feature).chunk(2, 1)
            box = self.cv2[i](feature + box_delta)
            cls = self.cv3[i](feature + cls_delta)
            x[i] = torch.cat((box, cls), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class RepSharedP2TaskResidualDetect(RepSharedDetect):
    """Task-specific spatial residual on the finest scale with merged coarse projections."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Specialize the finest scale and reserve merged projections for coarser scales."""
        super().__init__(nc, width_or_ch, ch)
        channels = self.cv2[0].in_channels
        self.task_residual = Conv(channels, 2 * channels, 3, 1, g=channels, act=False)
        nn.init.zeros_(self.task_residual.bn.weight)
        nn.init.zeros_(self.task_residual.bn.bias)
        self.merged_coarse = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply task specialization only at 160x160 and merge coarse output projections."""
        finest = self.stem[0](x[0])
        box_delta, cls_delta = self.task_residual(finest).chunk(2, 1)
        x[0] = torch.cat((self.cv2[0](finest + box_delta), self.cv3[0](finest + cls_delta)), 1)
        for i in range(1, self.nl):
            feature = self.stem[i](x[i])
            x[i] = self.merged_coarse[i - 1](feature) if len(self.merged_coarse) else torch.cat(
                (self.cv2[i](feature), self.cv3[i](feature)), 1
            )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Merge box/class projections on coarse scales while preserving finest task residuals."""
        if len(self.merged_coarse):
            return
        merged = nn.ModuleList()
        with torch.no_grad():
            for box, cls in zip(self.cv2[1:], self.cv3[1:]):
                projection = nn.Conv2d(
                    box.in_channels,
                    box.out_channels + cls.out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ).to(device=box.weight.device, dtype=box.weight.dtype)
                projection.weight.copy_(torch.cat((box.weight, cls.weight), dim=0))
                projection.bias.copy_(torch.cat((box.bias, cls.bias), dim=0))
                merged.append(projection)
        self.merged_coarse = merged
        self.cv2 = nn.ModuleList([self.cv2[0]])
        self.cv3 = nn.ModuleList([self.cv3[0]])


class RepSharedP2DualTaskResidualDetect(RepSharedDetect):
    """Hardware-friendly dual depthwise task residuals on the finest scale."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Use independently optimized depthwise kernels for box and class residuals."""
        super().__init__(nc, width_or_ch, ch)
        channels = self.cv2[0].in_channels
        self.box_residual = Conv(channels, channels, 3, 1, g=channels, act=False)
        self.cls_residual = Conv(channels, channels, 3, 1, g=channels, act=False)
        for residual in (self.box_residual, self.cls_residual):
            nn.init.zeros_(residual.bn.weight)
            nn.init.zeros_(residual.bn.bias)
        self.merged_coarse = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Specialize finest-scale tasks and use merged projections at coarser scales."""
        finest = self.stem[0](x[0])
        box = self.cv2[0](finest + self.box_residual(finest))
        cls = self.cv3[0](finest + self.cls_residual(finest))
        x[0] = torch.cat((box, cls), 1)
        for i in range(1, self.nl):
            feature = self.stem[i](x[i])
            x[i] = self.merged_coarse[i - 1](feature) if len(self.merged_coarse) else torch.cat(
                (self.cv2[i](feature), self.cv3[i](feature)), 1
            )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Merge coarse projections while retaining the two finest-scale residual paths."""
        if len(self.merged_coarse):
            return
        merged = nn.ModuleList()
        with torch.no_grad():
            for box, cls in zip(self.cv2[1:], self.cv3[1:]):
                projection = nn.Conv2d(
                    box.in_channels,
                    box.out_channels + cls.out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ).to(device=box.weight.device, dtype=box.weight.dtype)
                projection.weight.copy_(torch.cat((box.weight, cls.weight), dim=0))
                projection.bias.copy_(torch.cat((box.bias, cls.bias), dim=0))
                merged.append(projection)
        self.merged_coarse = merged
        self.cv2 = nn.ModuleList([self.cv2[0]])
        self.cv3 = nn.ModuleList([self.cv3[0]])


class RepSharedP2SemanticClsDetect(RepSharedDetect):
    """Use P5 semantic context only in the finest-scale classification branch."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build three detection scales plus one coarse semantic context input."""
        if isinstance(width_or_ch, (list, tuple)):
            all_channels, width = tuple(width_or_ch), 48
        else:
            all_channels, width = tuple(ch), int(width_or_ch)
        if len(all_channels) != 4:
            raise ValueError(f"RepSharedP2SemanticClsDetect expects three detection inputs plus P5, got {all_channels}")
        super().__init__(nc, width, all_channels[:3])
        self.semantic = Conv(all_channels[3], width, 1, act=False)
        nn.init.zeros_(self.semantic.bn.weight)
        nn.init.zeros_(self.semantic.bn.bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Preserve all box features while adding coarse context only to P2 classification."""
        context = x[3]
        predictions = x[:3]
        for i in range(self.nl):
            feature = self.stem[i](predictions[i])
            cls_feature = feature
            if i == 0:
                semantic = F.interpolate(self.semantic(context), size=feature.shape[2:], mode="nearest")
                cls_feature = feature + semantic
            predictions[i] = torch.cat((self.cv2[i](feature), self.cv3[i](cls_feature)), 1)
        if self.training:
            return predictions
        y = self._inference(predictions)
        return y if self.export else (y, predictions)


class RepSharedP2GlobalSemanticClsDetect(RepSharedDetect):
    """Use global P5 semantics to channel-calibrate only the P2 classification path."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build a zero-initialized global semantic gate for the finest classification feature."""
        if isinstance(width_or_ch, (list, tuple)):
            all_channels, width = tuple(width_or_ch), 48
        else:
            all_channels, width = tuple(ch), int(width_or_ch)
        if len(all_channels) != 4:
            raise ValueError(
                f"RepSharedP2GlobalSemanticClsDetect expects three detection inputs plus P5, got {all_channels}"
            )
        super().__init__(nc, width, all_channels[:3])
        self.semantic_pool = nn.AdaptiveAvgPool2d(1)
        self.semantic_gate = nn.Conv2d(all_channels[3], width, 1)
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)

    def _semantic_gate(self, p5: torch.Tensor) -> torch.Tensor:
        """Return an identity-initialized per-image channel gate in [0, 2]."""
        return 2.0 * self.semantic_gate(self.semantic_pool(p5)).sigmoid()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Preserve box features while applying image-level semantic calibration to P2 class logits."""
        gate = self._semantic_gate(x[3])
        predictions = x[:3]
        feature = self.stem[0](predictions[0])
        predictions[0] = torch.cat((self.cv2[0](feature), self.cv3[0](feature * gate)), 1)
        for i in range(1, self.nl):
            feature = self.stem[i](predictions[i])
            predictions[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return predictions
        y = self._inference(predictions)
        return y if self.export else (y, predictions)


class RepSharedP2GlobalScoreGateDetect(RepSharedDetect):
    """Use global P5 semantics as a scalar prior on P2 classification logits."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build a zero-initialized image-level score gate for the finest classification logits."""
        if isinstance(width_or_ch, (list, tuple)):
            all_channels, width = tuple(width_or_ch), 48
        else:
            all_channels, width = tuple(ch), int(width_or_ch)
        if len(all_channels) != 4:
            raise ValueError(
                f"RepSharedP2GlobalScoreGateDetect expects three detection inputs plus P5, got {all_channels}"
            )
        super().__init__(nc, width, all_channels[:3])
        self.semantic_pool = nn.AdaptiveAvgPool2d(1)
        self.score_gate = nn.Conv2d(all_channels[3], self.nc, 1)
        nn.init.zeros_(self.score_gate.weight)
        nn.init.zeros_(self.score_gate.bias)

    def _score_prior(self, p5: torch.Tensor) -> torch.Tensor:
        """Return an identity-initialized additive log prior for classification logits."""
        gate = 2.0 * self.score_gate(self.semantic_pool(p5)).sigmoid()
        return gate.clamp(1e-4, 2.0).log()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Add image-level P5 semantic prior to P2 classification logits only."""
        prior = self._score_prior(x[3])
        predictions = x[:3]
        feature = self.stem[0](predictions[0])
        predictions[0] = torch.cat((self.cv2[0](feature), self.cv3[0](feature) + prior), 1)
        for i in range(1, self.nl):
            feature = self.stem[i](predictions[i])
            predictions[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return predictions
        y = self._inference(predictions)
        return y if self.export else (y, predictions)


class RepSharedP5LiteDetect(RepSharedDetect):
    """Use V42 RepShared processing on P2-P4 and direct lightweight prediction on native P5."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build three RepShared scales plus a direct 1x1 P5 branch."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
            ch = tuple(ch)
        if len(ch) != 4:
            raise ValueError(f"RepSharedP5LiteDetect expects four detection inputs, got {ch}")
        super().__init__(nc, c, ch)
        self.stem = nn.ModuleList(RepConv(x, c, 3, 1, bn=(x == c)) for x in ch[:3])
        self.cv2 = nn.ModuleList([nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch[:3]] + [nn.Conv2d(ch[3], 4 * self.reg_max, 1)])
        self.cv3 = nn.ModuleList([nn.Conv2d(c, self.nc, 1) for _ in ch[:3]] + [nn.Conv2d(ch[3], self.nc, 1)])

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply RepShared stems to fine scales while keeping P5 at its native semantic stride."""
        for i in range(self.nl):
            feature = self.stem[i](x[i]) if i < self.nl - 1 else x[i]
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class RepSharedMergedDetect(RepSharedDetect):
    """RepShared head with exactly merged box and class projections at deployment."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Keep independent training projections and reserve an empty fused projection list."""
        super().__init__(nc, width_or_ch, ch)
        self.merged_cv = nn.ModuleList()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use one output projection per scale after deployment fusion."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = self.merged_cv[i](feature) if len(self.merged_cv) else torch.cat(
                (self.cv2[i](feature), self.cv3[i](feature)), 1
            )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Concatenate box and class 1x1 kernels into one exactly equivalent projection."""
        if len(self.merged_cv):
            return
        merged = nn.ModuleList()
        with torch.no_grad():
            for box, cls in zip(self.cv2, self.cv3):
                projection = nn.Conv2d(
                    box.in_channels,
                    box.out_channels + cls.out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ).to(device=box.weight.device, dtype=box.weight.dtype)
                projection.weight.copy_(torch.cat((box.weight, cls.weight), dim=0))
                projection.bias.copy_(torch.cat((box.bias, cls.bias), dim=0))
                merged.append(projection)
        self.merged_cv = merged
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()


class RepSharedMergedBoxAdapterDetect(RepSharedMergedDetect):
    """RepShared head with removable task-specific linear regression adapters."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add identity-initialized box adapters that fold into the box projections."""
        super().__init__(nc, width_or_ch, ch)
        channels = self.cv2[0].in_channels
        self.box_adapter = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(self.nl))
        for adapter in self.box_adapter:
            nn.init.dirac_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use task-specific box features during training and merged projections after fusion."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            if len(self.merged_cv):
                x[i] = self.merged_cv[i](feature)
            else:
                box = self.cv2[i](self.box_adapter[i](feature))
                x[i] = torch.cat((box, self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Fold each linear box adapter into cv2 before exact box/class merging."""
        if len(self.merged_cv):
            return
        with torch.no_grad():
            for box, adapter in zip(self.cv2, self.box_adapter):
                box_weight = box.weight.flatten(1)
                adapter_weight = adapter.weight.flatten(1)
                adapter_bias = adapter.bias
                box.weight.copy_((box_weight @ adapter_weight).view_as(box.weight))
                box.bias.copy_(box.bias + box_weight @ adapter_bias)
        self.box_adapter = nn.ModuleList()
        super().fuse()


class RepSharedMergedDGQPDetect(RepSharedMergedDetect):
    """Merged RepShared head with distribution-guided localization quality ranking."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add the shared five-parameter GFLv2-style quality predictor."""
        super().__init__(nc, width_or_ch, ch)
        self.reg_conf = nn.Sequential(nn.Conv2d(4, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.reg_conf[0].weight)
        nn.init.constant_(self.reg_conf[0].bias, math.log(9.0))

    def _localization_quality(self, regression: torch.Tensor) -> torch.Tensor:
        """Estimate localization quality from the four DFL edge distributions."""
        batch, _, height, width = regression.shape
        distribution = regression.view(batch, 4, self.reg_max, height, width).softmax(2)
        return self.reg_conf(distribution.amax(dim=2))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Train joint confidence and use the merged projection after fusion."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            if len(self.merged_cv):
                regression, classification = self.merged_cv[i](feature).split((4 * self.reg_max, self.nc), 1)
            else:
                regression = self.cv2[i](feature)
                classification = self.cv3[i](feature)
            if self.training:
                joint_probability = classification.sigmoid() * self._localization_quality(regression)
                classification = torch.logit(joint_probability.clamp(1e-4, 1.0 - 1e-4))
            x[i] = torch.cat((regression, classification), 1)
        if self.training:
            return x
        y = self._inference_with_quality(x)
        return y if self.export else (y, x)

    def _inference_with_quality(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Reuse the DFL softmax to decode boxes and estimate ranking quality."""
        shape = x[0].shape
        x_cat = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (anchor.transpose(0, 1) for anchor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, classification = x_cat.split((self.reg_max * 4, self.nc), 1)
        distribution = box.view(shape[0], 4, self.reg_max, -1).transpose(2, 1).softmax(1)
        distances = self.dfl.conv(distribution).view(shape[0], 4, -1)
        quality = self.reg_conf(distribution.amax(dim=1).unsqueeze(2)).squeeze(2)
        boxes = self.decode_bboxes(distances, self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((boxes, classification.sigmoid() * quality), 1)


class RepSharedMergedP3P4DGQPDetect(RepSharedMergedDGQPDetect):
    """Apply DGQP only to P3/P4 while preserving the latency-critical P2 ranking."""

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Bypass quality estimation on the 80x80 P2 scale."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            if len(self.merged_cv):
                regression, classification = self.merged_cv[i](feature).split((4 * self.reg_max, self.nc), 1)
            else:
                regression = self.cv2[i](feature)
                classification = self.cv3[i](feature)
            if self.training and i > 0:
                joint_probability = classification.sigmoid() * self._localization_quality(regression)
                classification = torch.logit(joint_probability.clamp(1e-4, 1.0 - 1e-4))
            x[i] = torch.cat((regression, classification), 1)
        if self.training:
            return x
        y = self._inference_with_quality(x)
        return y if self.export else (y, x)

    def _inference_with_quality(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode all scales together and calibrate only P3/P4 score slices."""
        shape = x[0].shape
        p2_count = x[0].shape[-2] * x[0].shape[-1]
        x_cat = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (anchor.transpose(0, 1) for anchor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, classification = x_cat.split((self.reg_max * 4, self.nc), 1)
        distribution = box.view(shape[0], 4, self.reg_max, -1).transpose(2, 1).softmax(1)
        distances = self.dfl.conv(distribution).view(shape[0], 4, -1)
        edge_maxima = distribution.amax(dim=1)
        quality = self.reg_conf(edge_maxima[:, :, p2_count:].unsqueeze(2)).squeeze(2)
        scores = classification.sigmoid()
        scores[:, :, p2_count:] *= quality
        boxes = self.decode_bboxes(distances, self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((boxes, scores), 1)


class RepSharedDetachClsDetect(RepSharedDetect):
    """RepShared head that isolates shared localization features from classification gradients."""

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Detach class inputs only while training; inference remains identical to RepSharedDetect."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            cls_feature = feature.detach() if self.training else feature
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](cls_feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class RepSharedDGQPDetect(RepSharedDetect):
    """RepShared head with distribution-guided localization quality estimation."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add a shared lightweight quality predictor over DFL distribution statistics."""
        super().__init__(nc, width_or_ch, ch)
        self.reg_conf = nn.Sequential(
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.reg_conf[0].weight)
        nn.init.constant_(self.reg_conf[0].bias, math.log(9.0))

    def _localization_quality(self, regression: torch.Tensor) -> torch.Tensor:
        """Estimate box quality from the top probabilities of each DFL edge distribution."""
        batch, _, height, width = regression.shape
        distribution = regression.view(batch, 4, self.reg_max, height, width).softmax(2)
        return self.reg_conf(distribution.amax(dim=2))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Fuse classification confidence with DFL-derived localization quality."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            regression = self.cv2[i](feature)
            classification = self.cv3[i](feature)
            if self.training:
                quality = self._localization_quality(regression)
                joint_probability = classification.sigmoid() * quality
                classification = torch.logit(joint_probability.clamp(1e-4, 1.0 - 1e-4))
            x[i] = torch.cat((regression, classification), 1)
        if self.training:
            return x
        y = self._inference_with_quality(x)
        return y if self.export else (y, x)

    def _inference_with_quality(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode DFL boxes and estimate quality from the same softmax distribution."""
        shape = x[0].shape
        x_cat = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (anchor.transpose(0, 1) for anchor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, classification = x_cat.split((self.reg_max * 4, self.nc), 1)
        distribution = box.view(shape[0], 4, self.reg_max, -1).transpose(2, 1).softmax(1)
        distances = self.dfl.conv(distribution).view(shape[0], 4, -1)
        quality = self.reg_conf(distribution.amax(dim=1).unsqueeze(2)).squeeze(2)
        boxes = self.decode_bboxes(distances, self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((boxes, classification.sigmoid() * quality), 1)


class RepSharedChannelScaleDetect(RepSharedDetect):
    """RepShared head with foldable task-specific classification channel selection."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add bounded identity-initialized channel gains before each class projection."""
        super().__init__(nc, width_or_ch, ch)
        channels = self.cv3[0].in_channels
        self.cls_scale_logits = nn.ParameterList(nn.Parameter(torch.zeros(channels)) for _ in range(self.nl))
        self._cls_scale_fused = False

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Select classification channels without changing the regression feature path."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            cls_feature = feature
            if not self._cls_scale_fused:
                gain = 2.0 * self.cls_scale_logits[i].sigmoid()
                cls_feature = feature * gain.view(1, -1, 1, 1)
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](cls_feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def fuse(self):
        """Fold the learned channel gains into classification projection weights."""
        if self._cls_scale_fused:
            return
        with torch.no_grad():
            for cls, logits in zip(self.cv3, self.cls_scale_logits):
                gain = 2.0 * logits.sigmoid()
                cls.weight.mul_(gain.view(1, -1, 1, 1))
        self.cls_scale_logits = nn.ParameterList()
        self._cls_scale_fused = True


class RepSharedClsAdapterDetect(Detect):
    """RepShared head with a tiny task-specific nonlinear classification adapter."""

    legacy = True

    def __init__(
        self, nc: int = 80, width_or_ch: int | tuple = 48, adapter_width: int = 16, ch: tuple = ()
    ):
        """Keep the shared regression path and add a narrow 1x1 classification transform."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(RepConv(x, c, 3, 1, bn=(x == c)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(c, adapter_width, 1), nn.Conv2d(adapter_width, self.nc, 1)) for _ in ch
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Predict boxes from shared features and classes through the narrow adapter."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and final classification predictor biases."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class RepSharedGatedClsDetect(Detect):
    """RepShared head with a zero-initialized gated classification projection."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Predict classification values and gates in one 1x1 convolution."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(RepConv(x, c, 3, 1, bn=(x == c)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, 2 * self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Modulate class values with a bounded gate while leaving box features unchanged."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            value, gate = self.cv3[i](feature).chunk(2, 1)
            cls = value * (2.0 * gate.sigmoid())
            x[i] = torch.cat((self.cv2[i](feature), cls), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize the value path normally and the gate as an identity multiplier."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
            cls.weight.data[self.nc :].zero_()
            cls.bias.data[self.nc :].zero_()


class RepSharedIsolatedGatedClsDetect(RepSharedGatedClsDetect):
    """Train the V42 main path unchanged while learning a detached deployment-time class gate."""

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Keep gates outside TAL and main-loss gradients during training."""
        if not self.training:
            return super().forward(x)

        self.gated_cls_maps = []
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            predictor = self.cv3[i]
            value = F.conv2d(feature, predictor.weight[: self.nc], predictor.bias[: self.nc])
            gate = F.conv2d(feature.detach(), predictor.weight[self.nc :], predictor.bias[self.nc :])
            self.gated_cls_maps.append(value.detach() * (2.0 * gate.sigmoid()))
            x[i] = torch.cat((self.cv2[i](feature), value), 1)
        return x


class RepSharedP5DirectDetect(Detect):
    """RepShared spatial heads on P2-P4 with a direct low-cost P5 predictor."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Use RepConv stems except on the final low-resolution detection scale."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(
            RepConv(x, c, 3, 1, bn=(x == c)) if i < len(ch) - 1 else nn.Identity()
            for i, x in enumerate(ch)
        )
        predictor_channels = [c] * (len(ch) - 1) + [ch[-1]]
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in predictor_channels)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in predictor_channels)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply spatial refinement to P2-P4 and direct prediction to P5."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for every detection scale."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class RepSharedP5DWDetect(RepSharedP5DirectDetect):
    """Use one depthwise spatial filter before the otherwise direct final-scale predictor."""

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Recover final-scale spatial context with a low-cost depthwise convolution."""
        actual_ch = tuple(width_or_ch) if isinstance(width_or_ch, (list, tuple)) else tuple(ch)
        super().__init__(nc, width_or_ch, ch)
        self.stem[-1] = DWConv(actual_ch[-1], actual_ch[-1], 3, 1)


class RepSharedP5AuxDetect(RepSharedP5DirectDetect):
    """Fast direct final-scale head with spatial final-scale supervision used only during training."""

    auxiliary_train = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Add a removable RepConv auxiliary predictor to the direct final detection scale."""
        actual_ch = tuple(width_or_ch) if isinstance(width_or_ch, (list, tuple)) else tuple(ch)
        super().__init__(nc, width_or_ch, ch)
        c = 48 if isinstance(width_or_ch, (list, tuple)) else int(width_or_ch)
        self.aux_stem = RepConv(actual_ch[-1], c, 3, 1, bn=(actual_ch[-1] == c))
        self.aux_cv2 = nn.Conv2d(c, 4 * self.reg_max, 1)
        self.aux_cv3 = nn.Conv2d(c, self.nc, 1)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | dict | tuple:
        """Return a detached-context auxiliary pyramid in training and only the fast head in inference."""
        features = [self.stem[i](x[i]) for i in range(self.nl)]
        main = [torch.cat((self.cv2[i](features[i]), self.cv3[i](features[i])), 1) for i in range(self.nl)]
        if self.training:
            if not self.stride.any():
                return main
            aux_feature = self.aux_stem(x[-1])
            aux_final = torch.cat((self.aux_cv2(aux_feature), self.aux_cv3(aux_feature)), 1)
            auxiliary = [prediction.detach() for prediction in main[:-1]] + [aux_final]
            return {"main": main, "auxiliary": auxiliary}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize deployment predictors and the training-only final-scale predictor."""
        super().bias_init()
        stride = self.stride[-1]
        self.aux_cv2.bias.data[:] = 1.0
        self.aux_cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def fuse(self):
        """Delete all training-only final-scale modules before deployment."""
        self.aux_stem = nn.Identity()
        self.aux_cv2 = nn.Identity()
        self.aux_cv3 = nn.Identity()


class RepSharedLeadAuxDetect(RepSharedDetect):
    """V42 deployment head with removable lead-guided localization supervision from backbone P5."""

    auxiliary_train = "lead_guided"

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build the three deployed scales and a training-only box predictor from a fourth input."""
        actual_ch = tuple(width_or_ch) if isinstance(width_or_ch, (list, tuple)) else tuple(ch)
        if len(actual_ch) != 4:
            raise ValueError(f"RepSharedLeadAuxDetect expects three main inputs plus backbone P5, got {len(actual_ch)}")
        width = 48 if isinstance(width_or_ch, (list, tuple)) else int(width_or_ch)
        super().__init__(nc, width, actual_ch[:3])
        self.aux_stem = RepConv(actual_ch[3], width, 3, 1, bn=(actual_ch[3] == width))
        self.aux_cv2 = nn.ModuleList(nn.Conv2d(width, 4 * self.reg_max, 1) for _ in range(self.nl))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | dict | tuple:
        """Use main-head assignments to supervise a removable P5 localization branch during training."""
        features = [self.stem[i](x[i]) for i in range(self.nl)]
        main = [torch.cat((self.cv2[i](features[i]), self.cv3[i](features[i])), 1) for i in range(self.nl)]
        if self.training:
            if not self.stride.any():
                return main
            auxiliary_feature = self.aux_stem(x[self.nl])
            auxiliary_box = [
                predictor(F.interpolate(auxiliary_feature, size=prediction.shape[-2:], mode="nearest"))
                for predictor, prediction in zip(self.aux_cv2, main)
            ]
            return {"main": main, "auxiliary_box": auxiliary_box}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize deployed predictors and the training-only box projection."""
        super().bias_init()
        for predictor in self.aux_cv2:
            predictor.bias.data[:] = 1.0

    def fuse(self):
        """Delete the complete training-only branch before deployment."""
        self.aux_stem = nn.Identity()
        self.aux_cv2 = nn.ModuleList()


class RepSharedLeadPyramidAuxDetect(RepSharedDetect):
    """V68 deployment head with removable scale-matched backbone localization supervision."""

    auxiliary_train = "lead_guided"

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build three deployed scales and direct training-only box projections from matching backbone scales."""
        actual_ch = tuple(width_or_ch) if isinstance(width_or_ch, (list, tuple)) else tuple(ch)
        if len(actual_ch) != 6:
            raise ValueError(
                f"RepSharedLeadPyramidAuxDetect expects three main and three auxiliary inputs, got {len(actual_ch)}"
            )
        width = 48 if isinstance(width_or_ch, (list, tuple)) else int(width_or_ch)
        super().__init__(nc, width, actual_ch[:3])
        self.aux_cv2 = nn.ModuleList(nn.Conv2d(channels, 4 * self.reg_max, 1) for channels in actual_ch[3:])

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | dict | tuple:
        """Predict deployed outputs and scale-aligned backbone localization outputs during training."""
        features = [self.stem[i](x[i]) for i in range(self.nl)]
        main = [torch.cat((self.cv2[i](features[i]), self.cv3[i](features[i])), 1) for i in range(self.nl)]
        if self.training:
            if not self.stride.any():
                return main
            auxiliary_box = [predictor(x[self.nl + i]) for i, predictor in enumerate(self.aux_cv2)]
            return {"main": main, "auxiliary_box": auxiliary_box}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize deployed predictors and direct auxiliary box projections."""
        super().bias_init()
        for predictor in self.aux_cv2:
            predictor.bias.data[:] = 1.0

    def fuse(self):
        """Delete all training-only box projections before deployment."""
        self.aux_cv2 = nn.ModuleList()


class RepSharedFineSpatialDetect(Detect):
    """Use spatial refinement only on the finest detection scale."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Keep a RepConv on P2 and predict directly from already-fused P3/P4 neck features."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(
            RepConv(x, c, 3, 1, bn=(x == c)) if i == 0 else nn.Identity() for i, x in enumerate(ch)
        )
        predictor_channels = [c, *ch[1:]]
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in predictor_channels)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in predictor_channels)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Refine P2 spatially and project all scales to box and class outputs."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class biases for each scale."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class PartialSpatialDetect(Detect):
    """Efficient three-scale head with pointwise projection and partial spatial refinement."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Refine one quarter of projected channels spatially at every detection scale."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(
            nn.Sequential(Conv(x, c, 1, 1), Partial_conv3(c, 4, "split_cat")) for x in ch
        )
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply partial spatial refinement before task-specific projections."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each feature-map stride."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class AsymRepSharedDetect(Detect):
    """Three-scale shared-task head with foldable asymmetric spatial stems."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 40, ch: tuple = ()):
        """Use 40-channel asymmetric RepConv stems that deploy as one 3x3 per scale."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 40
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.stem = nn.ModuleList(AsymRepConv(x, c, 3, 1, bn=(x == c)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for _ in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Apply foldable spatial refinement before box and class prediction."""
        for i in range(self.nl):
            feature = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize box and class output biases for each scale."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class RepBoxDetect(Detect):
    """Detect head with a reparameterizable spatial box branch and direct classification."""

    legacy = True

    def __init__(self, nc: int = 80, width_or_ch: int | tuple = 48, ch: tuple = ()):
        """Build a configurable-width RepConv regression stem and direct class predictors."""
        if isinstance(width_or_ch, (list, tuple)):
            ch, c = tuple(width_or_ch), 48
        else:
            c = int(width_or_ch)
        super().__init__(nc, ch)
        self.cv2 = nn.ModuleList(
            nn.Sequential(RepConv(x, c, 3, 1, bn=(x == c)), nn.Conv2d(c, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

    def bias_init(self):
        """Initialize reparameterizable box and direct class predictor biases."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        if self.export:
            # NCNN fix
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        """Fuse text features with model weights for efficient inference."""
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()

        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward_lrpc(self, x: list[torch.Tensor], return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x: list[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for detection heads."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self, nc: int = 80, nm: int = 32, npr: int = 256, embed: int = 512, with_bn: bool = False, ch: tuple = ()
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> tuple | torch.Tensor:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mc, p

        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(
        self,
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)

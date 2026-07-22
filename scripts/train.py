"""Train SRPA-YOLO with scale-aligned response distillation."""

from __future__ import annotations

import argparse
import math
import sys
from types import MethodType
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model


class ScaleAlignedDistillationTrainer(DetectionTrainer):
    """Add training-only KD without registering the teacher in student checkpoints."""

    teacher_weights: str
    kd_temperature: float
    kd_cls_weight: float
    kd_cls_focus_weight: float
    kd_cls_focus_gamma: float
    kd_cls_focus_start: float
    kd_hard_negative_weight: float
    kd_hard_negative_gamma: float
    gt_hard_negative_weight: float
    gt_hard_negative_gamma: float
    localization_rank_weight: float
    quality_residual_weight: float
    center_heatmap_weight: float
    kd_gate_weight: float
    kd_gate_gamma: float
    kd_dfl_weight: float
    kd_dfl_focus_weight: float
    kd_dfl_focus_gamma: float
    kd_box_focus_weight: float
    kd_dfl_tal_weight: float
    kd_cross_head: bool
    kd_detached_head: bool
    kd_cls_schedule: str
    kd_dfl_schedule: str
    kd_foreground_topk: int
    kd_feature_weight: float
    kd_multiscale_attention_weight: float
    kd_assignment_distillation: bool
    kd_assignment_end_epoch: int
    kd_feature_schedule: str
    kd_student_feature_layer: int
    kd_teacher_feature_layer: int
    tal_topk: int
    tal_alpha: float
    tal_beta: float
    nwd_weight: float
    nwd_constant: float
    nwd_max_size: float
    inner_iou_ratio: float
    alpha_iou_power: float
    use_varifocal: bool
    quality_focal_weight: float
    quality_focal_gamma: float
    rank_sort_weight: float
    rank_sort_start_epoch: int
    cls_only_start_epoch: int
    lead_aux_end_epoch: int

    def _model_train(self) -> None:
        """Optionally restrict the late training phase to classification projections."""
        super()._model_train()
        student = unwrap_model(self.model)
        detect = student.model[-1]
        if getattr(detect, "contrast_private_training", False):
            student.eval()
            for stem in detect.private_stem:
                stem.contrast_conv.train()
            return
        if getattr(detect, "diverse_private_training", False):
            student.eval()
            for stem in detect.private_stem:
                stem.avg_conv.train()
            return
        if getattr(detect, "quality_residual_training", False):
            student.eval()
            detect.cv3_quality.train()
            return
        if getattr(detect, "private_residual_training", False):
            student.eval()
            detect.train()
            detect.shared_stem.eval()
            detect.cv2.eval()
            detect.cv3_shared.eval()
            detect.private_stem.train()
            detect.cv3_private.train()
            return
        if self.cls_only_start_epoch <= 0 or self.epoch < self.cls_only_start_epoch:
            return

        if not getattr(self, "_cls_only_active", False):
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in detect.cv3.parameters():
                parameter.requires_grad_(True)
            self._cls_only_active = True
            trainable = sum(parameter.numel() for parameter in detect.cv3.parameters() if parameter.requires_grad)
            LOGGER.info(
                "Classification-only calibration enabled at epoch %d: %d trainable cv3 parameters",
                self.epoch + 1,
                trainable,
            )
        student.eval()
        detect.cv3.train()

    def _setup_train(self) -> None:
        super()._setup_train()
        teacher = YOLO(self.teacher_weights).model.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_model = teacher

        student = unwrap_model(self.model)
        detect = student.model[-1]
        if getattr(detect, "contrast_private_training", False):
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for stem in detect.private_stem:
                for parameter in stem.contrast_conv.parameters():
                    parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            LOGGER.info("Foldable center-surround private-context training: %d trainable parameters", trainable)
        elif getattr(detect, "diverse_private_training", False):
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for stem in detect.private_stem:
                for parameter in stem.avg_conv.parameters():
                    parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            LOGGER.info("Foldable diverse private-context training: %d trainable parameters", trainable)
        elif getattr(detect, "quality_residual_training", False):
            detect.cache_quality_logits = True
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in detect.cv3_quality.parameters():
                parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            LOGGER.info("Foldable localization-quality residual training: %d trainable parameters", trainable)
        elif getattr(detect, "private_residual_training", False):
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in detect.private_stem.parameters():
                parameter.requires_grad_(True)
            for parameter in detect.cv3_private.parameters():
                parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            LOGGER.info("Private classification residual training: %d trainable parameters", trainable)
        self.student_feature = None
        self.teacher_feature = None
        self.student_head_inputs = None
        self.student_head_features = None
        self.student_multiscale_features = None
        self.teacher_multiscale_features = None
        if self.kd_cross_head:
            student.model[-1].register_forward_pre_hook(
                lambda module, inputs: setattr(self, "student_head_inputs", tuple(inputs[0]))
            )
        if self.kd_detached_head:
            detect = student.model[-1]
            self.student_head_features = [None] * detect.nl
            for index, stem in enumerate(detect.stem):
                stem.register_forward_hook(
                    lambda module, inputs, output, index=index: self.student_head_features.__setitem__(index, output)
                )
        if self.kd_feature_weight > 0:
            student.model[self.kd_student_feature_layer].register_forward_hook(
                lambda module, inputs, output: setattr(self, "student_feature", output)
            )
            teacher.model[self.kd_teacher_feature_layer].register_forward_hook(
                lambda module, inputs, output: setattr(self, "teacher_feature", output)
            )
        if self.kd_multiscale_attention_weight > 0:
            student.model[-1].register_forward_pre_hook(
                lambda module, inputs: setattr(self, "student_multiscale_features", tuple(inputs[0]))
            )
            teacher.model[-1].register_forward_pre_hook(
                lambda module, inputs: setattr(self, "teacher_multiscale_features", tuple(inputs[0]))
            )
        original_loss = student.loss
        trainer = self

        def loss_with_distillation(
            model: DetectionModel, batch: dict[str, torch.Tensor], preds=None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if preds is None:
                preds = model.forward(batch["img"])
            if (
                trainer.nwd_weight > 0
                or trainer.tal_topk > 0
                or trainer.tal_alpha > 0
                or trainer.tal_beta > 0
                or trainer.inner_iou_ratio != 1.0
                or trainer.use_varifocal
                or trainer.quality_focal_weight > 0
            ):
                if getattr(model, "criterion", None) is None:
                    model.criterion = model.init_criterion()
            if hasattr(model.criterion, "main"):
                criteria = tuple(
                    criterion
                    for criterion in (model.criterion.main, getattr(model.criterion, "auxiliary", None))
                    if criterion is not None
                )
            else:
                criteria = (model.criterion,)
            for criterion in criteria:
                if trainer.use_varifocal:
                    criterion.use_varifocal = True
                if trainer.quality_focal_weight > 0:
                    criterion.quality_focal_weight = trainer.quality_focal_weight
                    criterion.quality_focal_gamma = trainer.quality_focal_gamma
                criterion.rank_sort_weight = (
                    trainer.rank_sort_weight
                    if trainer.rank_sort_weight > 0 and trainer.epoch >= trainer.rank_sort_start_epoch
                    else 0.0
                )
                if trainer.tal_topk > 0:
                    criterion.assigner.topk = trainer.tal_topk
                if trainer.tal_alpha > 0:
                    criterion.assigner.alpha = trainer.tal_alpha
                if trainer.tal_beta > 0:
                    criterion.assigner.beta = trainer.tal_beta
                if trainer.nwd_weight > 0:
                    criterion.bbox_loss.nwd_weight = trainer.nwd_weight
                    criterion.bbox_loss.nwd_constant = trainer.nwd_constant
                    criterion.bbox_loss.nwd_max_size = trainer.nwd_max_size
                if trainer.inner_iou_ratio != 1.0:
                    criterion.bbox_loss.inner_iou_ratio = trainer.inner_iou_ratio
                if trainer.alpha_iou_power != 1.0:
                    criterion.bbox_loss.alpha_iou_power = trainer.alpha_iou_power
                if trainer.kd_dfl_tal_weight > 0:
                    criterion.cache_assignments = True
                if trainer.gt_hard_negative_weight > 0:
                    criterion.cache_assignments = True
                if trainer.localization_rank_weight > 0:
                    criterion.cache_assignments = True
                if trainer.quality_residual_weight > 0:
                    criterion.cache_assignments = True
            if hasattr(model.criterion, "auxiliary_weight"):
                auxiliary_active = trainer.lead_aux_end_epoch <= 0 or trainer.epoch < trainer.lead_aux_end_epoch
                model.criterion.auxiliary_weight = (
                    model.criterion.base_auxiliary_weight if auxiliary_active else 0.0
                )
                if (
                    not auxiliary_active
                    and not getattr(trainer, "_lead_aux_disabled_logged", False)
                ):
                    LOGGER.info(
                        "Lead-guided localization auxiliary disabled at epoch %d; continuing with the deployed head only",
                        trainer.epoch + 1,
                    )
                    trainer._lead_aux_disabled_logged = True
            with torch.no_grad():
                teacher_output = trainer.teacher_model(batch["img"])
                teacher_maps = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
            assignment_distillation_active = trainer.kd_assignment_distillation and (
                trainer.kd_assignment_end_epoch <= 0 or trainer.epoch < trainer.kd_assignment_end_epoch
            )
            if (
                trainer.kd_assignment_distillation
                and not assignment_distillation_active
                and not getattr(trainer, "_assignment_distillation_disabled_logged", False)
            ):
                LOGGER.info(
                    "Label Assignment Distillation disabled at epoch %d; returning TAL guidance to the student",
                    trainer.epoch + 1,
                )
                trainer._assignment_distillation_disabled_logged = True
            if assignment_distillation_active:
                for criterion in criteria:
                    criterion.assignment_guide = teacher_maps
            try:
                detection_loss, loss_items = original_loss(batch, preds)
            finally:
                if assignment_distillation_active:
                    for criterion in criteria:
                        criterion.assignment_guide = None

            student_maps = preds[1] if isinstance(preds, tuple) else preds
            # End-to-end heads train one-to-many and one-to-one branches together.
            # Distill only the deployed one-to-one branch so KD directly improves
            # the inference path without duplicating supervision on the auxiliary head.
            if isinstance(student_maps, dict):
                student_maps = student_maps.get("one2one", student_maps.get("main"))

            if trainer.gt_hard_negative_weight > 0:
                cache = criteria[0].assignment_cache
                if cache is None:
                    raise RuntimeError("Ground-truth hard-negative loss requires cached TAL assignments")
                detect = model.model[-1]
                flat_scores = torch.cat(
                    [feature.view(feature.shape[0], detect.no, -1) for feature in student_maps], dim=2
                )[:, 4 * detect.reg_max :].permute(0, 2, 1).contiguous()
                negative_mask = ~cache["fg_mask"]
                negative_logits = flat_scores[negative_mask]
                negative_probability = negative_logits.sigmoid()
                negative_weight = negative_probability.detach().pow(trainer.gt_hard_negative_gamma)
                gt_hard_negative = (F.softplus(negative_logits) * negative_weight).sum() / negative_weight.sum().clamp_min(1.0)
            else:
                gt_hard_negative = student_maps[0].new_zeros(())

            if trainer.localization_rank_weight > 0:
                cache = criteria[0].assignment_cache
                if cache is None:
                    raise RuntimeError("Localization ranking requires cached TAL assignments")
                detect = model.model[-1]
                reg_channels = 4 * detect.reg_max
                batch_size = student_maps[0].shape[0]
                flat_predictions = torch.cat(
                    [feature.view(batch_size, detect.no, -1) for feature in student_maps], dim=2
                )
                distribution, class_logits = flat_predictions.split((reg_channels, detect.nc), dim=1)
                distribution = distribution.permute(0, 2, 1).contiguous()
                class_scores = class_logits.permute(0, 2, 1).contiguous().amax(dim=-1)
                with torch.no_grad():
                    predicted_boxes = criteria[0].bbox_decode(cache["anchor_points"], distribution.detach())
                    target_boxes = cache["target_bboxes"] / cache["stride_tensor"]

                localization_rank = class_scores.new_zeros(())
                valid_images = 0
                for image_index in range(batch_size):
                    foreground = cache["fg_mask"][image_index]
                    if foreground.sum() < 2:
                        continue
                    scores = class_scores[image_index, foreground]
                    quality = bbox_iou(
                        predicted_boxes[image_index, foreground],
                        target_boxes[image_index, foreground],
                        xywh=False,
                    ).squeeze(-1).clamp_(0, 1)
                    quality_gap = quality[:, None] - quality[None, :]
                    ordered = quality_gap > 0
                    if not ordered.any():
                        continue
                    score_gap = scores[:, None] - scores[None, :]
                    pair_weight = quality_gap[ordered]
                    localization_rank += (
                        F.softplus(-score_gap[ordered]) * pair_weight
                    ).sum() / pair_weight.sum().clamp_min(1e-6)
                    valid_images += 1
                localization_rank = localization_rank / max(valid_images, 1)
            else:
                localization_rank = student_maps[0].new_zeros(())

            if trainer.quality_residual_weight > 0:
                cache = criteria[0].assignment_cache
                if cache is None:
                    raise RuntimeError("Quality residual supervision requires cached TAL assignments")
                detect = model.model[-1]
                if not detect.base_cls_maps or not detect.quality_residual_maps:
                    raise RuntimeError("Quality residual head did not expose its training logits")
                batch_size = student_maps[0].shape[0]
                reg_channels = 4 * detect.reg_max
                distribution = torch.cat(
                    [feature[:, :reg_channels].view(batch_size, reg_channels, -1) for feature in student_maps], dim=2
                ).permute(0, 2, 1).contiguous()
                base_logits = torch.cat(
                    [feature.view(batch_size, detect.nc, -1) for feature in detect.base_cls_maps], dim=2
                ).permute(0, 2, 1).contiguous()
                residual_logits = torch.cat(
                    [feature.view(batch_size, detect.nc, -1) for feature in detect.quality_residual_maps], dim=2
                ).permute(0, 2, 1).contiguous()
                foreground = cache["fg_mask"]
                if foreground.any():
                    with torch.no_grad():
                        predicted_boxes = criteria[0].bbox_decode(cache["anchor_points"], distribution.detach())
                        target_boxes = cache["target_bboxes"] / cache["stride_tensor"]
                        iou = bbox_iou(
                            predicted_boxes[foreground], target_boxes[foreground], xywh=False
                        ).squeeze(-1).clamp_(0.05, 0.95)
                        target_classes = cache["target_scores"][foreground].argmax(dim=-1, keepdim=True)
                        selected_base = base_logits.detach()[foreground].gather(1, target_classes).squeeze(1)
                        target_correction = torch.logit(iou) - selected_base
                        pair_weight = cache["target_scores"][foreground].amax(dim=-1).clamp_min(1e-3)
                    selected_residual = residual_logits[foreground].gather(1, target_classes).squeeze(1)
                    quality_residual = (
                        F.smooth_l1_loss(selected_residual, target_correction, reduction="none") * pair_weight
                    ).sum() / pair_weight.sum().clamp_min(1e-6)
                else:
                    quality_residual = student_maps[0].new_zeros(())
            else:
                quality_residual = student_maps[0].new_zeros(())

            if trainer.center_heatmap_weight > 0:
                detect = model.model[-1]
                if not detect.base_cls_maps or not detect.quality_residual_maps:
                    raise RuntimeError("Center heatmap supervision requires cached residual logits")
                batch_size = student_maps[0].shape[0]
                image_indices = batch["batch_idx"].long().view(-1)
                boxes = batch["bboxes"]
                center_heatmap = student_maps[0].new_zeros(())
                valid_scales = 0
                for base_map, residual_map in zip(detect.base_cls_maps, detect.quality_residual_maps):
                    height, width = residual_map.shape[-2:]
                    yy, xx = torch.meshgrid(
                        torch.arange(height, device=residual_map.device, dtype=residual_map.dtype),
                        torch.arange(width, device=residual_map.device, dtype=residual_map.dtype),
                        indexing="ij",
                    )
                    target = residual_map.new_zeros((batch_size, 1, height, width))
                    for object_index, image_index in enumerate(image_indices.tolist()):
                        cx, cy, box_width, box_height = boxes[object_index]
                        center_x, center_y = cx * width, cy * height
                        sigma_x = (box_width * width / 6).clamp_min(0.5)
                        sigma_y = (box_height * height / 6).clamp_min(0.5)
                        gaussian = torch.exp(
                            -0.5 * (((xx - center_x) / sigma_x).pow(2) + ((yy - center_y) / sigma_y).pow(2))
                        )
                        target[image_index, 0] = torch.maximum(target[image_index, 0], gaussian)
                    final_logits = base_map.detach() + residual_map
                    probability = final_logits.sigmoid()
                    focal_weight = target * (1 - probability.detach()).pow(2)
                    focal_weight += (1 - target).pow(4) * probability.detach().pow(2)
                    scale_loss = F.binary_cross_entropy_with_logits(
                        final_logits, target.expand_as(final_logits), reduction="none"
                    )
                    center_heatmap += (scale_loss * focal_weight).sum() / focal_weight.sum().clamp_min(1.0)
                    valid_scales += 1
                center_heatmap = center_heatmap / max(valid_scales, 1)
            else:
                center_heatmap = student_maps[0].new_zeros(())

            if trainer.kd_cross_head:
                kd_maps = model._cross_head_predictions(teacher_maps)
            elif trainer.kd_detached_head:
                kd_maps = model._detached_head_predictions()
            else:
                kd_maps = student_maps
            kd_cls, kd_cls_focus, kd_hard_negative, kd_dfl, kd_dfl_focus, matched = model._scale_aligned_kd(
                kd_maps, teacher_maps, batch["batch_idx"]
            )
            if not matched:
                raise RuntimeError("No spatially aligned student/teacher detection maps were found")
            batch_size = batch["img"].shape[0]
            detection_loss = detection_loss.clone()
            progress = min(max((trainer.epoch + 1) / trainer.epochs, 0.0), 1.0)

            def response_scale(schedule: str) -> float:
                if schedule != "cosine":
                    return 1.0
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            cls_scale = response_scale(trainer.kd_cls_schedule)
            dfl_scale = response_scale(trainer.kd_dfl_schedule)
            cls_focus_scale = 1.0
            if trainer.kd_cls_focus_start > 0:
                cls_focus_scale = max(
                    0.0,
                    min(
                        1.0,
                        (progress - trainer.kd_cls_focus_start) / max(1.0 - trainer.kd_cls_focus_start, 1e-6),
                    ),
                )
            detection_loss[0] += dfl_scale * trainer.kd_dfl_weight * kd_dfl * batch_size
            detection_loss[0] += dfl_scale * trainer.kd_dfl_focus_weight * kd_dfl_focus * batch_size
            if trainer.kd_box_focus_weight > 0:
                kd_box_focus = model._teacher_confident_box_kd(kd_maps, teacher_maps, criteria[0])
                detection_loss[0] += dfl_scale * trainer.kd_box_focus_weight * kd_box_focus * batch_size
            if trainer.kd_dfl_tal_weight > 0:
                kd_dfl_tal = model._tal_foreground_dfl_kd(kd_maps, teacher_maps, criteria[0])
                detection_loss[0] += dfl_scale * trainer.kd_dfl_tal_weight * kd_dfl_tal * batch_size
            detection_loss[1] += cls_scale * trainer.kd_cls_weight * kd_cls * batch_size
            detection_loss[1] += (
                cls_scale * cls_focus_scale * trainer.kd_cls_focus_weight * kd_cls_focus * batch_size
            )
            detection_loss[1] += trainer.kd_hard_negative_weight * kd_hard_negative * batch_size
            detection_loss[1] += trainer.gt_hard_negative_weight * gt_hard_negative * batch_size
            detection_loss[1] += trainer.localization_rank_weight * localization_rank * batch_size
            detection_loss[1] += trainer.quality_residual_weight * quality_residual * batch_size
            detection_loss[1] += trainer.center_heatmap_weight * center_heatmap * batch_size
            if trainer.kd_gate_weight > 0:
                gate_kd = model._isolated_gate_kd(teacher_maps)
                detection_loss[1] += trainer.kd_gate_weight * gate_kd * batch_size
            if trainer.kd_feature_weight > 0:
                if trainer.student_feature is None or trainer.teacher_feature is None:
                    raise RuntimeError("P5 feature hooks did not capture student and teacher tensors")
                feature_kd = model._attention_transfer_kd(trainer.student_feature, trainer.teacher_feature)
                feature_weight = trainer.kd_feature_weight
                if trainer.kd_feature_schedule == "cosine":
                    progress = min(max((trainer.epoch + 1) / trainer.epochs, 0.0), 1.0)
                    feature_weight *= 0.5 * (1.0 + math.cos(math.pi * progress))
                detection_loss[1] += feature_weight * feature_kd * batch_size
            if trainer.kd_multiscale_attention_weight > 0:
                multiscale_attention = model._multiscale_attention_kd()
                detection_loss[1] += trainer.kd_multiscale_attention_weight * multiscale_attention * batch_size
            return detection_loss, loss_items

        def cross_head_predictions(
            model: DetectionModel, teacher_maps: list[torch.Tensor]
        ) -> list[torch.Tensor]:
            """Route student neck features through frozen teacher predictors for conflict-free KD."""
            if trainer.student_head_inputs is None:
                raise RuntimeError("Cross-head KD did not capture student detection-head inputs")

            teacher_detect = trainer.teacher_model.model[-1]
            predictions = []
            used_teacher_indices = set()
            for student_feature in trainer.student_head_inputs:
                spatial_size = student_feature.shape[-2:]
                teacher_index = next(
                    (
                        index
                        for index, teacher_map in enumerate(teacher_maps)
                        if teacher_map.shape[-2:] == spatial_size and index not in used_teacher_indices
                    ),
                    None,
                )
                if teacher_index is None:
                    raise RuntimeError(f"No teacher head branch matches student feature size {spatial_size}")
                used_teacher_indices.add(teacher_index)
                expected_channels = teacher_detect.cv2[teacher_index][0].conv.in_channels
                if student_feature.shape[1] != expected_channels:
                    raise RuntimeError(
                        f"Cross-head channel mismatch at {spatial_size}: "
                        f"student={student_feature.shape[1]}, teacher={expected_channels}"
                    )
                predictions.append(
                    torch.cat(
                        (
                            teacher_detect.cv2[teacher_index](student_feature),
                            teacher_detect.cv3[teacher_index](student_feature),
                        ),
                        dim=1,
                    )
                )
            return predictions

        def detached_head_predictions(model: DetectionModel) -> list[torch.Tensor]:
            """Recompute predictions with detached output weights so KD updates shared features only."""
            if not trainer.student_head_features or any(
                feature is None for feature in trainer.student_head_features
            ):
                raise RuntimeError("Detached-head KD did not capture shared stem features")
            detect = model.model[-1]
            predictions = []
            for feature, box, cls in zip(trainer.student_head_features, detect.cv2, detect.cv3):
                if not isinstance(box, torch.nn.Conv2d) or not isinstance(cls, torch.nn.Conv2d):
                    raise TypeError("Detached-head KD requires direct Conv2d box and class projections")
                predictions.append(
                    torch.cat(
                        (
                            F.conv2d(
                                feature,
                                box.weight.detach(),
                                None if box.bias is None else box.bias.detach(),
                            ),
                            F.conv2d(
                                feature,
                                cls.weight.detach(),
                                None if cls.bias is None else cls.bias.detach(),
                            ),
                        ),
                        dim=1,
                    )
                )
            return predictions

        def scale_aligned_kd(
            model: DetectionModel,
            student_maps: list[torch.Tensor],
            teacher_maps: list[torch.Tensor],
            batch_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            detect = model.model[-1]
            reg_max, nc = detect.reg_max, detect.nc
            temperature = trainer.kd_temperature
            teacher_by_size = {feature.shape[-2:]: feature.detach() for feature in teacher_maps}
            aligned = [
                (student_feature, teacher_by_size[student_feature.shape[-2:]])
                for student_feature in student_maps
                if student_feature.shape[-2:] in teacher_by_size
                and teacher_by_size[student_feature.shape[-2:]].shape[1] == student_feature.shape[1]
            ]
            cls_total = student_maps[0].new_zeros(())
            cls_focus_total = student_maps[0].new_zeros(())
            hard_negative_total = student_maps[0].new_zeros(())
            dfl_total = student_maps[0].new_zeros(())
            dfl_focus_total = student_maps[0].new_zeros(())
            matched = len(aligned)

            if not matched:
                return cls_total, cls_focus_total, hard_negative_total, dfl_total, dfl_focus_total, 0

            confidences = [
                torch.sigmoid(teacher_feature[:, 4 * reg_max :] / temperature).amax(dim=1, keepdim=True)
                for _, teacher_feature in aligned
            ]
            flat_confidence = torch.cat([confidence.flatten(1) for confidence in confidences], dim=1)
            foreground_masks = None
            if trainer.kd_foreground_topk > 0:
                foreground = torch.zeros_like(flat_confidence, dtype=torch.bool)
                gt_counts = torch.bincount(batch_indices.long(), minlength=flat_confidence.shape[0])
                for image_index, gt_count in enumerate(gt_counts):
                    topk = min(
                        flat_confidence.shape[1],
                        max(trainer.kd_foreground_topk, int(gt_count) * trainer.kd_foreground_topk),
                    )
                    indices = flat_confidence[image_index].topk(topk).indices
                    foreground[image_index, indices] = True
                foreground_masks = torch.split(
                    foreground, [confidence.shape[-2] * confidence.shape[-1] for confidence in confidences], dim=1
                )

            for scale_index, ((student_feature, teacher_feature), teacher_probability) in enumerate(
                zip(aligned, confidences)
            ):
                student_reg, student_cls = student_feature.split((4 * reg_max, nc), dim=1)
                teacher_reg, teacher_cls = teacher_feature.split((4 * reg_max, nc), dim=1)

                confidence = 0.05 + 0.95 * teacher_probability.amax(dim=1, keepdim=True)
                cls_map = F.binary_cross_entropy_with_logits(
                    student_cls / temperature, torch.sigmoid(teacher_cls / temperature), reduction="none"
                )
                cls_total += (cls_map * confidence).sum() / confidence.sum().clamp_min(1.0)
                cls_focus_confidence = teacher_probability.pow(trainer.kd_cls_focus_gamma)
                cls_focus_total += (cls_map * cls_focus_confidence).sum() / cls_focus_confidence.sum().clamp_min(1e-6)
                student_probability = torch.sigmoid(student_cls)
                teacher_cls_probability = torch.sigmoid(teacher_cls).detach()
                hard_negative_weight = (1.0 - teacher_cls_probability).pow(trainer.kd_hard_negative_gamma)
                hard_negative_weight = hard_negative_weight * student_probability.detach().pow(2.0)
                hard_negative_map = F.softplus(student_cls)
                hard_negative_total += (hard_negative_map * hard_negative_weight).sum() / hard_negative_weight.sum().clamp_min(1.0)

                shape = student_reg.shape
                student_distribution = student_reg.view(shape[0], 4, reg_max, *shape[2:])
                teacher_distribution = teacher_reg.view(shape[0], 4, reg_max, *shape[2:])
                dfl_map = F.kl_div(
                    F.log_softmax(student_distribution / temperature, dim=2),
                    F.softmax(teacher_distribution / temperature, dim=2),
                    reduction="none",
                ).sum(dim=2)
                if foreground_masks is None:
                    dfl_confidence = confidence
                else:
                    foreground_mask = foreground_masks[scale_index]
                    dfl_confidence = teacher_probability * foreground_mask.view(
                        foreground_mask.shape[0], 1, *teacher_probability.shape[-2:]
                    )
                dfl_total += (dfl_map * dfl_confidence).sum() * (temperature**2) / (
                    4 * dfl_confidence.sum().clamp_min(1e-6)
                )
                focus_confidence = teacher_probability.pow(trainer.kd_dfl_focus_gamma)
                dfl_focus_total += (dfl_map * focus_confidence).sum() * (temperature**2) / (
                    4 * focus_confidence.sum().clamp_min(1e-6)
                )

            return (
                cls_total * (temperature**2) / max(matched, 1),
                cls_focus_total * (temperature**2) / max(matched, 1),
                hard_negative_total / max(matched, 1),
                dfl_total / max(matched, 1),
                dfl_focus_total / max(matched, 1),
                matched,
            )

        def attention_transfer_kd(
            model: DetectionModel, student_feature: torch.Tensor, teacher_feature: torch.Tensor
        ) -> torch.Tensor:
            student_attention = student_feature.float().pow(2).mean(dim=1).flatten(1)
            teacher_attention = teacher_feature.detach().float().pow(2).mean(dim=1).flatten(1)
            if student_attention.shape != teacher_attention.shape:
                raise RuntimeError(
                    f"P5 attention shapes do not align: {student_attention.shape} vs {teacher_attention.shape}"
                )
            student_attention = F.normalize(student_attention, p=2, dim=1)
            teacher_attention = F.normalize(teacher_attention, p=2, dim=1)
            return (student_attention - teacher_attention).pow(2).sum(dim=1).mean()

        def tal_foreground_dfl_kd(
            model: DetectionModel,
            student_maps: list[torch.Tensor],
            teacher_maps: list[torch.Tensor],
            criterion: object,
        ) -> torch.Tensor:
            """Distill localization distributions only at quality-weighted TAL positives."""
            cache = getattr(criterion, "assignment_cache", None)
            if not cache:
                raise RuntimeError("TAL foreground DFL distillation requires cached assignments")

            detect = model.model[-1]
            reg_channels = 4 * detect.reg_max
            teacher_by_size = {feature.shape[-2:]: feature.detach() for feature in teacher_maps}
            aligned_teacher = [teacher_by_size.get(feature.shape[-2:]) for feature in student_maps]
            if any(feature is None for feature in aligned_teacher):
                raise RuntimeError("Teacher DFL maps do not align with every student detection scale")

            batch_size = student_maps[0].shape[0]
            student_reg = torch.cat(
                [feature[:, :reg_channels].view(batch_size, reg_channels, -1) for feature in student_maps], dim=2
            ).permute(0, 2, 1)
            teacher_reg = torch.cat(
                [feature[:, :reg_channels].view(batch_size, reg_channels, -1) for feature in aligned_teacher], dim=2
            ).permute(0, 2, 1)
            student_reg = student_reg.view(batch_size, -1, 4, detect.reg_max)
            teacher_reg = teacher_reg.view(batch_size, -1, 4, detect.reg_max)

            temperature = trainer.kd_temperature
            dfl_kl = F.kl_div(
                F.log_softmax(student_reg / temperature, dim=-1),
                F.softmax(teacher_reg / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            foreground = cache["fg_mask"]
            quality = cache["target_scores"].sum(dim=-1)
            if not foreground.any():
                return dfl_kl.new_zeros(())
            weights = quality[foreground].unsqueeze(-1)
            return (dfl_kl[foreground] * weights).sum() * (temperature**2) / (
                4 * weights.sum().clamp_min(1e-6)
            )

        def teacher_confident_box_kd(
            model: DetectionModel,
            student_maps: list[torch.Tensor],
            teacher_maps: list[torch.Tensor],
            criterion: object,
        ) -> torch.Tensor:
            """Match decoded teacher geometry at confidence-weighted locations."""
            detect = model.model[-1]
            reg_channels = 4 * detect.reg_max
            teacher_by_size = {feature.shape[-2:]: feature.detach() for feature in teacher_maps}
            aligned_teacher = [teacher_by_size.get(feature.shape[-2:]) for feature in student_maps]
            if any(feature is None for feature in aligned_teacher):
                raise RuntimeError("Teacher box maps do not align with every student detection scale")

            batch_size = student_maps[0].shape[0]
            student_reg = torch.cat(
                [feature[:, :reg_channels].view(batch_size, reg_channels, -1) for feature in student_maps], dim=2
            ).permute(0, 2, 1)
            teacher_reg = torch.cat(
                [feature[:, :reg_channels].view(batch_size, reg_channels, -1) for feature in aligned_teacher], dim=2
            ).permute(0, 2, 1)
            teacher_cls = torch.cat(
                [feature[:, reg_channels:].view(batch_size, detect.nc, -1) for feature in aligned_teacher], dim=2
            ).permute(0, 2, 1)

            anchor_points, _ = make_anchors(student_maps, detect.stride, 0.5)
            student_boxes = criterion.bbox_decode(anchor_points, student_reg)
            teacher_boxes = criterion.bbox_decode(anchor_points, teacher_reg).detach()
            confidence = torch.sigmoid(teacher_cls / trainer.kd_temperature).amax(dim=-1)
            confidence = confidence.pow(trainer.kd_dfl_focus_gamma)
            ciou_loss = 1.0 - bbox_iou(student_boxes, teacher_boxes, xywh=False, CIoU=True).squeeze(-1)
            return (ciou_loss * confidence).sum() / confidence.sum().clamp_min(1e-6)

        def multiscale_attention_kd(model: DetectionModel) -> torch.Tensor:
            """Match channel-agnostic spatial attention at every aligned detection scale."""
            if not trainer.student_multiscale_features or not trainer.teacher_multiscale_features:
                raise RuntimeError("Multiscale attention hooks did not capture detection features")
            teacher_by_size = {
                feature.shape[-2:]: feature.detach() for feature in trainer.teacher_multiscale_features
            }
            total = trainer.student_multiscale_features[0].new_zeros((), dtype=torch.float32)
            matched = 0
            for student_feature in trainer.student_multiscale_features:
                teacher_feature = teacher_by_size.get(student_feature.shape[-2:])
                if teacher_feature is None:
                    continue
                total += model._attention_transfer_kd(student_feature, teacher_feature)
                matched += 1
            if not matched:
                raise RuntimeError("No student/teacher detection features align for multiscale attention transfer")
            return total / matched

        def isolated_gate_kd(model: DetectionModel, teacher_maps: list[torch.Tensor]) -> torch.Tensor:
            detect = model.model[-1]
            gate_maps = getattr(detect, "gated_cls_maps", None)
            if not gate_maps:
                raise RuntimeError("Isolated classification gate maps were not captured")
            temperature = trainer.kd_temperature
            teacher_by_size = {feature.shape[-2:]: feature.detach() for feature in teacher_maps}
            total = gate_maps[0].new_zeros(())
            matched = 0
            for gate_map in gate_maps:
                teacher_feature = teacher_by_size.get(gate_map.shape[-2:])
                if teacher_feature is None:
                    continue
                teacher_cls = teacher_feature[:, 4 * detect.reg_max :]
                teacher_probability = torch.sigmoid(teacher_cls / temperature)
                confidence = teacher_probability.amax(dim=1, keepdim=True).pow(trainer.kd_gate_gamma)
                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_map / temperature, teacher_probability, reduction="none"
                )
                total += (gate_loss * confidence).sum() / confidence.sum().clamp_min(1e-6)
                matched += 1
            if not matched:
                raise RuntimeError("No teacher maps aligned with isolated classification gates")
            return total * (temperature**2) / matched

        student._scale_aligned_kd = MethodType(scale_aligned_kd, student)
        student._cross_head_predictions = MethodType(cross_head_predictions, student)
        student._detached_head_predictions = MethodType(detached_head_predictions, student)
        student._attention_transfer_kd = MethodType(attention_transfer_kd, student)
        student._tal_foreground_dfl_kd = MethodType(tal_foreground_dfl_kd, student)
        student._teacher_confident_box_kd = MethodType(teacher_confident_box_kd, student)
        student._multiscale_attention_kd = MethodType(multiscale_attention_kd, student)
        student._isolated_gate_kd = MethodType(isolated_gate_kd, student)
        student.loss = MethodType(loss_with_distillation, student)
        LOGGER.info(
            "Scale-aligned KD enabled: teacher=%s T=%.2f cls=%.3f/%s cls_focus=%.3f gamma=%.2f start=%.2f gate=%.4f gamma=%.2f dfl=%.3f/%s dfl_focus=%.3f gamma=%.2f box_focus=%.3f dfl_tal=%.3f cross_head=%s detached_head=%s assignment_distillation=%s foreground_topk=%d tal_topk=%d feature=%.3f/%s multiscale_attention=%.3f nwd=%.3f max_size=%.1f alpha_iou=%.2f qfl=%.3f gamma=%.2f rank_sort=%.3f start=%d",
            self.teacher_weights,
            self.kd_temperature,
            self.kd_cls_weight,
            self.kd_cls_schedule,
            self.kd_cls_focus_weight,
            self.kd_cls_focus_gamma,
            self.kd_cls_focus_start,
            self.kd_gate_weight,
            self.kd_gate_gamma,
            self.kd_dfl_weight,
            self.kd_dfl_schedule,
            self.kd_dfl_focus_weight,
            self.kd_dfl_focus_gamma,
            self.kd_box_focus_weight,
            self.kd_dfl_tal_weight,
            self.kd_cross_head,
            self.kd_detached_head,
            self.kd_assignment_distillation,
            self.kd_foreground_topk,
            self.tal_topk,
            self.kd_feature_weight,
            self.kd_feature_schedule,
            self.kd_multiscale_attention_weight,
            self.nwd_weight,
            self.nwd_max_size,
            self.alpha_iou_power,
            self.quality_focal_weight,
            self.quality_focal_gamma,
            self.rank_sort_weight,
            self.rank_sort_start_epoch + 1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not cli.config.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {cli.config}")
    if cli.resume is not None and not cli.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {cli.resume}")
    config = dict(YAML.load(cli.config))
    teacher = Path(config.pop("teacher_weights"))
    teacher = teacher if teacher.is_absolute() else ROOT / teacher
    if not teacher.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher}")
    ScaleAlignedDistillationTrainer.teacher_weights = str(teacher)
    ScaleAlignedDistillationTrainer.kd_temperature = float(config.pop("kd_temperature", 2.0))
    ScaleAlignedDistillationTrainer.kd_cls_weight = float(config.pop("kd_cls_weight", 0.25))
    ScaleAlignedDistillationTrainer.kd_cls_focus_weight = float(config.pop("kd_cls_focus_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_cls_focus_gamma = float(config.pop("kd_cls_focus_gamma", 2.0))
    ScaleAlignedDistillationTrainer.kd_cls_focus_start = float(config.pop("kd_cls_focus_start", 0.0))
    ScaleAlignedDistillationTrainer.kd_hard_negative_weight = float(config.pop("kd_hard_negative_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_hard_negative_gamma = float(config.pop("kd_hard_negative_gamma", 2.0))
    ScaleAlignedDistillationTrainer.gt_hard_negative_weight = float(config.pop("gt_hard_negative_weight", 0.0))
    ScaleAlignedDistillationTrainer.gt_hard_negative_gamma = float(config.pop("gt_hard_negative_gamma", 2.0))
    ScaleAlignedDistillationTrainer.localization_rank_weight = float(config.pop("localization_rank_weight", 0.0))
    ScaleAlignedDistillationTrainer.quality_residual_weight = float(config.pop("quality_residual_weight", 0.0))
    ScaleAlignedDistillationTrainer.center_heatmap_weight = float(config.pop("center_heatmap_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_gate_weight = float(config.pop("kd_gate_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_gate_gamma = float(config.pop("kd_gate_gamma", 2.0))
    ScaleAlignedDistillationTrainer.kd_dfl_weight = float(config.pop("kd_dfl_weight", 0.25))
    ScaleAlignedDistillationTrainer.kd_dfl_focus_weight = float(config.pop("kd_dfl_focus_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_dfl_focus_gamma = float(config.pop("kd_dfl_focus_gamma", 2.0))
    ScaleAlignedDistillationTrainer.kd_box_focus_weight = float(config.pop("kd_box_focus_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_dfl_tal_weight = float(config.pop("kd_dfl_tal_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_cross_head = bool(config.pop("kd_cross_head", False))
    ScaleAlignedDistillationTrainer.kd_detached_head = bool(config.pop("kd_detached_head", False))
    response_schedule = str(config.pop("kd_response_schedule", "constant"))
    ScaleAlignedDistillationTrainer.kd_cls_schedule = str(config.pop("kd_cls_schedule", response_schedule))
    ScaleAlignedDistillationTrainer.kd_dfl_schedule = str(config.pop("kd_dfl_schedule", response_schedule))
    ScaleAlignedDistillationTrainer.kd_foreground_topk = int(config.pop("kd_foreground_topk", 10))
    ScaleAlignedDistillationTrainer.kd_feature_weight = float(config.pop("kd_feature_weight", 0.0))
    ScaleAlignedDistillationTrainer.kd_multiscale_attention_weight = float(
        config.pop("kd_multiscale_attention_weight", 0.0)
    )
    ScaleAlignedDistillationTrainer.kd_assignment_distillation = bool(
        config.pop("kd_assignment_distillation", False)
    )
    ScaleAlignedDistillationTrainer.kd_assignment_end_epoch = int(config.pop("kd_assignment_end_epoch", 0))
    ScaleAlignedDistillationTrainer.kd_feature_schedule = str(config.pop("kd_feature_schedule", "constant"))
    ScaleAlignedDistillationTrainer.kd_student_feature_layer = int(config.pop("kd_student_feature_layer", -1))
    ScaleAlignedDistillationTrainer.kd_teacher_feature_layer = int(config.pop("kd_teacher_feature_layer", -1))
    ScaleAlignedDistillationTrainer.tal_topk = int(config.pop("tal_topk", 0))
    ScaleAlignedDistillationTrainer.tal_alpha = float(config.pop("tal_alpha", 0.0))
    ScaleAlignedDistillationTrainer.tal_beta = float(config.pop("tal_beta", 0.0))
    ScaleAlignedDistillationTrainer.nwd_weight = float(config.pop("nwd_weight", 0.0))
    ScaleAlignedDistillationTrainer.nwd_constant = float(config.pop("nwd_constant", 12.8))
    ScaleAlignedDistillationTrainer.nwd_max_size = float(config.pop("nwd_max_size", 0.0))
    ScaleAlignedDistillationTrainer.inner_iou_ratio = float(config.pop("inner_iou_ratio", 1.0))
    ScaleAlignedDistillationTrainer.alpha_iou_power = float(config.pop("alpha_iou_power", 1.0))
    ScaleAlignedDistillationTrainer.use_varifocal = bool(config.pop("use_varifocal", False))
    ScaleAlignedDistillationTrainer.quality_focal_weight = float(config.pop("quality_focal_weight", 0.0))
    ScaleAlignedDistillationTrainer.quality_focal_gamma = float(config.pop("quality_focal_gamma", 2.0))
    ScaleAlignedDistillationTrainer.rank_sort_weight = float(config.pop("rank_sort_weight", 0.0))
    ScaleAlignedDistillationTrainer.rank_sort_start_epoch = int(config.pop("rank_sort_start_epoch", 0))
    ScaleAlignedDistillationTrainer.cls_only_start_epoch = int(config.pop("cls_only_start_epoch", 0))
    ScaleAlignedDistillationTrainer.lead_aux_end_epoch = int(config.pop("lead_aux_end_epoch", 0))
    config.pop("task", None)
    config.pop("mode", None)

    if cli.resume:
        model = YOLO(cli.resume)
        resume_overrides = {
            key: config[key]
            for key in ("data", "workers", "cache", "batch", "device", "epochs", "patience", "plots", "val")
            if key in config
        }
        model.train(trainer=ScaleAlignedDistillationTrainer, resume=True, **resume_overrides)
    else:
        model = YOLO(config.pop("model"))
        model.train(trainer=ScaleAlignedDistillationTrainer, **config)


if __name__ == "__main__":
    main()

#模块一想法二：在Ultralytics的DetectionTrainer基础上，创建一个LQADetectionTrainer，集成LQALossWrapper和LQAAssignerWrapper，实现低质量感知的训练过程。通过配置LQA参数，可以灵活控制损失权重调整的方式和程度，同时保持原有的训练流程和模型架构不变。
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer


def _find_tensor_device(obj):
    """Find the first tensor device from nested predictions or batch."""
    if torch.is_tensor(obj):
        return obj.device
    if isinstance(obj, dict):
        for v in obj.values():
            d = _find_tensor_device(v)
            if d is not None:
                return d
    if isinstance(obj, (list, tuple)):
        for v in obj:
            d = _find_tensor_device(v)
            if d is not None:
                return d
    return None


def _move_tensors_in_object(obj, device):
    """Move tensor and device attributes inside Ultralytics loss objects."""
    if obj is None or not hasattr(obj, "__dict__"):
        return

    if hasattr(obj, "device"):
        try:
            obj.device = device
        except Exception:
            pass

    for k, v in list(vars(obj).items()):
        if torch.is_tensor(v):
            if v.device != device:
                setattr(obj, k, v.to(device))
        elif isinstance(v, torch.device):
            setattr(obj, k, device)
        elif isinstance(v, dict):
            new_v, changed = {}, False
            for kk, vv in v.items():
                if torch.is_tensor(vv) and vv.device != device:
                    new_v[kk], changed = vv.to(device), True
                elif isinstance(vv, torch.device):
                    new_v[kk], changed = device, True
                else:
                    new_v[kk] = vv
            if changed:
                setattr(obj, k, new_v)
        elif isinstance(v, (list, tuple)):
            new_items, changed = [], False
            for item in v:
                if torch.is_tensor(item) and item.device != device:
                    new_items.append(item.to(device)); changed = True
                elif isinstance(item, torch.device):
                    new_items.append(device); changed = True
                else:
                    new_items.append(item)
            if changed:
                try:
                    setattr(obj, k, type(v)(new_items))
                except Exception:
                    setattr(obj, k, new_items)
        elif hasattr(v, "__dict__"):
            if hasattr(v, "device"):
                try:
                    v.device = device
                except Exception:
                    pass
            for kk, vv in list(vars(v).items()):
                if torch.is_tensor(vv) and vv.device != device:
                    setattr(v, kk, vv.to(device))
                elif isinstance(vv, torch.device):
                    setattr(v, kk, device)


class LQAAssignerWrapper:
    """Object/assigned-anchor level LQA weighting wrapper for Ultralytics assigner.

    The base Ultralytics detection loss uses target_scores from the assigner for
    classification loss and for bbox/DFL foreground weighting. Instead of
    multiplying the whole batch loss by one scalar, this wrapper multiplies
    target_scores for foreground assigned anchors according to the quality of
    the assigned object/image: darkness and normalized object size.

    This keeps inference unchanged and applies LQA only during training.
    """

    def __init__(
        self,
        assigner,
        alpha: float = 0.25,
        max_weight: float = 1.25,
        warmup_epochs: int = 20,
        small_area_thr: float = 0.01,
        darkness_weight: float = 0.6,
        small_weight: float = 0.4,
        enabled: bool = True,
    ):
        self.assigner = assigner
        self.alpha = float(alpha)
        self.max_weight = float(max_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.small_area_thr = float(small_area_thr)
        self.darkness_weight = float(darkness_weight)
        self.small_weight = float(small_weight)
        self.enabled = bool(enabled)
        self.current_epoch = 0
        self.darkness: Optional[torch.Tensor] = None
        self.img_hw: Optional[Tuple[int, int]] = None

    def set_context(self, batch: Dict[str, torch.Tensor], epoch: int = 0):
        self.current_epoch = int(epoch)
        img = batch.get("img", None)
        if img is None or not torch.is_tensor(img):
            self.darkness = None
            self.img_hw = None
            return

        img = img.detach().float()
        if img.max() > 2.0:
            img = img / 255.0
        self.darkness = (1.0 - img.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)).detach()
        self.img_hw = (int(img.shape[2]), int(img.shape[3]))

    def _warmup(self, device):
        if self.warmup_epochs <= 0:
            return torch.tensor(1.0, device=device)
        v = min(1.0, float(self.current_epoch + 1) / float(self.warmup_epochs))
        return torch.tensor(v, device=device)

    def _weight_target_scores(self, outputs):
        if not self.enabled or self.darkness is None or self.img_hw is None:
            return outputs
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
            return outputs

        # Standard Ultralytics assigner output:
        # target_labels, target_bboxes, target_scores, fg_mask, [target_gt_idx]
        out = list(outputs)
        target_bboxes = out[1]
        target_scores = out[2]
        fg_mask = out[3]

        if not (torch.is_tensor(target_bboxes) and torch.is_tensor(target_scores) and torch.is_tensor(fg_mask)):
            return outputs
        if target_bboxes.numel() == 0 or target_scores.numel() == 0:
            return outputs
        if target_bboxes.ndim != 3 or target_bboxes.shape[-1] != 4:
            return outputs

        device = target_scores.device
        dtype = target_scores.dtype
        darkness = self.darkness.to(device=device, dtype=dtype).view(-1, 1)
        h, w = self.img_hw

        # target_bboxes are expected to be xyxy in image scale after assignment.
        bw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0) / max(float(w), 1.0)
        bh = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0) / max(float(h), 1.0)
        area = (bw * bh).clamp(0.0, 1.0)
        small_score = ((self.small_area_thr - area) / max(self.small_area_thr, 1e-9)).clamp(0.0, 1.0)

        q = (self.darkness_weight * darkness + self.small_weight * small_score).clamp(0.0, 1.0)
        raw_weight = torch.clamp(1.0 + self.alpha * q, min=1.0, max=self.max_weight)
        weight = 1.0 + self._warmup(device).to(dtype=dtype) * (raw_weight - 1.0)

        # Only foreground anchors should be reweighted. Background target_scores are zeros,
        # but explicit masking avoids accidental non-foreground modification.
        fg = fg_mask.to(device=device).bool()
        while fg.ndim < target_scores.ndim:
            fg = fg.unsqueeze(-1)
        weighted_scores = torch.where(fg, target_scores * weight.unsqueeze(-1), target_scores)
        out[2] = weighted_scores
        return type(outputs)(out) if isinstance(outputs, tuple) else out

    def __call__(self, *args, **kwargs):
        outputs = self.assigner(*args, **kwargs)
        return self._weight_target_scores(outputs)

    def __getattr__(self, name):
        # Delegate attributes such as topk, num_classes, etc.
        return getattr(self.assigner, name)


class LQALossWrapper:
    """Low-quality-aware loss wrapper without changing inference architecture."""

    def __init__(
        self,
        base_loss,
        alpha: float = 0.25,
        max_weight: float = 1.25,
        warmup_epochs: int = 20,
        small_area_thr: float = 0.01,
        darkness_weight: float = 0.6,
        small_weight: float = 0.4,
        enabled: bool = True,
        object_level: bool = False,
        batch_level_weight: bool = True,
    ):
        self.base_loss = base_loss
        self.alpha = float(alpha)
        self.max_weight = float(max_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.small_area_thr = float(small_area_thr)
        self.darkness_weight = float(darkness_weight)
        self.small_weight = float(small_weight)
        self.enabled = bool(enabled)
        self.object_level = bool(object_level)
        self.batch_level_weight = bool(batch_level_weight)
        self.current_epoch = 0

        if self.object_level and hasattr(self.base_loss, "assigner"):
            # Avoid double wrapping if a resumed/deepcopied model already contains wrapper.
            if not isinstance(self.base_loss.assigner, LQAAssignerWrapper):
                self.base_loss.assigner = LQAAssignerWrapper(
                    self.base_loss.assigner,
                    enabled=self.enabled,
                    alpha=self.alpha,
                    max_weight=self.max_weight,
                    warmup_epochs=self.warmup_epochs,
                    small_area_thr=self.small_area_thr,
                    darkness_weight=self.darkness_weight,
                    small_weight=self.small_weight,
                )

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _get_warmup(self, device):
        if self.warmup_epochs <= 0:
            return torch.tensor(1.0, device=device)
        warmup = min(1.0, float(self.current_epoch + 1) / float(self.warmup_epochs))
        return torch.tensor(warmup, device=device)

    def _compute_batch_quality_weight(self, batch: Dict[str, torch.Tensor], device):
        if not self.enabled:
            return torch.tensor(1.0, device=device)
        img = batch.get("img", None)
        if img is None:
            return torch.tensor(1.0, device=device)
        img = img.float()
        if img.max() > 2.0:
            img = img / 255.0
        darkness = 1.0 - img.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        batch_size = img.shape[0]
        bboxes = batch.get("bboxes", None)
        batch_idx = batch.get("batch_idx", None)
        if bboxes is not None and batch_idx is not None and bboxes.numel() > 0:
            bboxes = bboxes.float()
            batch_idx = batch_idx.view(-1).long().clamp(0, batch_size - 1)
            area = (bboxes[:, 2] * bboxes[:, 3]).clamp(min=0.0, max=1.0)
            small_score = ((self.small_area_thr - area) / max(self.small_area_thr, 1e-9)).clamp(0.0, 1.0)
            obj_darkness = darkness[batch_idx]
            q_obj = (self.darkness_weight * obj_darkness + self.small_weight * small_score).clamp(0.0, 1.0)
            q = q_obj.mean()
        else:
            q = darkness.mean().clamp(0.0, 1.0)
        raw_weight = torch.clamp(1.0 + self.alpha * q, min=1.0, max=self.max_weight)
        warmup = self._get_warmup(device)
        return (1.0 + warmup * (raw_weight - 1.0)).to(device)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]):
        device = _find_tensor_device(preds) or _find_tensor_device(batch) or torch.device("cpu")
        _move_tensors_in_object(self.base_loss, device)

        # For object-level LQA, pass current batch context to assigner wrapper before base loss call.
        if self.object_level and hasattr(self.base_loss, "assigner") and isinstance(self.base_loss.assigner, LQAAssignerWrapper):
            self.base_loss.assigner.set_context(batch, epoch=self.current_epoch)

        out = self.base_loss(preds, batch)
        if not isinstance(out, tuple) or len(out) != 2:
            return out

        loss, loss_items = out

        # In object-level mode, default is no extra batch scalar. Object weighting has already
        # been injected through target_scores. For old batch-level experiments, keep previous behavior.
        if not self.batch_level_weight:
            return loss, loss_items

        weight = self._compute_batch_quality_weight(batch, device=device)
        weighted_loss = loss * weight
        if torch.is_tensor(loss_items):
            weighted_items = loss_items * weight.detach()
        elif isinstance(loss_items, dict):
            weighted_items = {k: (v * weight.detach() if torch.is_tensor(v) else v) for k, v in loss_items.items()}
        else:
            weighted_items = loss_items
        return weighted_loss, weighted_items


class LQADetectionTrainer(DetectionTrainer):
    """DetectionTrainer with low-quality-aware loss wrapper."""

    lqa_cfg: Dict[str, Any] = {}

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if not hasattr(model, "args"):
            model.args = self.args

        if hasattr(model, "init_criterion"):
            base_loss = model.init_criterion()
        elif hasattr(model, "criterion"):
            base_loss = model.criterion
        else:
            raise RuntimeError("Cannot find model criterion/init_criterion for LQA loss wrapping.")

        object_level = bool(self.lqa_cfg.get("object_level", False))
        # Default: old experiments keep batch scalar; object-level experiments use assigner weighting only.
        batch_level_weight = bool(self.lqa_cfg.get("batch_level_weight", not object_level))

        model.criterion = LQALossWrapper(
            base_loss=base_loss,
            enabled=self.lqa_cfg.get("enabled", True),
            alpha=self.lqa_cfg.get("alpha", 0.25),
            max_weight=self.lqa_cfg.get("max_weight", 1.25),
            warmup_epochs=self.lqa_cfg.get("warmup_epochs", 20),
            small_area_thr=self.lqa_cfg.get("small_area_thr", 0.01),
            darkness_weight=self.lqa_cfg.get("darkness_weight", 0.6),
            small_weight=self.lqa_cfg.get("small_weight", 0.4),
            object_level=object_level,
            batch_level_weight=batch_level_weight,
        )
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if hasattr(self, "model") and hasattr(self.model, "criterion"):
            criterion = self.model.criterion
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(getattr(self, "epoch", 0))
        return batch


def build_lqa_trainer(lqa_cfg: Dict[str, Any]):
    """Create an isolated trainer class with given LQA config."""
    class ConfiguredLQADetectionTrainer(LQADetectionTrainer):
        pass
    ConfiguredLQADetectionTrainer.lqa_cfg = dict(lqa_cfg or {})
    return ConfiguredLQADetectionTrainer

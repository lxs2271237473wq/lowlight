from __future__ import annotations

from typing import Any, Dict

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
    ):
        self.base_loss = base_loss
        self.alpha = float(alpha)
        self.max_weight = float(max_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.small_area_thr = float(small_area_thr)
        self.darkness_weight = float(darkness_weight)
        self.small_weight = float(small_weight)
        self.enabled = bool(enabled)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _get_warmup(self, device):
        if self.warmup_epochs <= 0:
            return torch.tensor(1.0, device=device)
        warmup = min(1.0, float(self.current_epoch + 1) / float(self.warmup_epochs))
        return torch.tensor(warmup, device=device)

    def _compute_quality_weight(self, batch: Dict[str, torch.Tensor], device):
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

        out = self.base_loss(preds, batch)
        if not isinstance(out, tuple) or len(out) != 2:
            return out

        loss, loss_items = out
        weight = self._compute_quality_weight(batch, device=device)
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

        model.criterion = LQALossWrapper(
            base_loss=base_loss,
            enabled=self.lqa_cfg.get("enabled", True),
            alpha=self.lqa_cfg.get("alpha", 0.25),
            max_weight=self.lqa_cfg.get("max_weight", 1.25),
            warmup_epochs=self.lqa_cfg.get("warmup_epochs", 20),
            small_area_thr=self.lqa_cfg.get("small_area_thr", 0.01),
            darkness_weight=self.lqa_cfg.get("darkness_weight", 0.6),
            small_weight=self.lqa_cfg.get("small_weight", 0.4),
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

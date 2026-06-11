#检查本地 Ultralytics loss 接口
from pathlib import Path
import inspect
import ultralytics

print("=" * 80)
print("Ultralytics version")
print("=" * 80)
print(ultralytics.__version__)

print("\n" + "=" * 80)
print("DetectionTrainer")
print("=" * 80)

try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    print("DetectionTrainer module:", DetectionTrainer.__module__)
    print("DetectionTrainer file:", inspect.getfile(DetectionTrainer))
    print("DetectionTrainer methods:")
    for name in ["get_model", "get_validator", "build_dataset", "get_dataloader", "preprocess_batch", "criterion"]:
        if hasattr(DetectionTrainer, name):
            obj = getattr(DetectionTrainer, name)
            print(f"- {name}: {inspect.signature(obj)}")
except Exception as e:
    print("DetectionTrainer inspect failed:", repr(e))

print("\n" + "=" * 80)
print("Loss classes")
print("=" * 80)

try:
    import ultralytics.utils.loss as loss_mod
    print("loss.py file:", inspect.getfile(loss_mod))

    for cls_name in [
        "v8DetectionLoss",
        "v8SegmentationLoss",
        "E2EDetectLoss",
        "BboxLoss",
    ]:
        if hasattr(loss_mod, cls_name):
            cls = getattr(loss_mod, cls_name)
            print(f"\n[{cls_name}]")
            print("signature:", inspect.signature(cls))
            if hasattr(cls, "__call__"):
                print("__call__:", inspect.signature(cls.__call__))
            if hasattr(cls, "forward"):
                print("forward:", inspect.signature(cls.forward))
except Exception as e:
    print("loss inspect failed:", repr(e))

print("\n" + "=" * 80)
print("Model loss attributes")
print("=" * 80)

try:
    from ultralytics import YOLO
    model = YOLO("weights/pretrained/yolo11s.pt")
    print("YOLO model type:", type(model.model))
    attrs = [a for a in dir(model.model) if "loss" in a.lower() or "criterion" in a.lower()]
    print("loss-related attrs:", attrs)
except Exception as e:
    print("model inspect failed:", repr(e))

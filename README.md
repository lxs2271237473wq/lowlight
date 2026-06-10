# Low-light Object Detection

This repository has been reset for the new experimental design.

## Directory rule

Keep the project flat and concise.

```text
lowlight-object-detection/
├── train.py
├── val.py
├── train_config.yaml
├── val_config.yaml
├── baseline_summary.csv
├── requirements.txt
├── tool/
└── runs/
    └── baselines/
        └── baselines_summary.csv
```

## Important rules

- Do not upload datasets or weights to GitHub.
- Do not create empty future module directories.
- Create a module directory only when that module starts.
- Each module keeps its own summary table.
- The root `baseline_summary.csv` records baselines and the best result from each module.
- Formal experiments should use at least 100 epochs.

## Start training

Edit `train_config.yaml`, then run:

```bash
python train.py
```

## Validate

Edit `val_config.yaml`, then run:

```bash
python val.py
```

## Create a new module only when needed

```bash
python -m tool.new_module module1_lqa_loss
```

This creates:

```text
runs/module1_lqa_loss/module1_lqa_loss_summary.csv
```

## Quality diagnosis example

```bash
python -m tool.quality_diagnosis \
  --data datasets_yolo/ExDark_YOLO/exdark.yaml \
  --split val \
  --out runs/baselines/exdark_quality \
  --name exdark
```

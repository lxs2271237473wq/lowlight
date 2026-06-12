# AODRaw YOLO11s LQA Loss Warmup Diagnosis

## Experiment

- Experiment name: aodraw_yolo11s_lqa_loss_warmup
- Dataset: AODRaw sRGB
- Detector: YOLO11s
- Module: Module1 - Batch-level LQA Loss Warmup
- Epochs: 100

## Overall metric comparison

| Metric | AODRaw YOLO11s Baseline | LQA Loss Warmup | Change |
|---|---:|---:|---:|
| best_mAP50 | 0.42416 | 0.42338 | -0.00078 |
| best_mAP50-95 | 0.28529 | 0.28343 | -0.00186 |

## Group recall diagnosis

| Group | LQA Loss Warmup Recall@IoU50 |
|---|---:|
| large | 0.896582 |
| medium | 0.883903 |
| small | 0.688158 |
| small-dark | 0.688281 |

## Small objects under illumination bins

| Group | Recall@IoU50 |
|---|---:|
| easy_light_small | 0.683930 |
| extreme_dark_small | 0.649840 |
| hard_light_small | 0.713753 |
| medium_light_small | 0.692655 |

## Conclusion

Batch-level LQA Loss Warmup does not improve AODRaw. Compared with the YOLO11s baseline, both overall mAP50-95 and small/small-dark recall decrease slightly.

This indicates that simple batch-level loss reweighting is not sufficiently robust for AODRaw, where the small-object distribution is much harder than ExDark. Further improvement should not rely on scalar loss reweighting alone. The next module should introduce a stronger mechanism, such as feature-level low-light enhancement, illumination-aware attention, or teacher-student supervision.

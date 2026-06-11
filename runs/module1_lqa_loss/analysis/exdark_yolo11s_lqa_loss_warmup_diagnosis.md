# ExDark YOLO11s LQA Loss Warmup Diagnosis

## Experiment

- Experiment name: exdark_yolo11s_lqa_loss_warmup
- Dataset: ExDark
- Detector: YOLO11s
- Module: Module1 - Low Quality Aware Loss Warmup
- Epochs: 100
- Purpose: verify whether low-quality-aware loss improves the main baseline failure groups, especially small and small-dark objects.

## Overall Metric Comparison

| Metric | YOLO11s Baseline | LQA Loss Warmup | Change |
|---|---:|---:|---:|
| best_mAP50 | 0.78393 | 0.78147 | -0.00246 |
| best_mAP50-95 | 0.50075 | 0.50299 | +0.00224 |
| best_precision | 0.81075 | 0.79263 | -0.01812 |
| best_recall | 0.71119 | 0.71190 | +0.00071 |

## Group Recall Comparison

| Group | YOLO11s Baseline Recall | LQA Loss Warmup Recall | Change |
|---|---:|---:|---:|
| large | 0.9538 | 0.9625 | +0.0087 |
| medium | 0.9336 | 0.9305 | -0.0031 |
| small | 0.8584 | 0.8706 | +0.0122 |
| small-dark | 0.8549 | 0.8709 | +0.0160 |

## Detailed LQA Loss Warmup Diagnosis

### Scale groups

| Scale | Objects | Matched | Missed | Recall@IoU50 |
|---|---:|---:|---:|---:|
| large | 2186 | 2104 | 82 | 0.962489 |
| medium | 1641 | 1527 | 114 | 0.930530 |
| small | 1144 | 996 | 148 | 0.870629 |

### Small-dark flag

| is_small_dark | Objects | Matched | Missed | Recall@IoU50 |
|---|---:|---:|---:|---:|
| 0 | 3910 | 3703 | 207 | 0.947059 |
| 1 | 1061 | 924 | 137 | 0.870877 |

### Small objects under different illumination bins

| Group | Objects | Matched | Missed | Recall@IoU50 |
|---|---:|---:|---:|---:|
| easy_light_small | 5 | 5 | 0 | 1.000000 |
| extreme_dark_small | 756 | 654 | 102 | 0.865079 |
| hard_light_small | 305 | 270 | 35 | 0.885246 |
| medium_light_small | 78 | 67 | 11 | 0.858974 |

## Conclusion

The previous repeat-based LQA sampling strategy degraded small and small-dark recall, indicating that directly changing the sample distribution is unstable.

In contrast, LQA Loss Warmup improves the main failure groups identified in Stage 0. Small-object recall increases from 0.8584 to 0.8706, and small-dark recall increases from 0.8549 to 0.8709. This shows that the loss-level low-quality-aware strategy is directionally effective.

However, the overall gain is still weak. best_mAP50-95 only increases by 0.00224, while precision decreases by 0.01812. Therefore, this version should be kept as a positive but preliminary module result. The next step should be to strengthen the loss design or tune the weighting strategy, instead of expanding this version directly to all datasets.

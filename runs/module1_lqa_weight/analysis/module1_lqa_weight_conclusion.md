# Module1 LQA Weight Conclusion

## Experiment

- Experiment name: exdark_yolo11s_lqa_weight
- Baseline: exdark_yolo11s_baseline
- Method: repeat-based LQA sampling for small-dark images
- Dataset: ExDark
- Model: YOLO11s
- Epochs: 100

## Result Summary

The repeat-based LQA sampling strategy failed to improve performance.

Compared with ExDark + YOLO11s baseline:

| Metric | Baseline | LQA Weight | Change |
|---|---:|---:|---:|
| mAP50-95 | 0.50075 | 0.48030 | -0.02045 |
| small recall | 0.8584 | 0.7928 | -0.0656 |
| small-dark recall | 0.8549 | 0.7917 | -0.0631 |

## Diagnosis

The method reduced recall across object scales, especially for small and small-dark objects.

This suggests that image-level repeat sampling disturbed the original training distribution and did not provide useful low-light robustness. Repeating difficult low-quality samples may amplify noisy, blurry, or ambiguous object features instead of improving the detector's representation.

## Decision

This version should be marked as a failed Module1 attempt.

It should not be extended to AODRaw, YOLOv8s, cross-dataset validation, or later module combinations.

## Next Step

Replace repeat-based sampling with loss-level low-quality-aware weighting.

Recommended next experiment:

module1_lqa_loss

exdark_yolo11s_lqa_loss_warmup

The next version should avoid changing the image sampling distribution. Instead, it should apply mild object-level loss weighting with warmup and clipping.

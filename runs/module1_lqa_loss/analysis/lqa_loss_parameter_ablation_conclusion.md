# LQA Loss Parameter Ablation Conclusion

## Main finding

The original LQA Loss Warmup setting remains the best Module1 candidate for low-light small-object robustness.

Although some parameter variants improve overall mAP50-95, they fail to preserve the intended small-dark object recall improvement. Therefore, the main selection criterion should not be overall mAP alone, but whether the module improves the diagnosed failure groups: small and small-dark objects.

## Experiments

| Experiment | best_mAP50 | best_mAP50-95 | Small Recall | Small-Dark Recall | Conclusion |
|---|---:|---:|---:|---:|---|
| exdark_yolo11s_lqa_loss_warmup | 0.78147 | 0.50299 | 0.8706 | 0.8709 | Main Module1 result |
| exdark_yolo11s_lqa_loss_small_focus_a030 | 0.78036 | 0.50133 | 0.8654 | 0.8633 | Small-object overweighting is ineffective |
| exdark_yolo11s_lqa_loss_mild_a020 | 0.78515 | 0.50378 | 0.8558 | 0.8511 | Better overall mAP but weak small-dark robustness |
| exdark_yolo11s_lqa_loss_mid_a023 | 0.77616 | 0.50095 | - | - | Eliminated by overall metrics |
| exdark_yolo11s_lqa_loss_dark_focus_a025 | 0.78012 | 0.50342 | 0.8365 | 0.8332 | High precision but poor hard-object recall |

## Interpretation

The original warmup setting improves small recall from 0.8584 to 0.8706 and small-dark recall from 0.8549 to 0.8709. This directly addresses the Stage 0 diagnosis, where small and small-dark objects were the main failure groups.

The small-focus variant increases the small-object weight, but it does not improve small-dark recall. This suggests that simply increasing the small-object term over-emphasizes scale difficulty and does not solve illumination-related feature degradation.

The mild variant improves best mAP50-95, but its small and small-dark recall drop below the baseline. It should be treated as a precision-oriented trade-off, not as a low-light robustness improvement.

The dark-focus variant also improves precision and keeps mAP50-95 above the original warmup, but its small-dark recall drops sharply. This indicates that over-emphasizing image-level darkness can suppress difficult small-object recall.

## Final decision

Use `exdark_yolo11s_lqa_loss_warmup` as the current Module1 main result.

Keep `mild_a020` and `dark_focus_a025` as trade-off ablation evidence, but do not use them as the main low-light small-object robustness result.

Further gains should not rely on simple scalar reweighting. The next step should introduce a stronger mechanism, such as object-level LQA weighting, feature-level auxiliary supervision, or teacher-student distillation.

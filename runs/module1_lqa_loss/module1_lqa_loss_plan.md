# Module1 LQA Loss Plan

## Motivation

The previous repeat-based LQA sampling experiment failed.

Failed experiment:

- module: module1_lqa_weight
- experiment: exdark_yolo11s_lqa_weight
- method: repeat-based sampling for small-dark images

Failure diagnosis:

- mAP50-95 decreased from 0.50075 to 0.48030.
- small recall decreased from 0.8584 to 0.7928.
- small-dark recall decreased from 0.8549 to 0.7917.

This suggests that changing the image sampling distribution is unstable.

## New Direction

Module1 LQA Loss should avoid image repeat sampling.

The new strategy is:

- keep the original dataset distribution unchanged;
- apply mild object-level low-quality-aware loss weighting;
- use warmup to avoid early-stage instability;
- clip the maximum weight to avoid over-emphasizing noisy samples.

## First Target Experiment

Experiment name:

exdark_yolo11s_lqa_loss_warmup

Baseline:

exdark_yolo11s_baseline

Dataset:

ExDark

Model:

YOLO11s

Epochs:

100

## Decision Rule

Continue this direction only if at least one of the following holds:

1. mAP50-95 improves over baseline.
2. mAP50-95 is not worse, and small-dark recall improves.
3. small-dark recall improves clearly, with only minor mAP drop.

Reject this direction if both mAP50-95 and small-dark recall decrease.

# Model Selection Report

Generated: 2026-10-15T10:29:40.627842+00:00

## Selected Model

- **Experiment:** E02_optimized
- **Validation mAP50-95:** 0.4205
- **Validation mAP50:** 0.7472
- **Precision:** 0.7471
- **Recall:** 0.7153
- **Model size:** 6.2 MB
- **Checkpoint:** D:\Fabric defect detection\J26-DS-357-smart-apparel-lk\runs\component1_training\E02_optimized/weights/best.pt

## Selection Criterion

Primary: highest validation mAP50-95

## All Candidates

| Rank | Experiment | mAP50-95 | mAP50 |
|---|---|---|---|
| 1 | E02_optimized | 0.4205 | 0.7472 | **←**
| 2 | E01_baseline | 0.3757 | 0.7108 |
| 3 | E03_high_resolution | 0.3071 | 0.639 |

## Next Steps

1. Review per-class metrics for the selected model.
2. Freeze this selection before test evaluation.
3. Run locked test evaluation with --authorize-test.
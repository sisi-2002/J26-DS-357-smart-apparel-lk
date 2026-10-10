# Model Freeze Manifest — E02_optimized (v1)

**Freeze ID:** F1.1_model_selection_freeze_v1
**Created:** 2026-10-15
**Milestone:** Function 1 model-selection milestone (NOT full Function 1 completion)
**Machine-readable manifest:** [model_freeze_E02_optimized_v1.json](model_freeze_E02_optimized_v1.json)

## Selected Detector

| Field | Value |
|---|---|
| Experiment | E02_optimized |
| Checkpoint path | `runs/component1_training/E02_optimized/weights/best.pt` |
| SHA256 | `c578003ca158a892ac90de3db7b7552aefabd5292e311c6dd0c76faf717628d3` |
| Size | 6,503,722 bytes (6.20 MB) |
| Architecture | YOLOv8n |

## Comparison Table

| Experiment | ImgSz | mAP50 | mAP50-95 | Precision | Recall | Size (MB) |
|---|---|---|---|---|---|---|
| E01_baseline | 640 | 0.7108 | 0.3757 | 0.7192 | 0.6613 | 6.20 |
| **E02_optimized** | **640** | **0.7472** | **0.4205** | **0.7471** | **0.7153** | **6.20** |
| E03_high_resolution | 704 | 0.6390 | 0.3071 | 0.6772 | 0.6157 | 6.20 |

E02 wins on every metric and every per-class AP50 — no tie-breaking required.

## Per-Class AP50

| Class | E01 | E02 | E03 |
|---|---|---|---|
| baekra | 0.7056 | **0.7596** | 0.6578 |
| color issues | **0.7489** | 0.7324 | 0.7296 |
| contamination | 0.9078 | **0.9599** | 0.8151 |
| cut | 0.6508 | **0.7106** | 0.5472 |
| gray stitch | 0.5920 | **0.6026** | 0.5189 |
| selvet | 0.5100 | **0.6012** | 0.3881 |
| stain | 0.8606 | **0.8644** | 0.8162 |

E02 is highest on 6 of 7 classes (color issues is marginally E01's best by 0.0165, within noise).

## Dataset Consistency

- All three experiments trained/validated against identical `chenab_detection_training_v1` (same `data_yaml_sha256`, same `split_manifest.csv`).
- Same class names and order across all three.
- 377 valid images, 1900 train images, 378 test images (locked, untouched).
- Verified zero source-family group overlap between train/valid/test.

## Resource Constraints

- Model size: identical (6.20 MB) across all candidates — not discriminating.
- No documented hard resource constraint beyond the architecture plan's CPU-latency risk note (benchmark pending, not yet measured for any candidate).
- Inference latency/FPS/memory: **not available** — Phase 5.2 benchmarking has not been run.

## Decision

**SELECTED: E02_optimized** as the final research detector for Component 1 Function 1, based on consistent, leakage-verified validation evidence. See JSON manifest for full rationale, limitations, and unverified assumptions.

## What This Freeze Does NOT Do

- Does not evaluate the locked test split.
- Does not benchmark inference latency/FPS/memory.
- Does not modify, rename, or move the original checkpoint.
- Does not start backend/frontend integration.
- Does not run the A1–A6 complexity-based XAI evaluation pipeline against this checkpoint (separate, subsequent task).

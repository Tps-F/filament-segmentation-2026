# Filament segmentation 2026

Reproducible inference for our accepted Kaggle submission **58287816**, public score **0.40** on 13 November 2026. This repository documents the selected own-model solution.

The model combines our YOLO26s detector, an equal EfficientNet-B5/ConvNeXt-Tiny semantic ensemble, a frozen instance-quality regressor, and confidence-ordered pixel ownership. All task training used the official competition training subset. Generic COCO/ImageNet pretraining is documented in the provenance files.

## Reproduce the accepted submission

Use Python 3.12 and a CUDA GPU with native BF16 support. The reference host is an NVIDIA GB10. The recorded CUDA 13.0 dependencies are pinned in `requirements.txt`; another machine or fresh installation has not been verified for this own-model release. T4/FP16 substitution is not supported by this numerical recipe.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_release test_reference_mode
.venv/bin/python run.py --execute --reference-mode report-only \
  --test-dir /path/to/official/test/test_images \
  --output /path/to/new/output-directory
```

The notebook in `notebooks/` provides the same path with configurable locations and an explicit `RUN_FULL` switch. Download the official competition data through your own authorized Kaggle access. The repository contains the trained inference weights and quality model in checksummed binary pieces; the runner restores them locally. Training and annotation files are unnecessary for inference.

Version **3.2** keeps the exact trained weights, numerical sources, precision and inference recipe. It adds an explicit `--reference-mode report-only` option: all canonical output hashes are still compared and recorded, while valid inference can finish when another compatible GPU produces different bytes. `--reference-mode strict` stops at a reference mismatch and is the CLI default. Neither mode relaxes input/weight/source integrity, native shapes, finite outputs, source identity, mask support, or zero overlap. Inspect `reference_comparison.json` and `reproduction_receipt.json` to see actual equality; a completed report-only run does not imply the submitted CSV was reproduced.

The exact fully exercised **v3.1** package is retained at the `v3.1-verified` Git tag. The v3.2 derivation has passed 19 CPU contracts; independent checks also accepted all 180 actual detector files and 540 saved probability arrays under its new validity guards. Its guard/report changes do not constitute another full GPU run. Deterministic settings do not establish byte equality across different GPUs or software stacks.

## Verified result

A fresh local Git clone of **v3.1** ran all three shipped notebook code cells as plain Python in an existing separate environment on 13 November. It regenerated all 180 images and the accepted CSV exactly in **2,486.81 seconds** of inference and export. This was not a new-machine installation or an actual Jupyter-kernel test.

- 1,219 instances, 179 nonempty images, zero overlapping output pixels.
- CSV SHA-256: `9dd513669eee8b12910e001732eea9fdd0f8a8adeb9eacab4421a544da5d9c3c`.
- Public score: **0.40**. No private score is available.
- Reused fold-0 development PQ: **0.4564809747563001** after the same disjoint export. Fold 0 was repeatedly reused for development; this local score is not an unbiased estimate of final performance.

Fixed complete-hybrid comparisons on separately trained folds 1 and 2 gave PQ **0.469083** and **0.444824**. Pooling their existing counts over 283 images and 121 date groups gives hybrid PQ **0.455972**, versus **0.445493** for the semantic pair: difference **+0.010480**, conditional 95% date-bootstrap interval **[0.001784, 0.018965]**. Each individual fold's difference interval includes zero. These comparisons condition on fixed predictions, exclude training and experiment-selection variability, and do not establish an untouched evaluation after repeated recipe screening. The scores belong to fold-specific models, not the all-data deployment checkpoints.

An independent CPU audit verified all 540 probability caches, native mask geometry, exact candidate bytes, and process exits. Receipts are in `verification/`. The source and weight inventory is `bundle_preparation.json`. Historical records under `docs/` describe earlier versions and preserved failed checks; the current result above supersedes their status snapshots.

## Scope and attribution

Inference uses fixed precision, thresholds and geometry, and never fits on test predictions. The runner does not upload a submission. Competition images, labels, credentials and cached probability maps are excluded.

## Technical report and participant

The [four-page technical report](docs/report/own_hybrid_submission_20261115.pdf), dated 15 November 2026, documents submission **58287816** and the validated own-model recipe. Its [LaTeX source](docs/report/own_hybrid_submission_20261115.tex) and supporting files are in `docs/report/`. The report preserves the distinction between reused development data, fold-specific confirmation, and the accepted all-data public result.

**Akito Okamoto** — Kaggle handle **ftps15**; Doshisha University, Japan. Contact: **ftpsflandre@gmail.com**.

Prepared with **OpenAI Codex assistance** for implementation, experiment orchestration, verification, and report drafting under Akito Okamoto's direction. Git commit co-author trailers record Codex assistance.

See `THIRD_PARTY_NOTICES.txt` and `licenses/` for upstream terms and the exact Ultralytics 8.4.211 AGPL-3.0 license. The detector derives from Ultralytics YOLO26s; semantic encoders and their generic initialization are recorded in the model provenance. No blanket license grant for third-party assets is asserted.

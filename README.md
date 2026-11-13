# Filament segmentation 2026

Reproducible inference for our accepted Kaggle submission **58287816**, public score **0.40** on 13 November 2026. This is the current solution; work toward the competition deadline continues.

The model combines our YOLO26s detector, an equal EfficientNet-B5/ConvNeXt-Tiny semantic ensemble, a frozen instance-quality regressor, and confidence-ordered pixel ownership. All task training used the official competition training subset. Generic COCO/ImageNet pretraining is documented in the provenance files.

## Reproduce the accepted submission

Use Python 3.12 and a CUDA GPU with native BF16 support. The reference host is an NVIDIA GB10. The recorded CUDA 13.0 dependencies are pinned in `requirements.txt`; another machine or fresh installation has not been verified for this own-model release. T4/FP16 substitution is not supported by this numerical recipe.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_release
.venv/bin/python run.py --execute \
  --test-dir /path/to/official/test/test_images \
  --output /path/to/new/output-directory
```

The notebook in `notebooks/` provides the same path with configurable locations and an explicit `RUN_FULL` switch. Download the official competition data through your own authorized Kaggle access. The repository contains the trained inference weights and quality model in checksummed binary pieces; the runner restores them locally. Training and annotation files are unnecessary for inference.

Version 3.1 checks every detector output, every semantic member, their equal mean, and the final CSV against the reference hashes. It stops if any differs. Deterministic settings do not establish byte equality across different GPUs or software stacks.

## Verified result

A fresh local Git clone ran all three shipped notebook code cells as plain Python in an existing separate environment on 13 November. It regenerated all 180 images and the accepted CSV exactly in **2,486.81 seconds** of inference and export. This was not a new-machine installation or an actual Jupyter-kernel test.

- 1,219 instances, 179 nonempty images, zero overlapping output pixels.
- CSV SHA-256: `9dd513669eee8b12910e001732eea9fdd0f8a8adeb9eacab4421a544da5d9c3c`.
- Public score: **0.40**. No private score is available.
- Reused fold-0 development PQ: **0.4564809747563001** after the same disjoint export. Complete hybrid confirmation on independent folds is pending; this local score is not an unbiased estimate of final performance.

An independent CPU audit verified all 540 probability caches, native mask geometry, exact candidate bytes, and process exits. Receipts are in `verification/`. The source and weight inventory is `bundle_preparation.json`. Historical records under `docs/` describe earlier versions and preserved failed checks; the current result above supersedes their status snapshots.

## Scope and attribution

Inference uses fixed precision, thresholds and geometry, and never fits on test predictions. The runner does not upload a submission. Competition images, labels, credentials and cached probability maps are excluded. Final report and participant metadata will be added before the organizer deadline.

See `THIRD_PARTY_NOTICES.txt` and `licenses/` for upstream terms and the exact Ultralytics 8.4.211 AGPL-3.0 license. The detector derives from Ultralytics YOLO26s; semantic encoders and their generic initialization are recorded in the model provenance. No blanket license grant for third-party assets is asserted.

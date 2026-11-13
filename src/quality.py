"""Cross-fitted instance-quality filtering using image and mask evidence.

Every reported validation prediction comes from a quality model fitted without
that observation's date group. The base segmentation model must also hold out
the selected fold. Targets use the organizer's matching rule, not semantic Dice.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import joblib
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .metrics import PQStats, load_ground_truth
from .postprocess import PostprocessConfig, probabilities_to_labels, labels_to_rles


FEATURES = ["log_area", "log_width", "log_height", "aspect", "extent", "eccentricity",
            "compactness", "solidity", "radial_position", "prob_mean", "prob_std", "prob_q10",
            "prob_q50", "prob_q90", "prob_max", "group_mean", "gray_mean", "gray_std",
            "gray_q10", "gray_q90", "background_mean", "background_std", "dark_contrast",
            "contrast_snr", "edge_fraction", "components"]


def instance_features(probability, image, config):
    labels = probabilities_to_labels(probability, config)
    rows = []
    for label, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            raise ValueError("Expected contiguous instance IDs")
        y0, y1, x0, x1 = bounds[0].start, bounds[0].stop, bounds[1].start, bounds[1].stop
        ys = slice(max(0, y0 - 12), min(image.shape[0], y1 + 12))
        xs = slice(max(0, x0 - 12), min(image.shape[1], x1 + 12))
        mask = (labels[ys, xs] == label).astype(np.uint8)
        yy, xx = np.nonzero(mask)
        area = len(xx)
        p = probability[0, ys, xs][mask.astype(bool)]
        gray = image[ys, xs][mask.astype(bool)].astype(np.float32) / 255
        ring = cv2.dilate(mask, np.ones((17, 17), np.uint8)).astype(bool) & (labels[ys, xs] == 0)
        background = image[ys, xs][ring].astype(np.float32) / 255
        bm, bs = (float(background.mean()), float(background.std())) if len(background) else (float(gray.mean()), float(gray.std()))
        covariance = np.cov(np.stack([xx, yy])) if area > 1 else np.eye(2)
        eig = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
        eccentricity = float(np.sqrt(max(0, 1 - eig[0] / eig[1])))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        hull = cv2.convexHull(np.concatenate(contours))
        hull_area = max(float(cv2.contourArea(hull)), float(area))
        cx, cy = float(xx.mean() + xs.start), float(yy.mean() + ys.start)
        width, height = x1 - x0, y1 - y0
        gm = float(probability[1, ys, xs][mask.astype(bool)].mean()) if len(probability) > 1 else float(p.mean())
        dark = bm - float(gray.mean())
        boundary = mask & ~cv2.erode(mask, np.ones((3, 3), np.uint8))
        row = [np.log1p(area), np.log1p(width), np.log1p(height), max(width, height) / max(1, min(width, height)),
               area / (width * height), eccentricity, 4 * np.pi * area / max(1, perimeter**2), area / hull_area,
               np.hypot(cx - 1024, cy - 1024) / 1024, float(p.mean()), float(p.std()), *np.quantile(p, [.1, .5, .9]),
               float(p.max()), gm, float(gray.mean()), float(gray.std()), *np.quantile(gray, [.1, .9]),
               bm, bs, dark, dark / max(.01, bs), float(boundary.sum()) / area, len(contours)]
        rows.append(row)
    return labels, np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES))


def candidate_iou(records, labels):
    flat = labels.ravel()
    areas = np.bincount(flat)[1:]
    matrices = []
    for _, rles in records:
        iou = np.zeros((len(rles), len(areas)), dtype=np.float32)
        for i, rle in enumerate(rles):
            indices = np.flatnonzero(mu.decode(rle))
            intersections = np.bincount(flat[indices], minlength=len(areas) + 1)[1:]
            union = len(indices) + areas - intersections
            iou[i] = intersections / np.maximum(union, 1)
        matrices.append(iou)
    return matrices


def extract_one(job):
    stem, prob_path, image_path, group, records, config = job
    cv2.setNumThreads(1)
    with np.load(prob_path) as z:
        probability = z["probability"].astype(np.float32)
    image = np.load(image_path)
    labels, features = instance_features(probability, image, config)
    matrices = candidate_iou(records, labels)
    target = np.zeros(len(features), dtype=np.float32)
    for matrix in matrices:
        target += np.where(matrix > .5, matrix, 0).sum(axis=0)
    target /= max(1, len(matrices))
    return {"image_id": stem, "group": group, "features": features, "target": target,
            "weight": len(matrices), "iou": matrices}


def score_selection(entries, scores, threshold):
    total = PQStats()
    offset = 0
    for entry in entries:
        keep = scores[offset:offset + len(entry["target"])] >= threshold
        offset += len(keep)
        for matrix in entry["iou"]:
            iou = matrix[:, keep]
            hit = iou > .5
            total += PQStats(float(iou[hit].sum()), int(hit.sum()),
                             int((~hit.any(axis=0)).sum()), int((~hit.any(axis=1)).sum()))
    return total


def new_regressor():
    return HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=8, min_samples_leaf=30,
                                        learning_rate=.05, l2_regularization=10, early_stopping=False,
                                        random_state=2026)


def fit(args):
    crossfit_reference = getattr(args, "crossfit_reference", None)
    fixed_threshold = getattr(args, "fixed_threshold", None)
    if fixed_threshold is not None and (not np.isfinite(fixed_threshold) or not 0 <= fixed_threshold <= 1):
        raise ValueError("Fixed quality threshold must be finite and within [0, 1]")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((Path(args.prob_dir) / "inference_config.json").read_text())
    if metadata["split"] != "val" or metadata["fold"] != args.fold:
        raise ValueError("Quality CV requires probability files from the selected held-out fold")
    records = [r for r in manifest["train"] if r["fold"] == args.fold]
    config = PostprocessConfig(threshold=args.pixel_threshold, min_area=args.min_area,
                              close_radius=args.close_radius, grouping_threshold=args.grouping_threshold)
    output = Path(args.output)
    if (crossfit_reference is not None or fixed_threshold is not None) and output.exists() and any(output.iterdir()):
        raise ValueError("Fixed comparison mode requires a fresh output directory")
    reference = None
    if crossfit_reference is not None:
        from .quality_reference import load_reference
        reference = load_reference(crossfit_reference, manifest_path, args.fold, records, metadata, args.prob_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "features.joblib"
    identity = {"config": config.to_dict(), "inference": metadata, "fold": args.fold,
                "annotation_sha256": manifest["annotation_sha256"]}
    if reference is not None:
        identity["crossfit_reference"] = reference
    if fixed_threshold is not None:
        identity["fixed_threshold"] = fixed_threshold
    if cache_path.exists():
        cache = joblib.load(cache_path)
        if cache["identity"] != identity:
            raise ValueError("Existing features have a different source; use a fresh output directory")
        entries = cache["entries"]
    else:
        annotation_path = Path(manifest["annotation_path"])
        if not annotation_path.is_absolute():
            annotation_path = manifest_path.parent / annotation_path
        truth = load_ground_truth(annotation_path)
        jobs = []
        for record in records:
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            jobs.append((record["image_id"], str(Path(args.prob_dir) / (record["image_id"] + ".npz")),
                         str(image_path), record["date_group"], truth[record["image_id"]], config))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            entries = []
            for i, entry in enumerate(pool.map(extract_one, jobs), start=1):
                entries.append(entry)
                if i % 20 == 0 or i == len(jobs):
                    print(json.dumps({"features_completed": i, "total": len(jobs)}), flush=True)
        joblib.dump({"identity": identity, "entries": entries}, cache_path)
    x = np.concatenate([e["features"] for e in entries])
    y = np.concatenate([e["target"] for e in entries])
    weights = np.concatenate([np.full(len(e["target"]), e["weight"]) for e in entries])
    groups = np.concatenate([np.full(len(e["target"]), e["group"]) for e in entries])
    oof = np.zeros(len(x), dtype=np.float32)
    splitter = GroupKFold(n_splits=5)
    if reference is None:
        splits = splitter.split(x, y, groups)
    else:
        from .quality_reference import reference_splits
        splits = reference_splits(entries, reference["group_folds"])
    with threadpool_limits(limits=4):
        for fold, (train, valid) in enumerate(splits):
            model = new_regressor()
            model.fit(x[train], y[train], sample_weight=weights[train])
            oof[valid] = np.clip(model.predict(x[valid]), 0, 1)
            print(json.dumps({"quality_fold": fold, "train_instances": len(train), "valid_instances": len(valid)}), flush=True)
        ranking = []
        thresholds = np.arange(.05, .401, .025) if fixed_threshold is None else [fixed_threshold]
        for threshold in thresholds:
            ranking.append({"quality_threshold": float(threshold), **score_selection(entries, oof, threshold).to_dict()})
        ranking.sort(key=lambda r: r["pq"], reverse=True)
        model = new_regressor().fit(x, y, sample_weight=weights)
    report = {"base_fold": args.fold, "images": len(entries), "instances": len(x), "features": FEATURES,
              "pool_config": config.to_dict(), "best": ranking[0], "ranking": ranking,
              "interpretation": "Base segmentation held out this fold; quality scores cross-fitted by date group. Threshold selected on aggregated OOF scores; confirm on a second base fold before final claims."}
    if reference is not None:
        report["crossfit_reference"] = reference
    if fixed_threshold is not None:
        report["fixed_threshold"] = fixed_threshold
        report["interpretation"] = ("Base segmentation held out this fold; quality scores cross-fitted by date group. "
            "Quality threshold fixed before evaluation; no cutoff grid. Confirm on a second base fold before final claims.")
    if reference is not None:
        report["interpretation"] += " Meta-fold assignments are frozen from the validated reference, independent of this model's candidate counts."
    model_path = output / "quality_model.joblib"
    temporary_model = output / "quality_model.joblib.tmp"
    joblib.dump({"model": model, "report": report, "identity": identity}, temporary_model)
    temporary_model.replace(model_path)
    np.save(output / "oof_quality.npy", oof)
    offset, oof_images = 0, {}
    for entry in entries:
        count = len(entry["features"])
        oof_images[entry["image_id"]] = oof[offset:offset + count].tolist()
        offset += count
    oof_record = {
        "fold": args.fold, "pool_config": config.to_dict(),
        "quality_threshold": ranking[0]["quality_threshold"],
        "quality_crossfit": ("5 fixed reference folds by observation-date group" if reference is not None
                             else "5-fold GroupKFold by observation-date group"),
        "images": oof_images,
    }
    if reference is not None:
        oof_record["crossfit_reference"] = reference
    (output / "oof_instances.json").write_text(json.dumps(oof_record))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def apply(args):
    from .inference import write_submission
    saved = joblib.load(args.model)
    config = PostprocessConfig(**saved["report"]["pool_config"])
    threshold = args.quality_threshold if args.quality_threshold is not None else saved["report"]["best"]["quality_threshold"]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    predictions = []
    with threadpool_limits(limits=4):
        for idx, entry in enumerate(manifest["test"], start=1):
            with np.load(Path(args.prob_dir) / (entry["image_id"] + ".npz")) as z:
                probability = z["probability"].astype(np.float32)
            image_path = Path(entry["image_path"])
            image = np.load(image_path if image_path.is_absolute() else manifest_path.parent / image_path)
            labels, features = instance_features(probability, image, config)
            if len(features):
                keep = saved["model"].predict(features) >= threshold
                mapping = np.zeros(len(keep) + 1, dtype=np.int32)
                mapping[1:][keep] = np.arange(1, keep.sum() + 1)
                labels = mapping[labels]
            predictions.append((entry["image_id"], labels_to_rles(labels)))
            print(json.dumps({"image": idx, "instances": len(predictions[-1][1])}), flush=True)
    report = write_submission(predictions, args.output, [e["image_id"] for e in manifest["test"]])
    with open(args.model, "rb") as handle:
        model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report.update({"quality_model": args.model, "quality_model_sha256": model_hash,
                   "quality_threshold": threshold, "pool_config": config.to_dict(),
                   "probability_source": str(args.prob_dir),
                   "inference": json.loads((Path(args.prob_dir) / "inference_config.json").read_text())})
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("fit")
    train.add_argument("--prob-dir", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--manifest", default="data/cache/manifest.json")
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--pixel-threshold", type=float, default=.35)
    train.add_argument("--min-area", type=int, default=60)
    train.add_argument("--close-radius", type=int, default=5)
    train.add_argument("--grouping-threshold", type=float, default=0)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--crossfit-reference", help="Verified same-base-fold quality directory whose image/date meta-folds are frozen")
    train.add_argument("--fixed-threshold", type=float, help="Evaluate only this quality cutoff instead of selecting from a grid")
    predict = subparsers.add_parser("apply")
    predict.add_argument("--model", required=True)
    predict.add_argument("--prob-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--manifest", default="data/cache/manifest.json")
    predict.add_argument("--quality-threshold", type=float)
    args = parser.parse_args()
    (fit if args.command == "fit" else apply)(args)


if __name__ == "__main__":
    main()

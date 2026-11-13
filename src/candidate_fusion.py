"""CPU-only, date-cross-fitted object fusion for two held-out base models.

This experiment accepts only the same verified validation fold from own R34
and own YOLO models. Public all-data models are deliberately not supported.
The joint regressor uses raw mask/image/probability/detector features, never
another calibrator's OOF predictions. Threshold selection is exploratory;
confirm a selected rule on another base fold before deploying it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import joblib
import numpy as np
from pycocotools import mask as mu
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .ensemble import atomic_json, atomic_write, checkpoint_provenance, digest_file
from .metrics import PQStats, score_rles
from .postprocess import PostprocessConfig, probabilities_to_labels, labels_to_rles
from .quality import FEATURES, new_regressor, score_selection
from .quality_pool import POOL_CONFIG, validate_entries
from .train_yolo import validate_provenance


EXTRA_FEATURES = ["source_yolo", "detector_score", "cross_iou_max",
                  "cross_fraction_self", "cross_fraction_other",
                  "cross_evidence", "cross_overlap_count"]
THRESHOLDS = np.arange(.05, .401, .025)


def rle_iou(first, second):
    if not first or not second:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    return np.asarray(mu.iou(first, second, [0] * len(second)), dtype=np.float64)


def annotation_rles(annotation):
    return [{"size": [2048, 2048], "counts": counts} for counts in annotation["rles"]]


def greedy_nms(scores, pair_iou, threshold=.5):
    """Stable greedy mask NMS; ties keep the earlier source/candidate."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    for index in order:
        if not keep or not np.any(pair_iou[index, keep] > threshold):
            keep.append(int(index))
    return np.asarray(keep, dtype=np.int64)


def arbitrary_features(rle, probability, image, labels):
    """Same 26 quality measurements for an independently overlapping mask."""
    x0, y0, width, height = map(int, mu.toBbox(rle))
    xs = slice(max(0, x0 - 12), min(image.shape[1], x0 + width + 12))
    ys = slice(max(0, y0 - 12), min(image.shape[0], y0 + height + 12))
    mask = mu.decode(rle)[ys, xs].astype(np.uint8)
    foreground = mask.astype(bool)
    yy, xx = np.nonzero(mask)
    area = len(xx)
    if not area:
        raise ValueError("Empty candidate mask")
    p = probability[0, ys, xs][foreground]
    gray = image[ys, xs][foreground].astype(np.float32) / 255
    ring = (cv2.dilate(mask, np.ones((17, 17), np.uint8)).astype(bool)
            & ~foreground & (labels[ys, xs] == 0))
    background = image[ys, xs][ring].astype(np.float32) / 255
    bm, bs = ((float(background.mean()), float(background.std())) if len(background)
              else (float(gray.mean()), float(gray.std())))
    covariance = np.cov(np.stack([xx, yy])) if area > 1 else np.eye(2)
    eig = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
    eccentricity = float(np.sqrt(max(0, 1 - eig[0] / eig[1])))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(np.concatenate(contours)))), float(area))
    cx, cy = float(xx.mean() + xs.start), float(yy.mean() + ys.start)
    gm = float(probability[1, ys, xs][foreground].mean()) if len(probability) > 1 else float(p.mean())
    dark = bm - float(gray.mean())
    boundary = mask & ~cv2.erode(mask, np.ones((3, 3), np.uint8))
    return np.asarray([
        np.log1p(area), np.log1p(width), np.log1p(height), max(width, height) / max(1, min(width, height)),
        area / (width * height), eccentricity, 4 * np.pi * area / max(1, perimeter**2), area / hull_area,
        np.hypot(cx - 1024, cy - 1024) / 1024, float(p.mean()), float(p.std()), *np.quantile(p, [.1, .5, .9]),
        float(p.max()), gm, float(gray.mean()), float(gray.std()), *np.quantile(gray, [.1, .9]),
        bm, bs, dark, dark / max(.01, bs), float(boundary.sum()) / area, len(contours)], dtype=np.float32)


def matrix_signature(matrix):
    return (matrix.shape, hashlib.sha256(np.asarray(matrix, dtype=np.float32).tobytes()).hexdigest())


def extract_one(job):
    record, base, prob_path, image_path, raw_path = job
    cv2.setNumThreads(1)
    with np.load(prob_path, allow_pickle=False) as archive:
        probability = archive["probability"].astype(np.float32)
    if probability.shape[-2:] != (2048, 2048) or not np.isfinite(probability).all():
        raise ValueError("Expected finite native 2048-square probability maps")
    image = np.load(image_path, allow_pickle=False)
    labels = probabilities_to_labels(probability, PostprocessConfig(**POOL_CONFIG))
    r34 = labels_to_rles(labels)
    if len(r34) != len(base["features"]):
        raise ValueError("Rebuilt R34 candidate count differs from cached quality features")
    if not np.allclose(mu.area(r34), np.expm1(base["features"][:, 0]), atol=.03, rtol=0):
        raise ValueError("Rebuilt R34 candidate areas differ")
    raw = json.loads(Path(raw_path).read_text())
    for item in raw:
        if item["rle"]["size"] != [2048, 2048] or not .03 - 1e-7 <= item["score"] <= 1:
            raise ValueError("Invalid native YOLO detection")
        if int(mu.area(item["rle"])) != item["area"]:
            raise ValueError("YOLO cached area does not match RLE")
    yolo = [item for item in raw if item["area"] >= 60]
    rles = r34 + [item["rle"] for item in yolo]
    nr, ny = len(r34), len(yolo)
    x = np.concatenate([base["features"], np.asarray([
        arbitrary_features(item["rle"], probability, image, labels) for item in yolo],
        dtype=np.float32).reshape(-1, len(FEATURES))])
    pair_iou = rle_iou(rles, rles)
    area = np.asarray(mu.area(rles), dtype=np.float64)
    cross = pair_iou[:nr, nr:]
    intersections = np.rint(cross * (area[:nr, None] + area[None, nr:]) / (1 + cross))
    extras = np.zeros((nr + ny, len(EXTRA_FEATURES)), dtype=np.float32)
    extras[nr:, 0] = 1
    extras[nr:, 1] = [item["score"] for item in yolo]
    for start, stop, other_start, overlap, inter in [
            (0, nr, nr, cross, intersections), (nr, nr + ny, 0, cross.T, intersections.T)]:
        if not overlap.shape[1]:
            continue
        for row in range(stop - start):
            best = int(np.argmax(overlap[row]))
            here, other = start + row, other_start + best
            if overlap[row, best] > 0:
                evidence = extras[other, 1] if other >= nr else x[other, FEATURES.index("prob_mean")]
                extras[here, 2:6] = [overlap[row, best], inter[row, best] / area[here],
                                     inter[row, best] / area[other], evidence]
            extras[here, 6] = np.count_nonzero(overlap[row] > 0)
    matrices = [rle_iou(rles, annotation_rles(ann)).T for ann in record["annotators"]]
    # Manifest and original quality cache may order annotators differently.
    if sorted(matrix_signature(m[:, :nr]) for m in matrices) != sorted(matrix_signature(m) for m in base["iou"]):
        raise ValueError("Rebuilt R34 masks do not reproduce every original per-annotator IoU matrix")
    target = sum(np.where(matrix > .5, matrix, 0).sum(axis=0) for matrix in matrices) / len(matrices)
    return {"image_id": record["image_id"], "group": record["date_group"],
            "features": np.concatenate([x, extras], axis=1), "target": target.astype(np.float32),
            "weight": len(matrices), "iou": matrices, "pair_iou": pair_iou,
            "r34_count": nr, "rles": rles}


def score_kept(entries, selections):
    total, per_image = PQStats(), {}
    multiple_hits = 0
    for entry, selected in zip(entries, selections):
        stat = PQStats()
        for matrix in entry["iou"]:
            iou = matrix[:, selected]
            hit = iou > .5
            stat += PQStats(float(iou[hit].sum()), int(hit.sum()),
                            int((~hit.any(axis=0)).sum()), int((~hit.any(axis=1)).sum()))
            multiple_hits += int((hit.sum(axis=1) > 1).sum())
        total += stat
        per_image[entry["image_id"]] = {**stat.to_dict(), "instances": len(selected),
            "yolo_instances": int((selected >= entry["r34_count"]).sum())}
    return {**total.to_dict(), "instances": sum(len(s) for s in selections),
            "gt_records_with_multiple_prediction_hits": multiple_hits}, per_image


def quality_selections(entries, scores, threshold):
    offset, selected = 0, []
    for entry in entries:
        count = len(entry["target"])
        local = scores[offset:offset + count]
        eligible = np.flatnonzero(local >= threshold)
        selected.append(eligible[greedy_nms(local[eligible], entry["pair_iou"][np.ix_(eligible, eligible)])])
        offset += count
    if offset != len(scores):
        raise ValueError("Score count differs from candidate pool")
    return selected


def original_group_folds(base_entries):
    x = np.concatenate([entry["features"] for entry in base_entries])
    y = np.concatenate([entry["target"] for entry in base_entries])
    groups = np.concatenate([np.full(len(entry["target"]), entry["group"]) for entry in base_entries])
    assignment = {}
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
        for group in np.unique(groups[valid]):
            assignment[str(group)] = fold
    missing = {str(e["group"]) for e in base_entries} - set(assignment)
    if missing:
        raise ValueError(f"Original R34 meta-folds omit candidate-free date groups: {sorted(missing)}")
    return assignment


def crossfit(entries, group_folds):
    x = np.concatenate([entry["features"] for entry in entries])
    y = np.concatenate([entry["target"] for entry in entries])
    weights = np.concatenate([np.full(len(entry["target"]), entry["weight"]) for entry in entries])
    folds = np.concatenate([np.full(len(entry["target"]), group_folds[str(entry["group"])]) for entry in entries])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite candidate features or targets")
    oof = np.full(len(x), np.nan, dtype=np.float32)
    with threadpool_limits(limits=4):
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            model = new_regressor().fit(x[train], y[train], sample_weight=weights[train])
            oof[valid] = np.clip(model.predict(x[valid]), 0, 1)
            print(json.dumps({"quality_fold": fold, "train": int(train.sum()), "valid": int(valid.sum())}), flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete cross-fitted predictions")
    return oof


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    quality_dir, prob_dir, yolo_dir = map(Path, [args.r34_quality, args.r34_prob, args.yolo_dir])
    cache = joblib.load(quality_dir / "features.joblib")
    base_entries, base_identity = cache["entries"], cache["identity"]
    report = json.loads((quality_dir / "report.json").read_text())
    original_oof = np.load(quality_dir / "oof_quality.npy", allow_pickle=False)
    records = {r["image_id"]: r for r in manifest["train"] if r["fold"] == args.fold}
    validate_entries(base_entries, original_oof, records, report)
    metadata = json.loads((prob_dir / "inference_config.json").read_text())
    if (metadata != base_identity["inference"] or metadata["split"] != "val" or metadata["fold"] != args.fold
            or base_identity["fold"] != args.fold or base_identity["config"] != POOL_CONFIG
            or report["features"] != FEATURES or base_identity["annotation_sha256"] != manifest["annotation_sha256"]):
        raise ValueError("R34 source fold/config/features/annotation provenance differs")
    if metadata.get("manifest_sha256", digest_file(manifest_path)) != digest_file(manifest_path):
        raise ValueError("R34 manifest differs")
    r34_provenance = checkpoint_provenance(metadata, prob_dir, manifest, {})
    yolo_config = json.loads((yolo_dir / "inference_config.json").read_text())
    yolo_report = json.loads((yolo_dir / "run_report.json").read_text())
    if (yolo_config["split"] != "val" or yolo_config["fold"] != args.fold
            or yolo_config["manifest_sha256"] != digest_file(manifest_path)
            or any(yolo_report[key] != value for key, value in yolo_config.items())):
        raise ValueError("YOLO cache is not the requested complete held-out inference")
    validation = validate_provenance(yolo_report["source_provenance"], manifest, args.fold)
    if validation != yolo_config["validation"] or yolo_report["source_provenance"].get("benchmark"):
        raise ValueError("YOLO provenance validation differs or checkpoint is a benchmark")
    if digest_file(yolo_report["checkpoint"]) != yolo_config["checkpoint_sha256"]:
        raise ValueError("YOLO checkpoint fingerprint changed")
    # Read stored metadata on CPU; do not initialize a predictor or touch CUDA.
    import torch
    checkpoint = torch.load(yolo_report["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    base_model = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
    if getattr(base_model, "filament_provenance", None) != yolo_report["source_provenance"]:
        raise ValueError("YOLO report provenance differs from the actual checkpoint")
    del base_model, checkpoint
    if set(yolo_report["image_ids"]) != set(records) or yolo_report["images"] != len(records):
        raise ValueError("YOLO inference report is incomplete")
    if set(p.stem for p in (yolo_dir / "raw").glob("*.json")) != set(records):
        raise ValueError("YOLO raw image cache incomplete or contains extra images")
    if set(p.stem for p in prob_dir.glob("*.npz")) != set(records):
        raise ValueError("R34 probability cache incomplete or contains extra images")
    identity = {"schema": 1, "fold": args.fold, "manifest_sha256": digest_file(manifest_path),
                "r34_identity": base_identity, "r34_checkpoint_validation": r34_provenance,
                "r34_features_sha256": digest_file(quality_dir / "features.joblib"),
                "r34_oof_sha256": digest_file(quality_dir / "oof_quality.npy"),
                "yolo_inference": yolo_config, "yolo_source_provenance": yolo_report["source_provenance"],
                "raw_sha256": {stem: digest_file(yolo_dir / "raw" / f"{stem}.json") for stem in records},
                "prob_sha256": {stem: digest_file(prob_dir / f"{stem}.npz") for stem in records},
                "features": FEATURES + EXTRA_FEATURES, "r34_pool": POOL_CONFIG,
                "yolo_min_area": 60, "mask_nms_iou": .5, "fixed_addition_conf": .25,
                "fixed_addition_max_iou_to_r34": .1, "fixed_r34_quality_threshold": .2,
                "regressor_params": new_regressor().get_params(), "quality_thresholds": THRESHOLDS.tolist()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "features.joblib"
    if cache_path.exists():
        fusion_cache = joblib.load(cache_path)
        if fusion_cache["identity"] != identity:
            raise ValueError("Fusion feature cache identity changed; use a new output directory")
        entries = fusion_cache["entries"]
    else:
        jobs = []
        for base in base_entries:
            stem = base["image_id"]
            record = records[stem]
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            jobs.append((record, base, prob_dir / f"{stem}.npz", image_path, yolo_dir / "raw" / f"{stem}.json"))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            entries = []
            for index, entry in enumerate(pool.map(extract_one, jobs), 1):
                entries.append(entry)
                if index % 20 == 0 or index == len(jobs):
                    print(json.dumps({"images_extracted": index, "total": len(jobs)}), flush=True)
        atomic_write(cache_path, lambda stream: joblib.dump({"identity": identity, "entries": entries}, stream))
    group_folds = original_group_folds(base_entries)
    reproduced_oof = crossfit(base_entries, group_folds)
    if not np.allclose(reproduced_oof, original_oof, atol=1e-7, rtol=0):
        raise ValueError("R34 baseline cross-fit does not reproduce original OOF scores")
    base_selected, fixed_selected, yolo_selected = [], [], []
    offset = 0
    complement = {"gt_records": 0, "r34_selected_hit": 0, "r34_pool_hit": 0,
                  "yolo_pool_hit": 0, "yolo_fixed_hit": 0,
                  "yolo_pool_hit_r34_selected_miss": 0, "yolo_pool_hit_r34_pool_miss": 0,
                  "yolo_fixed_hit_r34_selected_miss": 0}
    for entry in entries:
        nr = entry["r34_count"]
        base_keep = np.flatnonzero(original_oof[offset:offset + nr] >= .2)
        offset += nr
        yi = np.flatnonzero(entry["features"][nr:, len(FEATURES) + 1] >= .25) + nr
        yi = yi[greedy_nms(entry["features"][yi, len(FEATURES) + 1], entry["pair_iou"][np.ix_(yi, yi)])]
        novel = yi[(entry["pair_iou"][np.ix_(yi, base_keep)] <= .1).all(axis=1)]
        base_selected.append(base_keep)
        yolo_selected.append(yi)
        fixed_selected.append(np.concatenate([base_keep, novel]))
        for matrix in entry["iou"]:
            rb = (matrix[:, base_keep] > .5).any(axis=1)
            rp = (matrix[:, :nr] > .5).any(axis=1)
            yp = (matrix[:, nr:] > .5).any(axis=1)
            yf = (matrix[:, yi] > .5).any(axis=1)
            for key, value in [("gt_records", len(matrix)), ("r34_selected_hit", rb.sum()),
                ("r34_pool_hit", rp.sum()), ("yolo_pool_hit", yp.sum()), ("yolo_fixed_hit", yf.sum()),
                ("yolo_pool_hit_r34_selected_miss", (yp & ~rb).sum()),
                ("yolo_pool_hit_r34_pool_miss", (yp & ~rp).sum()),
                ("yolo_fixed_hit_r34_selected_miss", (yf & ~rb).sum())]:
                complement[key] += int(value)
    baseline, baseline_images = score_kept(entries, base_selected)
    if not np.isclose(baseline["pq"], score_selection(base_entries, original_oof, .2).pq, atol=1e-7, rtol=0):
        raise ValueError("Native RLE baseline does not reproduce cached quality PQ")
    fixed, fixed_images = score_kept(entries, fixed_selected)
    yolo_fixed, _ = score_kept(entries, yolo_selected)
    atomic_json(output / "fixed_fusion.json", {"r34_baseline": baseline, "yolo_conf25_nms50_area60": yolo_fixed,
        "fixed_novel_additions": fixed, "delta_pq": fixed["pq"] - baseline["pq"], "complementarity": complement})
    print(json.dumps({"r34_baseline": baseline, "fixed_fusion": fixed, "complementarity": complement}), flush=True)
    oof = crossfit(entries, group_folds)
    ranking = []
    for threshold in THRESHOLDS:
        stat, _ = score_kept(entries, quality_selections(entries, oof, threshold))
        ranking.append({"quality_threshold": float(threshold), **stat})
    ranking.sort(key=lambda row: row["pq"], reverse=True)
    best = ranking[0]
    best_selected = quality_selections(entries, oof, best["quality_threshold"])
    _, best_images = score_kept(entries, best_selected)
    # Independently verify exact metric on the native RLE selected masks.
    direct = PQStats()
    for entry, selected in zip(entries, best_selected):
        for ann in records[entry["image_id"]]["annotators"]:
            direct += score_rles(annotation_rles(ann), [entry["rles"][i] for i in selected])
    if any(direct.to_dict()[k] != best[k] for k in ("tp", "fp", "fn")) or not np.isclose(direct.pq, best["pq"], atol=1e-12, rtol=0):
        raise ValueError("Joint matrix metric differs from native RLE exact metric")
    final_report = {"identity": identity, "images": len(entries), "instances": len(oof),
        "group_folds": group_folds, "r34_baseline": baseline, "yolo_conf25_nms50_area60": yolo_fixed,
        "fixed_novel_additions": fixed, "complementarity": complement, "best_joint_oof": best,
        "joint_delta_pq": best["pq"] - baseline["pq"], "ranking": ranking,
        "checks": {"original_r34_oof_reproduced": True, "all_original_r34_iou_matrices_reproduced": True,
                   "actual_yolo_checkpoint_embedded_provenance_verified": True,
                   "best_joint_native_rle_exact_metric": direct.to_dict()},
        "interpretation": "Both base models held out this fold. Quality crossfit uses the original R34 date-group partitions; raw features only, no prior OOF scores as learned inputs. Threshold selected on these OOF results, so confirm on another base fold. Public all-data YOLO is not used, and transfer to that stronger but different model remains unvalidated."}
    atomic_write(output / "oof_quality.npy", lambda stream: np.save(stream, oof))
    atomic_json(output / "report.json", final_report)
    atomic_json(output / "per_image.json", {stem: {"r34": baseline_images[stem], "fixed_fusion": fixed_images[stem],
        "joint_quality": best_images[stem]} for stem in baseline_images})
    print(json.dumps({"r34": baseline, "fixed": fixed, "joint_best": best,
                      "joint_delta_pq": final_report["joint_delta_pq"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--r34-quality", default="outputs/unet_r34_f0/quality_e040_tta4")
    parser.add_argument("--r34-prob", default="outputs/unet_r34_f0/val_e040_tta4")
    parser.add_argument("--yolo-dir", default="outputs/yolo26s_f0/val_e011")
    parser.add_argument("--output", default="outputs/diagnostics/fusion_r34_yolo_e011")
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

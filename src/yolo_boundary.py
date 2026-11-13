"""Confirm the frozen YOLO/R34 boundary refinement on held-out predictions.

Example (CPU only)::

    CUDA_VISIBLE_DEVICES='' python -m src.yolo_boundary \
      --yolo-dir outputs/yolo26s_f0/val_e011 \
      --prob-dir outputs/unet_r34_f0/val_e040_tta4 \
      --output outputs/diagnostics/boundary_confirmation_e011

The CLI accepts verified own-model validation caches only. It never runs a
predictor, handles test/public-model caches, or uploads a submission. All
parameters are frozen from the three-variant epoch-11 experiment. Ground truth
is loaded only after the label-free transformations finish.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from .ensemble import atomic_json, checkpoint_provenance, digest_file
from .infer_yolo import filter_detections
from .inference import write_submission
from .metrics import PQStats, json_rle, load_ground_truth, score_image
from .train_yolo import validate_provenance


PARAMETERS = {
    "r34_weight": .75, "gaussian_sigma": 3., "gaussian_kernel": 25,
    "dilation_radius": 12, "dilation_shape": "OpenCV ellipse",
    "pixel_threshold": .5, "confidence": .20, "mask_nms_iou": .5,
    "min_area": 5, "native_shape": [2048, 2048],
    "gaussian_border": "constant zero",
}


def mask_nms(detections):
    """Suppress duplicates by original detector confidence; retain native pixels."""
    kept = []
    for detection in sorted(detections, key=lambda item: -item["score"]):
        if kept and np.any(mu.iou([detection["rle"]], [d["rle"] for d in kept],
                                  [0] * len(kept)) > .5):
            continue
        kept.append(detection)
    return kept


def duplicate_pairs(detections):
    if len(detections) < 2:
        return 0
    rles = [d["rle"] for d in detections]
    return int(np.triu(mu.iou(rles, rles, [0] * len(rles)) > .5, 1).sum())


def refine_detections(raw, probability, *, confidence_floor=.20):
    """Pure CPU transformation. Inputs contain no ground truth or learned choices.

    Every retained object keeps its raw detection index and confidence. All
    disconnected pieces remain one instance; no new identities are introduced.
    The CLI keeps its frozen .20 floor. The optional keyword supports separate
    low-confidence proposal-quality experiments without changing geometry.
    """
    if not np.isfinite(confidence_floor) or not 0 <= confidence_floor <= 1:
        raise ValueError("Confidence floor must be finite and within [0, 1]")
    probability = np.asarray(probability, dtype=np.float32)
    if (probability.shape != (2048, 2048) or not np.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1):
        raise ValueError("Expected finite native foreground probabilities in [0, 1]")
    for detection in raw:
        score = detection["score"]
        rle = detection["rle"]
        if (not np.isfinite(score) or not 0 <= score <= 1
                or rle["size"] != [2048, 2048]
                or int(mu.area(rle)) != detection["area"]):
            raise ValueError("Invalid native YOLO detection")
    selected = filter_detections(raw, confidence_floor, min_area=5, disjoint=False, mask_nms_iou=.5)
    by_counts = {}
    for index, detection in sorted(enumerate(raw), key=lambda pair: -pair[1]["score"]):
        # Exact duplicate RLEs retain the highest-confidence original identity.
        by_counts.setdefault(detection["rle"]["counts"], (index, detection))
    baseline = []
    for rle in selected:
        index, detection = by_counts[rle["counts"]]
        baseline.append({"source_index": index, "score": detection["score"],
                         "rle": rle, "area": int(mu.area(rle))})
    refined, area_ratios = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for detection in baseline:
        x, y, width, height = map(int, mu.toBbox(detection["rle"]))
        xs = slice(max(0, x - 12), min(2048, x + width + 12))
        ys = slice(max(0, y - 12), min(2048, y + height + 12))
        mask = mu.decode(detection["rle"])[ys, xs].astype(np.uint8)
        support = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)
        softened = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 3.,
                                   borderType=cv2.BORDER_CONSTANT)
        adjusted = ((.25 * softened + .75 * probability[ys, xs]) >= .5) & support
        area = int(adjusted.sum())
        area_ratios.append(area / detection["area"])
        if area < 5:
            continue
        native = np.zeros((2048, 2048), dtype=np.uint8)
        native[ys, xs] = adjusted
        rle = json_rle(mu.encode(np.asfortranarray(native)))
        if not np.array_equal(mu.decode(rle), native):
            raise ValueError("Refined native RLE does not round-trip")
        refined.append({"source_index": detection["source_index"],
                        "score": detection["score"], "rle": rle, "area": area})
    retained = mask_nms(refined)
    if duplicate_pairs(retained):
        raise ValueError("Mask NMS left a duplicate pair above IoU .5")
    return {"baseline": baseline, "pre_nms": refined, "post_nms": retained,
            "dropped_area_lt5": len(baseline) - len(refined),
            "dropped_nms": len(refined) - len(retained),
            "duplicate_pairs_before_nms": duplicate_pairs(refined),
            "area_ratios": area_ratios}


def transform_one(job):
    stem, raw_path, probability_path = job
    cv2.setNumThreads(1)
    with np.load(probability_path, allow_pickle=False) as archive:
        array = archive["probability"]
        if array.ndim != 3 or array.shape[0] < 1:
            raise ValueError("Probability cache must contain CxHxW arrays")
        probability = array[0].astype(np.float32)
    return {"image_id": stem, **refine_detections(json.loads(Path(raw_path).read_text()), probability)}


def score_results(results, truth, key):
    total, per_image = PQStats(), {}
    matching = {"matched_gt_instances": 0, "gt_with_multiple_prediction_hits": 0,
                "predictions_with_multiple_gt_hits": 0}
    for result in results:
        stem = result["image_id"]
        rles = [d["rle"] for d in result[key]]
        stat = score_image(truth[stem], rles)
        total += stat
        per_image[stem] = stat.to_dict()
        for _, ground_truth in truth[stem]:
            matrix = (np.asarray(mu.iou(rles, ground_truth, [0] * len(ground_truth)))
                      if rles and ground_truth else np.zeros((len(rles), len(ground_truth))))
            hit = matrix > .5
            matching["matched_gt_instances"] += int(hit.any(axis=0).sum())
            matching["gt_with_multiple_prediction_hits"] += int((hit.sum(axis=0) > 1).sum())
            matching["predictions_with_multiple_gt_hits"] += int((hit.sum(axis=1) > 1).sum())
    return {**total.to_dict(), **matching}, per_image


def paired_bootstrap(baseline, refined, groups):
    unique = sorted(set(groups.values()))
    lookup = {group: index for index, group in enumerate(unique)}
    values = np.zeros((2, len(unique), 4), dtype=np.float64)
    for index, scores in enumerate((baseline, refined)):
        for stem, stat in scores.items():
            values[index, lookup[groups[stem]]] += [stat[k] for k in ("iou_sum", "tp", "fp", "fn")]
    def pq(array):
        return array[..., 0] / np.maximum(array[..., 1] + .5 * (array[..., 2] + array[..., 3]), 1)
    draws = np.random.default_rng(2026).integers(0, len(unique), size=(10000, len(unique)))
    sampled = values[:, draws, :].sum(axis=2)
    delta = pq(sampled[1]) - pq(sampled[0])
    return {"date_groups": len(unique), "draws": 10000, "seed": 2026,
            "delta": float(pq(values[1].sum(axis=0)) - pq(values[0].sum(axis=0))),
            "paired_date_group_bootstrap_95pct": np.quantile(delta, [.025, .975]).tolist(),
            "bootstrap_fraction_nonpositive": float(np.mean(delta <= 0))}


def validate_sources(yolo_dir, prob_dir, manifest_path, fold):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("This confirmation CLI must run without CUDA initialization")
    manifest = json.loads(manifest_path.read_text())
    records = {r["image_id"]: r for r in manifest["train"] if r["fold"] == fold}
    if fold < 0 or not records:
        raise ValueError("A nonempty held-out fold is required")
    yolo = json.loads((yolo_dir / "inference_config.json").read_text())
    report = json.loads((yolo_dir / "run_report.json").read_text())
    probability = json.loads((prob_dir / "inference_config.json").read_text())
    if (yolo["split"] != "val" or probability["split"] != "val"
            or yolo["fold"] != fold or probability["fold"] != fold):
        raise ValueError("Only two models that held out the same validation fold are supported")
    if yolo["manifest_sha256"] != digest_file(manifest_path):
        raise ValueError("YOLO manifest hash differs")
    if (yolo["raw_confidence"] > .20 or not yolo["retina_masks"]
            or yolo["end2end"] or yolo["nms_iou"] != .5):
        raise ValueError("Expected native one-to-many YOLO cache with raw confidence <=.20 and box NMS .5")
    if any(report[key] != value for key, value in yolo.items()):
        raise ValueError("YOLO raw cache configuration differs from its completed report")
    if (set(report["image_ids"]) != set(records) or report["images"] != len(records)
            or {p.stem for p in (yolo_dir / "raw").glob("*.json")} != set(records)
            or {p.stem for p in prob_dir.glob("*.npz")} != set(records)):
        raise ValueError("Prediction caches must cover exactly the held-out image set")
    if (validate_provenance(report["source_provenance"], manifest, fold) != yolo["validation"]
            or report["source_provenance"].get("benchmark")):
        raise ValueError("YOLO provenance is not a trained held-out model")
    if digest_file(report["checkpoint"]) != yolo["checkpoint_sha256"]:
        raise ValueError("YOLO checkpoint fingerprint changed")
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    if getattr(model, "filament_provenance", None) != report["source_provenance"]:
        raise ValueError("Actual YOLO checkpoint provenance differs from the report")
    del model, checkpoint
    provenance = checkpoint_provenance(probability, prob_dir, manifest, {})
    if len(provenance) != 1 or probability.get("ensemble"):
        raise ValueError("Frozen confirmation requires one R34 model, not an ensemble")
    checkpoint = torch.load(provenance[0]["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["config"].get("encoder") != "resnet34":
        raise ValueError("Frozen confirmation requires the R34 semantic model")
    del checkpoint
    if any(probability.get(key) != value for key, value in
           {"tta": 4, "scale": 1., "tile_size": 768, "stride": 512}.items()):
        raise ValueError("Frozen confirmation requires native R34 TTA4, tile768, stride512")
    annotation_path = Path(manifest["annotation_path"])
    if not annotation_path.is_absolute():
        annotation_path = manifest_path.parent / annotation_path
    if digest_file(annotation_path) != manifest["annotation_sha256"]:
        raise ValueError("Official annotation file hash changed")
    identity = {"schema": 1, "fold": fold, "parameters": PARAMETERS,
                "manifest_sha256": digest_file(manifest_path), "yolo": yolo,
                "r34": probability, "r34_checkpoint_provenance": provenance,
                "source_sha256": digest_file(__file__),
                "raw_sha256": {stem: digest_file(yolo_dir / "raw" / f"{stem}.json") for stem in records},
                "prob_sha256": {stem: digest_file(prob_dir / f"{stem}.npz") for stem in records}}
    return records, annotation_path, identity


def run(args):
    import torch
    started = time.monotonic()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve prior experiments")
    yolo_dir, prob_dir = Path(args.yolo_dir), Path(args.prob_dir)
    records, annotation_path, identity = validate_sources(yolo_dir, prob_dir, Path(args.manifest).resolve(), args.fold)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "identity.json", identity)
    ids = sorted(records)
    jobs = [(stem, yolo_dir / "raw" / f"{stem}.json", prob_dir / f"{stem}.npz") for stem in ids]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for index, result in enumerate(pool.map(transform_one, jobs), 1):
            results.append(result)
            if index % 20 == 0 or index == len(ids):
                print(json.dumps({"transformed_images": index, "total": len(ids)}), flush=True)
    atomic_json(output / "predictions.json", {r["image_id"]: r for r in results})
    exports = {}
    for key, name in (("baseline", "baseline.csv"), ("post_nms", "refined.csv")):
        path = output / name
        info = write_submission(((r["image_id"], [d["rle"] for d in r[key]]) for r in results), path, ids)
        exports[key] = {**info, "sha256": digest_file(path)}
    # Evaluation starts only after transformation and export, with no feedback.
    truth = load_ground_truth(annotation_path)
    scores, per_image = {}, {}
    for key in ("baseline", "pre_nms", "post_nms"):
        scores[key], per_image[key] = score_results(results, truth, key)
    ratios = np.asarray([ratio for r in results for ratio in r["area_ratios"]])
    for stem in ids:
        if (digest_file(yolo_dir / "raw" / f"{stem}.json") != identity["raw_sha256"][stem]
                or digest_file(prob_dir / f"{stem}.npz") != identity["prob_sha256"][stem]):
            raise ValueError("Source prediction cache changed during confirmation")
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    report = {"parameters": PARAMETERS, "images": len(ids), "scores": scores,
              "exports": exports, "instances": {key: sum(len(r[key]) for r in results)
                                                   for key in ("baseline", "pre_nms", "post_nms")},
              "dropped_area_lt5": sum(r["dropped_area_lt5"] for r in results),
              "dropped_nms": sum(r["dropped_nms"] for r in results),
              "duplicate_pairs_before_nms": sum(r["duplicate_pairs_before_nms"] for r in results),
              "duplicate_pairs_after_nms": 0,
              "area_ratio_mean": float(ratios.mean()) if len(ratios) else None,
              "area_ratio_median": float(np.median(ratios)) if len(ratios) else None,
              "bootstrap": paired_bootstrap(per_image["baseline"], per_image["post_nms"],
                                            {stem: r["date_group"] for stem, r in records.items()}),
              "seconds": time.monotonic() - started, "cuda_initialized": False,
              "source_hashes_unchanged": True, "automatic_submission": False,
              "interpretation": "Frozen epoch-11 boundary rule; exact PQ on held-out own-model predictions. "
                  "Ground truth never enters the transformation. Bootstrap is conditional on predictions; "
                  "this does not establish transfer to public models or test data."}
    atomic_json(output / "report.json", report)
    atomic_json(output / "per_image.json", per_image)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-dir", required=True)
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run this CPU confirmation with CUDA_VISIBLE_DEVICES=''")
    run(args)


if __name__ == "__main__":
    main()

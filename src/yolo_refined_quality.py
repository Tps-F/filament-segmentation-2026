"""CPU-only cross-fitted quality for frozen refined YOLO proposals.

Use only the existing raw confidence floor .03, fixed boundary geometry, and
mask NMS .5. Features are 26 raw mask/image/R34 measurements plus the original
YOLO confidence. The original R34 date-group meta-folds are reused exactly;
its OOF scores are never learned inputs. This experiment fits no test model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from pycocotools import mask as mu

from .candidate_fusion import arbitrary_features, annotation_rles, crossfit, original_group_folds, rle_iou
from .ensemble import atomic_json, atomic_write, digest_file
from .inference import write_submission
from .metrics import PQStats, score_rles
from .postprocess import PostprocessConfig, probabilities_to_labels
from .quality import FEATURES, new_regressor, score_selection
from .quality_pool import POOL_CONFIG, validate_entries
from .yolo_boundary import PARAMETERS, paired_bootstrap, refine_detections, validate_sources


QUALITY_FEATURES = FEATURES + ["yolo_confidence"]
THRESHOLDS = np.arange(.05, .401, .025)
RAW_FLOOR = .03


def quality_thresholds(fixed_threshold=None):
    if fixed_threshold is None:
        return THRESHOLDS
    if not np.isfinite(fixed_threshold) or not 0 <= fixed_threshold <= 1:
        raise ValueError("Fixed quality threshold must be finite and within [0,1]")
    return np.asarray([fixed_threshold], dtype=float)


def extract_one(job):
    record, raw_path, probability_path, image_path, expected_fixed = job
    cv2.setNumThreads(1)
    raw = json.loads(Path(raw_path).read_text())
    with np.load(probability_path, allow_pickle=False) as archive:
        probability = archive["probability"].astype(np.float32)
    image = np.load(image_path, allow_pickle=False)
    # Geometry and raw features do not receive annotation data.
    refined = refine_detections(raw, probability[0], confidence_floor=RAW_FLOOR)
    proposals = refined["post_nms"]
    fixed = [candidate for candidate in proposals if candidate["score"] >= .20]
    if fixed != expected_fixed:
        raise ValueError("Low-floor pool does not reproduce the frozen confidence-.20 masks exactly")
    labels = probabilities_to_labels(probability, PostprocessConfig(**POOL_CONFIG))
    features = np.asarray([arbitrary_features(candidate["rle"], probability, image, labels)
                           for candidate in proposals], dtype=np.float32).reshape(-1, len(FEATURES))
    confidence = np.asarray([candidate["score"] for candidate in proposals], dtype=np.float32)
    features = np.concatenate([features, confidence[:, None]], axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite raw proposal features")
    rles = [candidate["rle"] for candidate in proposals]
    matrices = [rle_iou(rles, annotation_rles(annotation)).T for annotation in record["annotators"]]
    target = sum(np.where(matrix > .5, matrix, 0).sum(axis=0) for matrix in matrices) / len(matrices)
    return {"image_id": record["image_id"], "group": record["date_group"], "features": features,
            "target": target.astype(np.float32), "weight": len(matrices), "iou": matrices,
            "rles": rles, "source_indices": [candidate["source_index"] for candidate in proposals],
            "confidence": confidence, "fixed_baseline_exact": True,
            "raw_count": len(raw), "dropped_area_lt5": refined["dropped_area_lt5"],
            "dropped_nms": refined["dropped_nms"]}


def evaluate(entries, scores, threshold):
    total, per_image, offset = PQStats(), {}, 0
    instances = 0
    matching = {"matched_gt_instances": 0, "gt_with_multiple_prediction_hits": 0,
                "predictions_with_multiple_gt_hits": 0}
    for entry in entries:
        count = len(entry["target"])
        keep = scores[offset:offset + count] >= threshold
        offset += count
        stat = PQStats()
        for matrix in entry["iou"]:
            selected = matrix[:, keep]
            hit = selected > .5
            stat += PQStats(float(selected[hit].sum()), int(hit.sum()),
                            int((~hit.any(axis=0)).sum()), int((~hit.any(axis=1)).sum()))
            matching["matched_gt_instances"] += int(hit.any(axis=1).sum())
            matching["gt_with_multiple_prediction_hits"] += int((hit.sum(axis=1) > 1).sum())
            matching["predictions_with_multiple_gt_hits"] += int((hit.sum(axis=0) > 1).sum())
        instances += int(keep.sum())
        total += stat
        per_image[entry["image_id"]] = stat.to_dict()
    if offset != len(scores):
        raise ValueError("Candidate and score counts differ")
    return {**total.to_dict(), **matching, "instances": instances}, per_image


def proposal_diagnostics(entries, confidence, quality, quality_threshold):
    """GT-aware coverage diagnostics only; never used by proposal transforms."""
    names = ("all_refined_proposals", "fixed_conf20", "quality_oof")
    coverage = {name: {"gt_records": 0, "covered_gt_records": 0,
        "best_matched_iou_sum": 0., "all_matched_pair_iou_sum": 0., "all_matched_pairs": 0,
        "uncovered_best_iou_lt_0.2": 0, "uncovered_best_iou_0.2_through_0.5": 0,
        "candidate_instances": 0} for name in names}
    overlap = {name: {"images_with_pixel_overlap": 0, "overlapping_mask_pairs": 0,
        "excess_mask_coverage_pixels": 0, "total_mask_pixels": 0} for name in names}
    filtering = {"pool_covered_but_conf20_missed": 0, "pool_covered_but_quality_missed": 0,
                 "quality_recovers_conf20_miss": 0, "quality_loses_conf20_hit": 0}
    offset = 0
    for entry in entries:
        n = len(entry["target"])
        selections = {"all_refined_proposals": np.ones(n, dtype=bool),
                      "fixed_conf20": confidence[offset:offset + n] >= .20,
                      "quality_oof": quality[offset:offset + n] >= quality_threshold}
        offset += n
        for name, keep in selections.items():
            rles = [rle for rle, retained in zip(entry["rles"], keep) if retained]
            coverage[name]["candidate_instances"] += len(rles)
            total_area = int(np.asarray(mu.area(rles)).sum()) if rles else 0
            union_area = int(mu.area(mu.merge(rles))) if rles else 0
            excess = total_area - union_area
            overlap[name]["total_mask_pixels"] += total_area
            overlap[name]["excess_mask_coverage_pixels"] += excess
            overlap[name]["images_with_pixel_overlap"] += int(excess > 0)
            if len(rles) > 1:
                overlap[name]["overlapping_mask_pairs"] += int(np.triu(mu.iou(rles, rles, [0] * len(rles)) > 0, 1).sum())
        for matrix in entry["iou"]:
            covered = {}
            for name, keep in selections.items():
                selected = matrix[:, keep]
                maxima = selected.max(axis=1) if selected.shape[1] else np.zeros(len(matrix))
                hit = maxima > .5
                covered[name] = hit
                row = coverage[name]
                row["gt_records"] += len(matrix)
                row["covered_gt_records"] += int(hit.sum())
                row["best_matched_iou_sum"] += float(maxima[hit].sum())
                row["all_matched_pair_iou_sum"] += float(selected[selected > .5].sum())
                row["all_matched_pairs"] += int((selected > .5).sum())
                row["uncovered_best_iou_lt_0.2"] += int((maxima < .2).sum())
                row["uncovered_best_iou_0.2_through_0.5"] += int(((maxima >= .2) & (maxima <= .5)).sum())
            pool, fixed, learned = (covered[name] for name in names)
            filtering["pool_covered_but_conf20_missed"] += int((pool & ~fixed).sum())
            filtering["pool_covered_but_quality_missed"] += int((pool & ~learned).sum())
            filtering["quality_recovers_conf20_miss"] += int((learned & ~fixed).sum())
            filtering["quality_loses_conf20_hit"] += int((fixed & ~learned).sum())
    if offset != len(confidence) or offset != len(quality):
        raise ValueError("Diagnostic score lengths differ")
    for row in coverage.values():
        row["coverage_fraction"] = row["covered_gt_records"] / max(1, row["gt_records"])
        row["mean_best_iou_for_covered_gt"] = row["best_matched_iou_sum"] / max(1, row["covered_gt_records"])
        row["mean_all_pair_matched_iou"] = row["all_matched_pair_iou_sum"] / max(1, row["all_matched_pairs"])
    for row in overlap.values():
        row["pixel_disjoint"] = row["excess_mask_coverage_pixels"] == 0
    return {"coverage": coverage, "filtering": filtering, "pixel_overlap": overlap,
            "interpretation": "Every GT instance in every annotator-image record is counted separately. "
                "Coverage means at least one candidate has IoU strictly above .5; mean-best IoU counts each covered GT once. "
                "All-pair mean follows the official matching rule. Oracle coverage is a diagnostic, not a prediction rule. "
                "Mask NMS does not guarantee pixel-disjoint outputs; output-contract suitability remains separate."}


def run(args):
    import torch
    fixed_threshold = getattr(args, "fixed_threshold", None)
    thresholds = quality_thresholds(fixed_threshold)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Run with CUDA_VISIBLE_DEVICES='' without CUDA initialization")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CPU workers")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory for this bounded experiment")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    yolo_dir, prob_dir, quality_dir, fixed_dir = map(Path,
        [args.yolo_dir, args.prob_dir, args.r34_quality, args.frozen_boundary])
    records, _, source_identity = validate_sources(yolo_dir, prob_dir, manifest_path, args.fold)
    if source_identity["yolo"]["raw_confidence"] != RAW_FLOOR:
        raise ValueError("Expected the existing fixed raw confidence floor .03")
    yolo_report = json.loads((yolo_dir / "run_report.json").read_text())
    if yolo_report["raw_instances"] > 5000:
        raise ValueError("Raw pool exceeds the bounded 5000-candidate budget; review before running")
    fixed_identity = json.loads((fixed_dir / "identity.json").read_text())
    if (fixed_identity["yolo"] != source_identity["yolo"] or fixed_identity["r34"] != source_identity["r34"]
            or fixed_identity["parameters"] != PARAMETERS):
        raise ValueError("Frozen baseline comes from different models or geometry")
    fixed_report = json.loads((fixed_dir / "report.json").read_text())
    fixed_predictions = json.loads((fixed_dir / "predictions.json").read_text())
    if set(fixed_predictions) != set(records):
        raise ValueError("Frozen baseline image set differs")
    cache = joblib.load(quality_dir / "features.joblib")
    base_entries = cache["entries"]
    base_oof = np.load(quality_dir / "oof_quality.npy", allow_pickle=False)
    base_report = json.loads((quality_dir / "report.json").read_text())
    validate_entries(base_entries, base_oof, records, base_report)
    if (cache["identity"]["fold"] != args.fold or cache["identity"]["config"] != POOL_CONFIG
            or cache["identity"]["inference"] != source_identity["r34"]
            or cache["identity"]["annotation_sha256"] != manifest["annotation_sha256"]):
        raise ValueError("R34 quality source provenance differs")
    group_folds = original_group_folds(base_entries)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"sources": source_identity, "features": QUALITY_FEATURES,
                "proposal_floor": RAW_FLOOR, "geometry": PARAMETERS,
                "note": "Geometry and CLIbaseline remain frozen; only proposal inclusion floor is .03 for quality training.",
                "r34_context_pool": POOL_CONFIG, "regressor_params": new_regressor().get_params(),
                "quality_thresholds": thresholds.tolist(), "group_folds": group_folds,
                "quality_features_sha256": digest_file(quality_dir / "features.joblib"),
                "quality_oof_sha256": digest_file(quality_dir / "oof_quality.npy"),
                "frozen_predictions_sha256": digest_file(fixed_dir / "predictions.json"),
                "source_code_sha256": digest_file(__file__),
                "boundary_code_sha256": digest_file(Path(__file__).with_name("yolo_boundary.py")),
                "feature_code_sha256": digest_file(Path(__file__).with_name("candidate_fusion.py"))}
    if fixed_threshold is not None:
        identity["fixed_threshold"] = float(fixed_threshold)
    atomic_json(output / "predeclared_plan.json", identity)
    jobs = []
    for base in base_entries:
        stem = base["image_id"]
        image_path = Path(records[stem]["image_path"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        jobs.append((records[stem], yolo_dir / "raw" / f"{stem}.json", prob_dir / f"{stem}.npz",
                     image_path, fixed_predictions[stem]["post_nms"]))
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for index, entry in enumerate(pool.map(extract_one, jobs), 1):
            entries.append(entry)
            if index % 20 == 0 or index == len(jobs):
                print(json.dumps({"images_extracted": index, "total": len(jobs)}), flush=True)
    pool_count = sum(len(entry["target"]) for entry in entries)
    if pool_count > 5000:
        raise ValueError("Refined candidate pool exceeds the declared budget")
    atomic_write(output / "features.joblib", lambda stream: joblib.dump({"identity": identity, "entries": entries}, stream))
    # Baseline proof uses original confidences only, never learned quality scores.
    confidence = np.concatenate([entry["confidence"] for entry in entries])
    fixed, fixed_images = evaluate(entries, confidence, .20)
    expected = fixed_report["scores"]["post_nms"]
    if any(fixed[key] != expected[key] for key in ("tp", "fp", "fn")) or not np.isclose(fixed["pq"], expected["pq"], atol=1e-12, rtol=0):
        raise ValueError("Exact fixed confidence-.20 baseline metric changed")
    reproduced_base = crossfit(base_entries, group_folds)
    if not np.array_equal(reproduced_base, base_oof):
        raise ValueError("Original R34 date-group OOF scores did not exactly reproduce")
    oof = crossfit(entries, group_folds)
    ranking = []
    for threshold in thresholds:
        score, _ = evaluate(entries, oof, threshold)
        ranking.append({"quality_threshold": float(threshold), **score})
    ranking.sort(key=lambda row: row["pq"], reverse=True)
    best = ranking[0]
    _, best_images = evaluate(entries, oof, best["quality_threshold"])
    offset, predictions, direct = 0, [], PQStats()
    for entry in entries:
        count = len(entry["target"])
        keep = oof[offset:offset + count] >= best["quality_threshold"]
        offset += count
        selected = [rle for rle, retained in zip(entry["rles"], keep) if retained]
        if len(selected) > 1:
            pair = mu.iou(selected, selected, [0] * len(selected))
            np.fill_diagonal(pair, 0)
            if pair.max() > .5:
                raise ValueError("Filtered proposal pool contains a mask duplicate above IoU .5")
        predictions.append((entry["image_id"], selected))
        for annotation in records[entry["image_id"]]["annotators"]:
            direct += score_rles(annotation_rles(annotation), selected)
    if any(direct.to_dict()[key] != best[key] for key in ("tp", "fp", "fn")) or not np.isclose(direct.pq, best["pq"], atol=1e-12, rtol=0):
        raise ValueError("Native RLE metric disagrees with cached matrices")
    export = write_submission(predictions, output / "selected_oof.csv", list(records))
    export["sha256"] = digest_file(output / "selected_oof.csv")
    base_stat = score_selection(base_entries, base_oof, .20)
    base_images, offset = {}, 0
    for entry in base_entries:
        count = len(entry["target"])
        base_images[entry["image_id"]] = score_selection([entry], base_oof[offset:offset + count], .20).to_dict()
        offset += count
    groups = {entry["image_id"]: entry["group"] for entry in entries}
    if torch.cuda.is_initialized():
        raise ValueError("Unexpected CUDA initialization")
    for stem in records:
        if (digest_file(yolo_dir / "raw" / f"{stem}.json") != source_identity["raw_sha256"][stem]
                or digest_file(prob_dir / f"{stem}.npz") != source_identity["prob_sha256"][stem]):
            raise ValueError("Source prediction cache changed during quality fitting")
    report = {"identity": identity, "images": len(entries), "raw_candidates": yolo_report["raw_instances"],
              "refined_pool_candidates": pool_count, "fixed_conf20_baseline": fixed,
              "r34_quality_q20": base_stat.to_dict(), "best": best, "ranking": ranking,
              "vs_fixed_conf20": paired_bootstrap(fixed_images, best_images, groups),
              "vs_r34": paired_bootstrap(base_images, best_images, groups),
              "proposal_diagnostics": proposal_diagnostics(entries, confidence, oof, best["quality_threshold"]),
              "checks": {"all_frozen_conf20_masks_exactly_reproduced": True,
                         "original_r34_meta_partition_and_oof_exactly_reproduced": True,
                         "native_rle_exact_metric": direct.to_dict(), "selected_mask_duplicate_pairs": 0,
                         "source_hashes_unchanged": True, "cuda_initialized": False},
              "export": export,
              "interpretation": "Refined YOLO proposals only; no R34 proposal fusion. Both base models hold out this fold. "
                  "Quality crossfit uses the original R34 observation-date meta partitions and raw features only. "
                  + ("The quality cutoff was fixed before this run; " if fixed_threshold is not None else
                     "The single quality-cutoff grid is selected on fold0 OOF scores; ") + "intervals condition on these fitted "
                  "predictions and exclude model/threshold-selection uncertainty. No final test calibrator or public/test application."}
    atomic_write(output / "oof_quality.npy", lambda stream: np.save(stream, oof))
    atomic_json(output / "report.json", report)
    atomic_json(output / "per_image.json", {stem: {"r34": base_images[stem], "fixed_conf20": fixed_images[stem],
                                                 "quality_oof": best_images[stem]} for stem in records})
    print(json.dumps({key: value for key, value in report.items() if key != "identity"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-dir", default="outputs/yolo26s_f0/val_e021")
    parser.add_argument("--prob-dir", default="outputs/unet_r34_f0/val_e040_tta4")
    parser.add_argument("--r34-quality", default="outputs/unet_r34_f0/quality_e040_tta4")
    parser.add_argument("--frozen-boundary", default="outputs/yolo26s_f0/val_e021_boundary_frozen")
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default="outputs/diagnostics/yolo_e021_refined_quality")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fixed-threshold", type=float, help="Evaluate only this frozen cutoff, without a grid")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Pool existing within-base-fold OOF evidence; fit a final TEST-only calibrator.

This module NEVER cross-fits a new quality model over mixed base folds. Each
segmentation checkpoint may have trained on the other base folds' labels, so
such a mixed meta-validation would not provide independent held-out evidence.
Only each source's already cross-fitted OOF predictions select the global cut.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .ensemble import atomic_json, atomic_write, checkpoint_provenance, digest_file
from .metrics import PQStats
from .postprocess import PostprocessConfig
from .quality import FEATURES, new_regressor, score_selection


POOL_CONFIG = PostprocessConfig(threshold=.35, min_area=60, close_radius=5).to_dict()
THRESHOLDS = np.arange(.05, .401, .025)


def validate_entries(entries, scores, records, report):
    ids = [entry["image_id"] for entry in entries]
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise ValueError("Quality cache does not contain the complete distinct held-out image set")
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("OOF scores must be finite values in [0,1]")
    count = 0
    for entry in entries:
        record = records[entry["image_id"]]
        x, target = np.asarray(entry["features"]), np.asarray(entry["target"])
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or target.shape != (len(x),):
            raise ValueError(f"Feature/target shape differs for {entry['image_id']}")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(target)) or np.any(target < 0):
            raise ValueError(f"Nonfinite features/targets or negative target for {entry['image_id']}")
        if entry["group"] != record["date_group"]:
            raise ValueError("Cached observation-date group differs from the manifest")
        matrices = entry["iou"]
        if len(matrices) != len(record["annotators"]) or entry["weight"] != len(matrices) or not matrices:
            raise ValueError("Annotator weights or IoU records differ")
        expected_target = np.zeros(len(x), dtype=np.float32)
        for matrix in matrices:
            if matrix.ndim != 2 or matrix.shape[1] != len(x) or not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
                raise ValueError("Invalid per-annotator candidate IoU matrix")
            expected_target += np.where(matrix > .5, matrix, 0).sum(axis=0)
        expected_target /= len(matrices)
        if not np.allclose(target, expected_target, atol=1e-7, rtol=0):
            raise ValueError("Cached regression targets do not match the official IoU rule")
        if sorted(m.shape[0] for m in matrices) != sorted(len(a["rles"]) for a in record["annotators"]):
            raise ValueError("GT instance counts differ from the selected manifest annotations")
        count += len(x)
    if len(scores) != count or report["instances"] != count or report["images"] != len(entries):
        raise ValueError("Feature, OOF, and report row counts differ")
    reproduced = score_selection(entries, scores, report["best"]["quality_threshold"]).to_dict()
    for key in ("tp", "fp", "fn"):
        if reproduced[key] != report["best"][key]:
            raise ValueError("Existing OOF predictions do not reproduce the source report")
    if not np.isclose(reproduced["pq"], report["best"]["pq"], atol=1e-10, rtol=0):
        raise ValueError("Existing OOF PQ does not reproduce the source report")
    return count


def load_sources(quality_dirs, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest_file(manifest_path)
    all_ids, all_groups, all_folds, verified, sources = set(), set(), set(), {}, []
    if len(quality_dirs) < 2:
        raise ValueError("Pooling requires at least two disjoint base folds")
    for directory in quality_dirs:
        directory = Path(directory).resolve()
        cache_path, scores_path, report_path = (directory / name for name in ("features.joblib", "oof_quality.npy", "report.json"))
        cache = joblib.load(cache_path)
        scores = np.load(scores_path, allow_pickle=False)
        report = json.loads(report_path.read_text())
        identity = cache["identity"]
        metadata = identity["inference"]
        fold = identity["fold"]
        if not isinstance(fold, int) or fold < 0 or fold in all_folds:
            raise ValueError("Source base folds must be distinct nonnegative folds")
        if report["base_fold"] != fold or metadata.get("split") != "val" or metadata.get("fold") != fold:
            raise ValueError("Cache/report/inference held-out fold provenance differs")
        if identity["annotation_sha256"] != manifest["annotation_sha256"]:
            raise ValueError("Source annotation SHA differs from the common manifest")
        if "manifest_sha256" in metadata and metadata["manifest_sha256"] != manifest_hash:
            raise ValueError("Source manifest SHA differs from the common manifest")
        if identity["config"] != POOL_CONFIG or report["pool_config"] != POOL_CONFIG:
            raise ValueError("All sources must use the common .35/60/close5 candidate pool")
        if report["features"] != FEATURES:
            raise ValueError("Source handcrafted feature names/order differ")
        records = {r["image_id"]: r for r in manifest["train"] if r["fold"] == fold}
        entries = cache["entries"]
        count = validate_entries(entries, scores, records, report)
        ids, groups = set(records), {entry["group"] for entry in entries}
        if ids & all_ids or groups & all_groups:
            raise ValueError("Base folds overlap in image IDs or observation-date groups")
        provenance = checkpoint_provenance(metadata, directory, manifest, verified)
        source_model = directory / "quality_model.joblib"
        if source_model.exists():
            saved = joblib.load(source_model)
            if saved["model"].get_params() != new_regressor().get_params():
                raise ValueError("Source quality regressor hyperparameters differ")
        source_identity = {
            "quality_dir": str(directory), "fold": fold, "cache_identity": identity,
            "features_sha256": digest_file(cache_path), "oof_scores_sha256": digest_file(scores_path),
            "report_sha256": digest_file(report_path), "checkpoint_provenance": provenance,
            "images": len(entries), "date_groups": len(groups), "instances": count,
        }
        sources.append({"entries": entries, "scores": scores, "report": report, "identity": source_identity})
        all_ids.update(ids); all_groups.update(groups); all_folds.add(fold)
        print(json.dumps({"source_fold": fold, "images": len(entries), "date_groups": len(groups), "instances": count,
                          "original_best": report["best"]}), flush=True)
    identity = {"kind": "pooled_final_quality_for_test_only", "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash, "annotation_sha256": manifest["annotation_sha256"],
                "pool_config": POOL_CONFIG, "features": FEATURES,
                "source_folds": sorted(all_folds), "sources": [s["identity"] for s in sources],
                "regressor_params": new_regressor().get_params(),
                "thresholds": THRESHOLDS.tolist(),
                "threshold_evidence": "Existing within-base-fold GroupKFold OOF only; no pooled meta-level crossfit"}
    return sources, identity


def aggregate_oof(sources):
    ranking = []
    for threshold in THRESHOLDS:
        total = PQStats()
        per_fold = {}
        for source in sources:
            score = score_selection(source["entries"], source["scores"], threshold)
            total += score
            per_fold[str(source["identity"]["fold"])] = score.to_dict()
        ranking.append({"quality_threshold": float(threshold), **total.to_dict(), "per_fold": per_fold})
    return sorted(ranking, key=lambda row: row["pq"], reverse=True)


def pool_quality(quality_dirs, output_dir, manifest_path="data/cache/manifest.json"):
    sources, identity = load_sources(quality_dirs, manifest_path)
    output = Path(output_dir).resolve()
    if output in {Path(path).resolve() for path in quality_dirs}:
        raise ValueError("Use an output directory separate from all source calibrators")
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "pool_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing pooled output has a different source identity")
    atomic_json(identity_path, identity)
    ranking = aggregate_oof(sources)
    best = ranking[0]
    entries = [entry for source in sources for entry in source["entries"]]
    original_scores = np.concatenate([source["scores"] for source in sources])
    x = np.concatenate([entry["features"] for entry in entries])
    y = np.concatenate([entry["target"] for entry in entries])
    weights = np.concatenate([np.full(len(entry["target"]), entry["weight"]) for entry in entries])
    source_rows, offset = [], 0
    for source in sources:
        for entry in source["entries"]:
            count = len(entry["target"])
            source_rows.append({"base_fold": source["identity"]["fold"], "image_id": entry["image_id"],
                                "date_group": entry["group"], "start": offset, "stop": offset+count})
            offset += count
    report = {
        "kind": "pooled_final_quality_for_test_only", "base_folds": identity["source_folds"],
        "images": len(entries), "instances": len(x), "date_groups": len({entry["group"] for entry in entries}),
        "features": FEATURES, "pool_config": POOL_CONFIG, "best": best, "ranking": ranking,
        "per_fold_at_global_threshold": best["per_fold"],
        "source_best_thresholds": {str(source["identity"]["fold"]): source["report"]["best"] for source in sources},
        "selected_instances_at_global_threshold": int((original_scores >= best["quality_threshold"]).sum()),
        "evaluation": "Summed official IoU/TP/FP/FN from each source's EXISTING independently date-cross-fitted OOF scores. Global threshold selected on these aggregated OOF scores.",
        "final_model_validation": "The final pooled calibrator has NO new held-out score. Base segmentation models trained on other base folds' labels, so a new crossfit across the combined features would have indirect leakage and was NOT performed.",
        "intended_use": "TEST-only application after explicit model selection; no test application or submission performed by this module.",
    }
    with threadpool_limits(limits=4):
        model = new_regressor().fit(x, y, sample_weight=weights)
    atomic_write(output / "quality_model.joblib", lambda stream: joblib.dump({"model": model, "report": report, "identity": identity}, stream))
    atomic_write(output / "aggregated_source_oof.npy", lambda stream: np.save(stream, original_scores))
    atomic_json(output / "oof_source_index.json", source_rows)
    atomic_json(output / "report.json", report)
    print(json.dumps({"images": len(entries), "instances": len(x), "base_folds": report["base_folds"],
                      "best_existing_oof_aggregate": best, "output": str(output)}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/r34_quality_pooled")
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    args = parser.parse_args()
    pool_quality(args.quality_dirs, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()

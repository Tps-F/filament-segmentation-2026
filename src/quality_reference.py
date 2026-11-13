"""Validated fixed date-group partitions for comparing quality models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .ensemble import checkpoint_provenance, digest_file


def reference_splits(entries, group_folds):
    """Apply a frozen map regardless of the current candidate counts/order."""
    groups = {str(entry["group"]) for entry in entries}
    if groups != set(group_folds) or set(group_folds.values()) != set(range(5)):
        raise ValueError("Fixed crossfit map must cover exactly these date groups and five folds")
    if any(type(value) is not int for value in group_folds.values()):
        raise ValueError("Fixed crossfit fold assignments must be integers")
    folds = np.concatenate([np.full(len(entry["target"]), group_folds[str(entry["group"])], dtype=np.int64)
                            for entry in entries])
    splits = []
    for fold in range(5):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(train) or not len(valid):
            raise ValueError("Each fixed meta-fold needs at least one training and validation candidate")
        splits.append((train, valid))
    return splits


def load_reference(directory, manifest_path, fold, records, target_metadata, target_prob_dir):
    # Runtime imports avoid a quality -> quality_pool -> quality import cycle.
    from .candidate_fusion import original_group_folds
    from .quality import FEATURES
    from .quality_pool import validate_entries

    directory = Path(directory).resolve()
    manifest_path = Path(manifest_path).resolve()
    target_prob_dir = Path(target_prob_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    record_map = {record["image_id"]: record for record in records}
    expected = {record["image_id"]: record for record in manifest["train"] if record["fold"] == fold}
    if fold < 0 or not expected or record_map != expected or len(record_map) != len(records):
        raise ValueError("Crossfit reference requires the exact distinct held-out image records")
    hashes = {name: digest_file(directory / name) for name in ("features.joblib", "oof_quality.npy", "report.json")}
    cache = joblib.load(directory / "features.joblib")
    identity, entries = cache["identity"], cache["entries"]
    report = json.loads((directory / "report.json").read_text())
    scores = np.load(directory / "oof_quality.npy", allow_pickle=False)
    metadata = identity["inference"]
    if (identity["fold"] != fold or report["base_fold"] != fold or metadata.get("split") != "val"
            or metadata.get("fold") != fold or target_metadata.get("split") != "val"
            or target_metadata.get("fold") != fold):
        raise ValueError("Reference and target must hold out the same base fold")
    if (identity["annotation_sha256"] != manifest["annotation_sha256"]
            or report["features"] != FEATURES or report["pool_config"] != identity["config"]):
        raise ValueError("Reference annotation or feature/pool provenance differs")
    manifest_sha = digest_file(manifest_path)
    for inference in (metadata, target_metadata):
        if inference.get("manifest_sha256", manifest_sha) != manifest_sha:
            raise ValueError("Reference or target inference uses a different manifest")
    annotation_path = Path(manifest["annotation_path"])
    if not annotation_path.is_absolute():
        annotation_path = manifest_path.parent / annotation_path
    if digest_file(annotation_path) != manifest["annotation_sha256"]:
        raise ValueError("Official annotation fingerprint changed")
    validate_entries(entries, scores, record_map, report)
    if {path.stem for path in target_prob_dir.glob("*.npz")} != set(record_map):
        raise ValueError("Target probability cache must contain exactly the held-out image set")
    verified = {}
    source_provenance = checkpoint_provenance(metadata, directory, manifest, verified)
    target_provenance = checkpoint_provenance(target_metadata, target_prob_dir, manifest, verified)
    # A previously fixed-reference run must preserve its stored map, rather
    # than deriving a different partition from that run's candidate counts.
    previous = identity.get("crossfit_reference")
    if previous is None:
        group_folds = original_group_folds(entries)
        assignment_origin = "original GroupKFold(5), reference candidate counts only"
    else:
        if report.get("crossfit_reference") != previous:
            raise ValueError("Stored reference crossfit identity/report differ")
        group_folds = previous["group_folds"]
        assignment_origin = "preserved frozen map from reference identity"
    reference_splits(entries, group_folds)
    image_folds = {stem: group_folds[str(record["date_group"])] for stem, record in sorted(record_map.items())}
    if previous is not None and previous["image_folds"] != image_folds:
        raise ValueError("Stored reference image assignments disagree with current date groups")
    for name, expected_sha in hashes.items():
        if digest_file(directory / name) != expected_sha:
            raise ValueError("Reference artifacts changed while being validated")
    return {"directory": str(directory), "artifact_sha256": hashes, "base_fold": fold,
        "manifest_sha256": manifest_sha, "annotation_sha256": manifest["annotation_sha256"],
        "group_folds": group_folds, "image_folds": image_folds,
        "image_date_groups": {stem: record["date_group"] for stem, record in sorted(record_map.items())},
        "reference_checkpoint_provenance": source_provenance,
        "target_checkpoint_provenance": target_provenance, "assignment_origin": assignment_origin,
        "reference_usage": "Image/date identities and candidate counts define partitions. Reference labels and OOF scores "
            "are read only for integrity checks, never used as target-model features or training targets."}

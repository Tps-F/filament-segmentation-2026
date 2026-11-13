"""CPU weighted means of complete native-resolution probability caches.

Example:
    python -m src.ensemble --prob-dirs outputs/r34/val outputs/b3/val \
        --weights 1 1 --output-dir outputs/ensemble/val

Sources must hold out the same validation fold, or all be test predictions.
A matching output directory resumes; any change to sources, metadata, weights,
or the manifest requires a fresh output directory. No models are instantiated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np


NATIVE_SHAPE = (2048, 2048)


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path, writer):
    """Only publish a complete file; temporary files live on the same filesystem."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2) + "\n").encode()))


def data_signature(manifest, fold):
    """Same signature as src.train, including all-data training with fold -1."""
    split = {
        "annotation_sha256": manifest.get("annotation_sha256"),
        "manifest_version": manifest.get("version"),
        "fold_grouping": manifest.get("fold_grouping"),
        "validation_fold": fold,
        "training_images": sorted(r["image_id"] for r in manifest["train"] if fold < 0 or r["fold"] != fold),
        "validation_images": sorted(r["image_id"] for r in manifest["train"] if fold >= 0 and r["fold"] == fold),
    }
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def resolve_checkpoint(raw_path, source_dir):
    path = Path(raw_path)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, Path(__file__).resolve().parents[1] / path, source_dir / path]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError(f"Checkpoint path is missing or ambiguous: {raw_path}")
    return found.pop()


def checkpoint_provenance(metadata, source_dir, manifest, verified):
    paths, fingerprints = metadata.get("checkpoint", []), metadata.get("checkpoint_fingerprints", [])
    if not isinstance(paths, list) or not paths or len(paths) != len(fingerprints):
        raise ValueError(f"Missing checkpoint/fingerprint provenance in {source_dir}")
    recorded_provenance = metadata.get("validation_provenance")
    if metadata["split"] == "val" and recorded_provenance and len(recorded_provenance) != len(paths):
        raise ValueError(f"Checkpoint validation provenance length differs in {source_dir}")
    provenance = []
    for index, (raw_path, fingerprint) in enumerate(zip(paths, fingerprints)):
        path = resolve_checkpoint(raw_path, source_dir)
        key = (str(path), fingerprint)
        if key not in verified:
            if digest_file(path) != fingerprint:
                raise ValueError(f"Checkpoint fingerprint differs: {path}")
            # CPU metadata access only: mmap avoids materializing model/optimizer tensors.
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            config = checkpoint.get("config", {})
            fold = config.get("fold")
            if not isinstance(fold, int) or fold < -1:
                raise ValueError(f"Checkpoint has no valid training fold: {path}")
            signature = config.get("data_signature")
            if signature is not None and signature != data_signature(manifest, fold):
                raise ValueError(f"Checkpoint annotation/train/validation signature differs: {path}")
            if signature is None:
                warnings.warn(f"Legacy checkpoint {path.name}: fold verified; historical data membership has no signature", stacklevel=2)
            verified[key] = {"checkpoint": str(path), "fingerprint": fingerprint, "fold": fold,
                             "data_signature": signature,
                             "verification": "data_signature" if signature is not None else "legacy_fold_only"}
            del checkpoint
        record = verified[key]
        if metadata["split"] == "val":
            if record["fold"] != metadata.get("fold"):
                raise ValueError(f"Checkpoint does not hold out the requested validation fold: {path}")
            if recorded_provenance:
                prior = recorded_provenance[index]
                if any(prior.get(k) != record[k] for k in ("fold", "data_signature", "verification")):
                    raise ValueError(f"Recorded checkpoint validation provenance differs: {path}")
        provenance.append(record)
    return provenance


def prepare(prob_dirs, weights, output_dir, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    source_dirs = [Path(path).resolve() for path in prob_dirs]
    output_dir = Path(output_dir).resolve()
    if len(set(source_dirs)) != len(source_dirs) or output_dir in source_dirs:
        raise ValueError("Use distinct source directories and a separate output directory")
    weights = np.asarray(weights if weights is not None else [1.] * len(source_dirs), dtype=np.float64)
    if not source_dirs or len(weights) != len(source_dirs) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Provide one finite positive weight per source")
    # Scale first so even large finite weights do not overflow the normalization.
    normalized = weights / weights.max()
    normalized /= normalized.sum()
    sources = []
    for directory, weight in zip(source_dirs, normalized):
        raw_metadata = (directory / "inference_config.json").read_bytes()
        metadata = json.loads(raw_metadata)
        if metadata.get("split") not in ("val", "test"):
            raise ValueError(f"Only held-out val and test caches are supported: {directory}")
        if "manifest_sha256" in metadata and metadata["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Source manifest differs from current manifest: {directory}")
        sources.append({"prob_dir": str(directory), "weight": float(weight), "metadata": metadata,
                        "metadata_sha256": hashlib.sha256(raw_metadata).hexdigest()})
    split = sources[0]["metadata"]["split"]
    folds = [s["metadata"].get("fold") for s in sources]
    if any(s["metadata"]["split"] != split for s in sources):
        raise ValueError("Cannot ensemble different data splits")
    if split == "val" and (not isinstance(folds[0], int) or folds[0] < 0 or any(fold != folds[0] for fold in folds)):
        raise ValueError("All validation caches must use the same held-out fold")
    fold = folds[0] if all(value == folds[0] for value in folds) else None
    entries = manifest["test"] if split == "test" else [r for r in manifest["train"] if r["fold"] == fold]
    ids = sorted(r["image_id"] for r in entries)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest has no selected images or duplicate selected image IDs")
    verified, checkpoints = {}, []
    for source in sources:
        directory = Path(source["prob_dir"])
        available = {path.stem for path in directory.glob("*.npz")}
        if available != set(ids):
            raise ValueError(f"Source must contain exactly {len(ids)} selected images: {directory}; "
                             f"missing={sorted(set(ids) - available)[:5]}, extra={sorted(available - set(ids))[:5]}")
        provenance = checkpoint_provenance(source["metadata"], directory, manifest, verified)
        source["checkpoint_provenance"] = provenance
        checkpoints.extend(provenance)
        source["probability_sha256"] = {stem: digest_file(directory / (stem + ".npz")) for stem in ids}
    metadata = {"schema_version": 1, "kind": "probability_ensemble", "split": split, "fold": fold,
                "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
                "checkpoint": [r["checkpoint"] for r in checkpoints],
                "checkpoint_fingerprints": [r["fingerprint"] for r in checkpoints],
                "validation_provenance": [{k: r[k] for k in ("fold", "data_signature", "verification")}
                                          for r in checkpoints] if split == "val" else [],
                "ensemble": {"method": "weighted_probability_mean", "weights": normalized.tolist(),
                             "requested_weights": weights.tolist(), "output_dtype": "float16",
                             "accumulation_dtype": "float32", "native_shape": list(NATIVE_SHAPE),
                             "image_ids": ids, "sources": sources}}
    return metadata, output_dir


def load_probability(path, expected_digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"Source probability changed after preflight: {path}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        probability = archive["probability"].astype(np.float32)
    if probability.ndim != 3 or probability.shape[0] not in (1, 2, 3, 4) or probability.shape[-2:] != NATIVE_SHAPE:
        raise ValueError(f"Expected native Cx2048x2048 probabilities, got {probability.shape}: {path}")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"Probability contains nonfinite or out-of-range values: {path}")
    return probability


def ensemble(prob_dirs, weights, output_dir, manifest_path="data/cache/manifest.json"):
    started = time.monotonic()
    metadata, output_dir = prepare(prob_dirs, weights, output_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "inference_config.json"
    ids = metadata["ensemble"]["image_ids"]
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Output cache identity differs; use a fresh output directory")
    elif list(output_dir.glob("*.npz")) or (output_dir / "ensemble_state.json").exists():
        raise ValueError("Existing output files have no matching provenance; use a fresh output directory")
    else:
        atomic_json(metadata_path, metadata)
    extra = {p.stem for p in output_dir.glob("*.npz")} - set(ids)
    if extra:
        raise ValueError(f"Output cache contains unexpected image files: {sorted(extra)[:5]}")
    identity = canonical_digest(metadata)
    state_path = output_dir / "ensemble_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"identity_sha256": identity, "completed": {}}
    if state.get("identity_sha256") != identity or set(state["completed"]) - set(ids):
        raise ValueError("Output progress state belongs to a different ensemble")
    print(json.dumps({"event": "start", "images": len(ids), "sources": len(prob_dirs),
                      "weights": metadata["ensemble"]["weights"], "split": metadata["split"], "fold": metadata["fold"]}), flush=True)
    for index, stem in enumerate(ids, start=1):
        destination = output_dir / (stem + ".npz")
        cached = state["completed"].get(stem)
        resumed = bool(cached and destination.exists() and digest_file(destination) == cached)
        if not resumed:
            result = None
            for source in metadata["ensemble"]["sources"]:
                probability = load_probability(Path(source["prob_dir"]) / (stem + ".npz"), source["probability_sha256"][stem])
                if result is not None and probability.shape != result.shape:
                    raise ValueError(f"Source probability channel/shape mismatch for {stem}")
                probability *= source["weight"]
                if result is None:
                    result = probability
                else:
                    result += probability
            atomic_write(destination, lambda stream: np.savez_compressed(stream, probability=np.clip(result, 0, 1).astype(np.float16)))
            state["completed"][stem] = digest_file(destination)
            atomic_json(state_path, state)
        print(json.dumps({"image": index, "total": len(ids), "id": stem, "resumed": resumed,
                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-dirs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    args = parser.parse_args()
    ensemble(args.prob_dirs, args.weights, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()

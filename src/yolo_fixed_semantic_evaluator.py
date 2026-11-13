"""Explicit-path fixed YOLO/equal-semantic evaluation, for reviewed CPU use.

The original R34 source validator is called on the actual R34 cache. Ensemble
provenance is validated separately; no frozen module constants are replaced.
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
import joblib
import numpy as np
from pycocotools import mask as mu

from .candidate_fusion import annotation_rles, arbitrary_features, crossfit, original_group_folds, rle_iou
from .ensemble import atomic_json, atomic_write, canonical_digest, digest_file, load_probability, prepare as ensemble_identity
from .inference import write_submission
from .metrics import PQStats, load_ground_truth, read_submission, score_image
from .postprocess import PostprocessConfig, probabilities_to_labels
from .quality import FEATURES, new_regressor
from .quality_pool import POOL_CONFIG, validate_entries
from .yolo_boundary import PARAMETERS, paired_bootstrap, refine_detections, validate_sources
from .yolo_refined_quality import QUALITY_FEATURES, evaluate, proposal_diagnostics
from .yolo_semantic_upgrade import assert_metric, assert_weighted_pixels, fold_contract


def cpu_guard():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Explicitly hide CUDA for fixed CPU evaluation')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


def generic_geometry_features(job):
    """Pure geometry/features. Job contains no annotation/target/OOF score."""
    cv2.setNumThreads(1)
    stem, raw_path, probability_path, image_path, expected_hash, sources = job
    raw = json.loads(Path(raw_path).read_text())
    probability = load_probability(probability_path, expected_hash)
    arrays = [load_probability(Path(s['prob_dir']) / f'{stem}.npz', s['probability_sha256'][stem]) for s in sources]
    assert_weighted_pixels(arrays, [s['weight'] for s in sources], probability.astype(np.float16))
    del arrays
    transformed = refine_detections(raw, probability[0], confidence_floor=.03)
    proposals = transformed['post_nms']
    fixed = refine_detections(raw, probability[0], confidence_floor=.20)['post_nms']
    if [d for d in proposals if d['score'] >= .20] != fixed:
        raise ValueError('Low-floor geometry differs from fixed confidence-.20')
    labels = probabilities_to_labels(probability, PostprocessConfig(**POOL_CONFIG))
    image = np.load(image_path, allow_pickle=False)
    features = np.asarray([arbitrary_features(d['rle'], probability, image, labels) for d in proposals],
                          dtype=np.float32).reshape(-1, len(FEATURES))
    confidence = np.asarray([d['score'] for d in proposals], dtype=np.float32)
    features = np.concatenate([features, confidence[:, None]], axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite quality features')
    return {'image_id': stem, 'features': features, 'rles': [d['rle'] for d in proposals],
        'source_indices': [d['source_index'] for d in proposals], 'confidence': confidence,
        'raw_count': len(raw), 'dropped_area_lt5': transformed['dropped_area_lt5'],
        'dropped_nms': transformed['dropped_nms']}


def native_audit(csv_path, records, annotation_path, *, entries=None, scores=None, threshold=None, raw_dir=None):
    """Organizer-GT scoring, selection identity, RLE and optional R12 audit."""
    truth = load_ground_truth(annotation_path)
    predictions = read_submission(csv_path)
    if set(predictions) - set(records):
        raise ValueError('CSV contains non-held-out images')
    selected = {}
    if entries is not None:
        offset = 0
        for entry in entries:
            count = len(entry['rles'])
            keep = scores[offset:offset + count] >= threshold
            offset += count
            selected[entry['image_id']] = [(r, i) for r, i, k in zip(entry['rles'], entry['source_indices'], keep) if k]
        if offset != len(scores):
            raise ValueError('Native audit score alignment differs')
    total, rows = PQStats(), {}
    overlap = {'instances': 0, 'images_with_pixel_overlap': 0, 'overlapping_mask_pairs': 0,
               'excess_mask_coverage_pixels': 0, 'total_mask_pixels': 0}
    source_ids = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for stem in sorted(records):
        rles = predictions.get(stem, [])
        if entries is not None:
            expected = selected[stem]
            if [r['counts'].decode() for r in rles] != [r['counts'] for r, _ in expected]:
                raise ValueError('Exported native CSV changed fixed selection')
            source_ids[stem] = [i for _, i in expected]
            if len(source_ids[stem]) != len(set(source_ids[stem])):
                raise ValueError('Duplicate original object identity')
        raw = json.loads((Path(raw_dir) / f'{stem}.json').read_text()) if raw_dir is not None else None
        for index, rle in enumerate(rles):
            mask = mu.decode(rle)
            if mask.shape != (2048, 2048) or mask.sum() < 5 or not np.array_equal(
                    mask, mu.decode(mu.encode(np.asfortranarray(mask)))):
                raise ValueError('Invalid native mask or failed RLE roundtrip')
            if raw is not None:
                original = mu.decode(raw[source_ids[stem][index]]['rle'])
                support = cv2.dilate(original, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
                if np.any(mask & ~support):
                    raise ValueError('Refined mask escaped original R12 support')
        stat = score_image(truth[stem], rles)
        total += stat
        rows[stem] = stat.to_dict()
        area = int(np.asarray(mu.area(rles)).sum()) if rles else 0
        union = int(mu.area(mu.merge(rles))) if rles else 0
        overlap['instances'] += len(rles)
        overlap['images_with_pixel_overlap'] += int(area > union)
        overlap['excess_mask_coverage_pixels'] += area - union
        overlap['total_mask_pixels'] += area
        if len(rles) > 1:
            pair = mu.iou(rles, rles, [0] * len(rles))
            if np.triu(pair, 1).max() > .5:
                raise ValueError('Mask NMS left a duplicate pair above.5')
            overlap['overlapping_mask_pairs'] += int(np.triu(pair > 0, 1).sum())
    overlap['pixel_disjoint'] = overlap['excess_mask_coverage_pixels'] == 0
    return {'score': total.to_dict(), 'per_image': rows, 'overlap': overlap, 'source_ids': source_ids}


def run(plan_path):
    cpu_guard()
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh fixed-evaluation output directory')
    manifest_path, yolo_dir, r34_dir, semantic_dir, quality_dir = map(Path,
        [plan['manifest'], plan['yolo_dir'], plan['r34_prob_dir'], plan['semantic_prob_dir'], plan['r34_quality_dir']])
    if plan['fold'] != 0 or plan['quality_cutoff'] != .20 or plan['semantic_weights'] != [.5, .5]:
        raise ValueError('Only the predeclared fold0/equal/.20 comparison is supported')
    records, annotation_path, original = validate_sources(yolo_dir, r34_dir, manifest_path.resolve(), 0)
    if original['yolo']['checkpoint_sha256'] != plan['checkpoint_sha256'] or original['yolo']['raw_confidence'] != .03:
        raise ValueError('Actual detector differs from reviewed source')
    boundary = Path(plan['r34_boundary_dir'])
    if json.loads((boundary / 'identity.json').read_text()) != original:
        raise ValueError('Actual original R34 boundary source differs')
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((semantic_dir / 'inference_config.json').read_text())
    if (metadata['split'] != 'val' or metadata['fold'] != 0 or metadata['ensemble']['weights'] != [.5, .5]
            or metadata['checkpoint_fingerprints'] != plan['semantic_checkpoint_sha256']):
        raise ValueError('Wrong equal-ensemble source')
    rebuilt, _ = ensemble_identity([s['prob_dir'] for s in metadata['ensemble']['sources']],
        metadata['ensemble']['requested_weights'], semantic_dir, manifest_path)
    if rebuilt != metadata or any(v['verification'] != 'data_signature' for v in metadata['validation_provenance']):
        raise ValueError('Equal-ensemble source provenance did not replay')
    state = json.loads((semantic_dir / 'ensemble_state.json').read_text())
    if (state['identity_sha256'] != canonical_digest(metadata) or set(state['completed']) != set(records)
            or {p.stem for p in semantic_dir.glob('*.npz')} != set(records)):
        raise ValueError('Incomplete semantic cache')
    base = joblib.load(quality_dir / 'features.joblib')
    base_oof = np.load(quality_dir / 'oof_quality.npy', allow_pickle=False)
    base_report = json.loads((quality_dir / 'report.json').read_text())
    validate_entries(base['entries'], base_oof, records, base_report)
    if base['identity']['inference'] != original['r34'] or base['identity']['config'] != POOL_CONFIG:
        raise ValueError('Original R34 meta-fold reference differs')
    groups = original_group_folds(base['entries'])
    if groups != plan['group_folds']:
        raise ValueError('Actual original R34 meta-fold map differs from predeclaration')
    for path, fingerprint in plan['fixed_file_sha256'].items():
        if digest_file(path) != fingerprint:
            raise ValueError(f'Fixed input changed: {path}')
    identity = {'plan': plan, 'plan_sha256': digest_file(plan_path), 'source_sha256': digest_file(__file__),
        'original_source_identity': original, 'semantic_metadata': metadata, 'semantic_hashes': state['completed'],
        'features': QUALITY_FEATURES, 'regressor_params': new_regressor().get_params(),
        'geometry': PARAMETERS, 'raw_floor': .03, 'quality_thresholds': [.20],
        'interpretation': 'Fixed mature-M fold0 development evaluation. No final calibrator, TEST or threshold search.'}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'predeclaration.json', identity)
    jobs = []
    for base_entry in base['entries']:
        stem = base_entry['image_id']
        image = Path(records[stem]['image_path'])
        image = image if image.is_absolute() else manifest_path.parent / image
        jobs.append((stem, yolo_dir / 'raw' / f'{stem}.json', semantic_dir / f'{stem}.npz', image,
                     state['completed'][stem], metadata['ensemble']['sources']))
    entries = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        for index, entry in enumerate(pool.map(generic_geometry_features, jobs), 1):
            entries.append(entry)
            if index % 20 == 0 or index == len(jobs):
                print(json.dumps({'geometry_images': index, 'total': len(jobs)}), flush=True)
    atomic_write(output / 'geometry_features.joblib', lambda stream: joblib.dump(entries, stream))
    # Annotation targets are attached only after all geometry/features are sealed.
    for entry in entries:
        record = records[entry['image_id']]
        matrices = [rle_iou(entry['rles'], annotation_rles(a)).T for a in record['annotators']]
        entry.update(group=record['date_group'], weight=len(matrices), iou=matrices,
            target=(sum(np.where(m > .5, m, 0).sum(axis=0) for m in matrices) / len(matrices)).astype(np.float32))
    if sum(len(e['target']) for e in entries) > 5000:
        raise ValueError('Fixed5000-candidate budget exceeded')
    atomic_write(output / 'features.joblib', lambda stream: joblib.dump({'identity': identity, 'entries': entries}, stream))
    atomic_json(output / 'fold_contract.json', fold_contract(entries, groups))
    confidence = np.concatenate([e['confidence'] for e in entries])
    atomic_json(output / 'fit_started.json', {'unix_time': time.time(), 'threshold': .20, 'plan_sha256': digest_file(plan_path)})
    oof = crossfit(entries, groups)
    reports, rows = {}, {}
    for name, scores in [('boundary_conf20', confidence), ('quality_q20', oof)]:
        predictions, offset = [], 0
        for entry in entries:
            n = len(entry['rles'])
            keep = scores[offset:offset + n] >= .20
            offset += n
            predictions.append((entry['image_id'], [r for r, k in zip(entry['rles'], keep) if k]))
        csv_path = output / f'{name}.csv'
        export = write_submission(predictions, csv_path, records)
        audit = native_audit(csv_path, records, annotation_path, entries=entries, scores=scores,
                             threshold=.20, raw_dir=yolo_dir / 'raw')
        cached, _ = evaluate(entries, scores, .20)
        assert_metric(audit['score'], cached)
        rows[name] = audit.pop('per_image')
        atomic_json(output / f'{name}_source_ids.json', audit.pop('source_ids'))
        reports[name] = {**audit, 'export': {**export, 'sha256': digest_file(csv_path)}}
    for stem in records:
        if (digest_file(yolo_dir / 'raw' / f'{stem}.json') != original['raw_sha256'][stem]
                or digest_file(semantic_dir / f'{stem}.npz') != state['completed'][stem]):
            raise ValueError('Prediction source changed while fitting')
    for path, fingerprint in plan['fixed_file_sha256'].items():
        if digest_file(path) != fingerprint:
            raise ValueError('Fixed input changed during fitting')
    cpu_guard()
    atomic_write(output / 'oof_quality.npy', lambda stream: np.save(stream, oof))
    report = {'identity': identity, 'scores': reports, 'pool_count': len(oof), 'images': len(entries),
        'quality_vs_boundary': paired_bootstrap(rows['boundary_conf20'], rows['quality_q20'],
                                               {e['image_id']: e['group'] for e in entries}),
        'proposal_diagnostics': proposal_diagnostics(entries, confidence, oof, .20),
        'checks': {'native_csv_exact': True, 'source_id_R12_RLE': True, 'source_hashes_unchanged': True,
                   'quality_cutoff_grid_used': False, 'final_model_fitted': False, 'cuda_initialized': False}}
    atomic_json(output / 'per_image.json', rows)
    atomic_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'identity'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    run(args.plan)

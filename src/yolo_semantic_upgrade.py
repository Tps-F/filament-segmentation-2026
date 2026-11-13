"""One fixed CPU development replacement of YOLO's R34 semantic source.

Preparation verifies the original untouched R34 validator and baselines, then
the equal B5/ConvNeXt ensemble. Fitting requires a separate exact-hash review.
No TEST, predictor, final calibrator, cutoff search or submission is supported.
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
from .ensemble import atomic_json, atomic_write, canonical_digest, digest_file, load_probability, prepare as verify_ensemble
from .inference import write_submission
from .metrics import PQStats, read_submission, score_rles
from .postprocess import PostprocessConfig, labels_to_rles, probabilities_to_labels
from .quality import FEATURES, new_regressor, score_selection
from .quality_pool import POOL_CONFIG, validate_entries
from .yolo_boundary import PARAMETERS, paired_bootstrap, refine_detections, validate_sources
from .yolo_refined_quality import QUALITY_FEATURES, evaluate, proposal_diagnostics


MANIFEST = Path('data/cache/manifest.json')
YOLO = Path('outputs/yolo26s_f0/val_e041')
R34 = Path('outputs/unet_r34_f0/val_e040_tta4')
R34_QUALITY = Path('outputs/unet_r34_f0/quality_e040_tta4')
ENSEMBLE = Path('outputs/ensembles/b5_convnext_e040_equal')
ENSEMBLE_QUALITY = Path('outputs/ensembles/b5_convnext_e040_equal_quality_fixed_r34')
ENSEMBLE_AUDIT = Path('outputs/diagnostics/b5_convnext_e040_equal_fixed_audit')
OLD = Path('outputs/diagnostics/yolo_e041_fixed_evaluation')
OUTPUT = Path('outputs/diagnostics/yolo_e041_semantic_upgrade_fixed')
YOLO_SHA = 'c5e278be3c54b95ac59dbd0e49bb852e63d4596999fd28e9295c6f97258db041'
SEMANTIC_SHAS = ['dcad6ee94d7694ac45c9e94df02f028d75c7e502406c938ea3ebc52e4206fad8',
                 'c442d7f1f4346a9181ec9c3039944e4ca3aa7651e205780bcf8e00d66ea47303']
SPEC = {'semantic_source': 'equal B5 + ConvNeXtTiny e40, both outer-fold0/native/TTA4',
        'semantic_weight': .75, 'gaussian_weight': .25, 'geometry': PARAMETERS,
        'raw_pool_floor': .03, 'fixed_confidence_baseline': .20, 'quality_cutoff': .20,
        'features': QUALITY_FEATURES, 'context_pool': POOL_CONFIG,
        'regressor': new_regressor().get_params(), 'workers': 4,
        'change': 'Replace R34 foreground, bridge and context-label features with equal ensemble; all other recipe choices frozen.',
        'interpretation': 'Follow-up development selected after e41 results, not independent confirmation. No grid, final model or TEST.'}


def cpu_only():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Hide CUDA explicitly for this CPU-only experiment')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


def assert_metric(actual, expected):
    if any(actual[k] != expected[k] for k in ('tp', 'fp', 'fn')) or not np.isclose(
            actual['iou_sum'], expected['iou_sum'], atol=1e-9, rtol=0) or not np.isclose(
            actual['pq'], expected['pq'], atol=1e-12, rtol=0):
        raise ValueError('Exact native metric replay differs')


def rounded_matrix_multiset(matrices):
    """Annotator/instance ordering is irrelevant; every float32 IoU must agree."""
    return sorted((matrix.shape, tuple(sorted(row.tobytes() for row in np.asarray(matrix, np.float32))))
                  for matrix in matrices)


def rescore(path, records):
    predictions = read_submission(path)
    if set(predictions) - set(records):
        raise ValueError('CSV contains a non-held-out image')
    total, rows = PQStats(), {}
    for stem, record in sorted(records.items()):
        stat = PQStats()
        for annotation in record['annotators']:
            stat += score_rles(annotation_rles(annotation), predictions.get(stem, []))
        total += stat
        rows[stem] = stat.to_dict()
    return total.to_dict(), rows


def fold_contract(entries, group_folds):
    ids = [entry['image_id'] for entry in entries]
    if len(ids) != len(set(ids)) or {str(e['group']) for e in entries} != set(group_folds):
        raise ValueError('Duplicate images or missing/extra date-group assignments')
    audit = []
    for fold in range(5):
        train = [e for e in entries if group_folds[str(e['group'])] != fold]
        valid = [e for e in entries if group_folds[str(e['group'])] == fold]
        train_groups, valid_groups = ({e['group'] for e in es} for es in (train, valid))
        if not train or not valid or train_groups & valid_groups:
            raise ValueError('Invalid/leaking date meta-fold')
        audit.append({'fold': fold, 'training_images': [e['image_id'] for e in train],
            'heldout_images': [e['image_id'] for e in valid],
            'training_groups': sorted(train_groups), 'heldout_groups': sorted(valid_groups)})
    return audit


def assert_weighted_pixels(source_arrays, weights, cached):
    result = None
    for array, weight in zip(source_arrays, weights):
        if array.shape != cached.shape or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError('Invalid semantic source probability')
        weighted = array.astype(np.float32) * weight
        result = weighted if result is None else result + weighted
    if not np.array_equal(np.clip(result, 0, 1).astype(np.float16), cached):
        raise ValueError('Ensemble pixels do not exactly reproduce its source mean')


def geometry_features(job):
    """Annotation-free worker: inputs are predictions, pixels and source hashes."""
    stem, image_path, metadata, ensemble_sha, expected_r34 = job
    cv2.setNumThreads(1)
    raw = json.loads((YOLO / 'raw' / f'{stem}.json').read_text())
    probability = load_probability(ENSEMBLE / f'{stem}.npz', ensemble_sha)
    sources = metadata['ensemble']['sources']
    arrays = [load_probability(Path(s['prob_dir']) / f'{stem}.npz', s['probability_sha256'][stem]) for s in sources]
    assert_weighted_pixels(arrays, [s['weight'] for s in sources], probability.astype(np.float16))
    del arrays
    with np.load(R34 / f'{stem}.npz', allow_pickle=False) as archive:
        old_probability = archive['probability'][0].astype(np.float32)
    original = refine_detections(raw, old_probability, confidence_floor=.03)
    if [d for d in original['post_nms'] if d['score'] >= .20] != expected_r34:
        raise ValueError('Original R34 geometry did not exactly reproduce')
    refined = refine_detections(raw, probability[0], confidence_floor=.03)
    fixed = refine_detections(raw, probability[0], confidence_floor=.20)['post_nms']
    proposals = refined['post_nms']
    if [d for d in proposals if d['score'] >= .20] != fixed:
        raise ValueError('New low-floor pool failed exact confidence-.20 geometry replay')
    image = np.load(image_path, allow_pickle=False)
    labels = probabilities_to_labels(probability, PostprocessConfig(**POOL_CONFIG))
    features = np.asarray([arbitrary_features(d['rle'], probability, image, labels) for d in proposals],
                          dtype=np.float32).reshape(-1, len(FEATURES))
    confidence = np.asarray([d['score'] for d in proposals], dtype=np.float32)
    features = np.concatenate([features, confidence[:, None]], axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite raw quality features')
    return {'image_id': stem, 'features': features, 'rles': [d['rle'] for d in proposals],
        'source_indices': [d['source_index'] for d in proposals], 'confidence': confidence,
        'semantic_rles': labels_to_rles(labels), 'raw_count': len(raw),
        'dropped_area_lt5': refined['dropped_area_lt5'], 'dropped_nms': refined['dropped_nms']}


def pin_files(paths):
    return {str(path): digest_file(path) for path in paths}


def check_pins(plan):
    for path, expected in plan['file_sha256'].items():
        if digest_file(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def prepare(output):
    cpu_only()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preparation requires a fresh directory')
    started = time.monotonic()
    manifest = json.loads(MANIFEST.read_text())
    # Preserve the original hardcoded validator: validate ORIGINAL R34 first.
    records, annotations, original = validate_sources(YOLO, R34, MANIFEST.resolve(), 0)
    yolo_report = json.loads((YOLO / 'run_report.json').read_text())
    if (len(records) != 141 or original['yolo']['checkpoint_sha256'] != YOLO_SHA
            or original['yolo']['raw_confidence'] != .03 or original['yolo']['imgsz'] != 1536):
        raise ValueError('Unexpected own-YOLO e41 source')
    boundary = OLD / 'boundary_frozen'
    old_identity = json.loads((boundary / 'identity.json').read_text())
    if original != old_identity:
        raise ValueError('Original source identity changed after e41 evaluation')
    metadata = json.loads((ENSEMBLE / 'inference_config.json').read_text())
    if (metadata['split'] != 'val' or metadata['fold'] != 0 or metadata['ensemble']['weights'] != [.5, .5]
            or metadata['checkpoint_fingerprints'] != SEMANTIC_SHAS):
        raise ValueError('Expected equal held-out B5/ConvNeXt source')
    reconstructed, _ = verify_ensemble([s['prob_dir'] for s in metadata['ensemble']['sources']],
        metadata['ensemble']['requested_weights'], ENSEMBLE, MANIFEST)
    if reconstructed != metadata or any(p['verification'] != 'data_signature' for p in metadata['validation_provenance']):
        raise ValueError('Ensemble provenance failed exact source replay')
    state = json.loads((ENSEMBLE / 'ensemble_state.json').read_text())
    if (state['identity_sha256'] != canonical_digest(metadata) or set(state['completed']) != set(records)
            or {p.stem for p in ENSEMBLE.glob('*.npz')} != set(records)):
        raise ValueError('Incomplete/different ensemble cache')
    for stem in records:
        if digest_file(ENSEMBLE / f'{stem}.npz') != state['completed'][stem]:
            raise ValueError('Actual ensemble cache differs from completion hash')
    r34_cache = joblib.load(R34_QUALITY / 'features.joblib')
    r34_oof = np.load(R34_QUALITY / 'oof_quality.npy', allow_pickle=False)
    r34_report = json.loads((R34_QUALITY / 'report.json').read_text())
    validate_entries(r34_cache['entries'], r34_oof, records, r34_report)
    if (r34_cache['identity']['inference'] != original['r34'] or r34_cache['identity']['config'] != POOL_CONFIG
            or r34_cache['identity']['annotation_sha256'] != manifest['annotation_sha256']):
        raise ValueError('Original R34 quality identity differs')
    group_folds = original_group_folds(r34_cache['entries'])
    old_quality = OLD / 'refined_quality_fixed'
    old_report = json.loads((old_quality / 'report.json').read_text())
    if (old_report['identity']['group_folds'] != group_folds or old_report['identity']['features'] != QUALITY_FEATURES
            or old_report['identity']['regressor_params'] != new_regressor().get_params()
            or old_report['identity']['quality_thresholds'] != [.20]
            or old_report['identity']['sources'] != original or old_report['identity']['geometry'] != PARAMETERS
            or old_report['identity']['proposal_floor'] != .03 or old_report['identity']['fixed_threshold'] != .20):
        raise ValueError('Original e41 quality recipe or date folds differ')
    semantic_cache = joblib.load(ENSEMBLE_QUALITY / 'features.joblib')
    semantic_oof = np.load(ENSEMBLE_QUALITY / 'oof_quality.npy', allow_pickle=False)
    semantic_report = json.loads((ENSEMBLE_QUALITY / 'report.json').read_text())
    validate_entries(semantic_cache['entries'], semantic_oof, records, semantic_report)
    if (semantic_cache['identity']['inference'] != metadata or semantic_cache['identity']['config'] != POOL_CONFIG
            or semantic_report['best']['quality_threshold'] != .20):
        raise ValueError('Ensemble quality baseline identity differs')
    semantic_audit = json.loads((ENSEMBLE_AUDIT / 'native_audit.json').read_text())['candidate']
    if (semantic_audit['probability_sha256'] != state['completed'] or semantic_audit['quality_threshold'] != .20
            or semantic_audit['fold'] != 0 or semantic_audit['images'] != len(records)
            or Path(semantic_audit['quality_directory']).resolve() != ENSEMBLE_QUALITY.resolve()):
        raise ValueError('Native ensemble audit source differs')
    for name, expected in semantic_audit['quality_artifact_sha256'].items():
        if digest_file(ENSEMBLE_QUALITY / name) != expected:
            raise ValueError('Native ensemble audit quality artifact changed')
    assert_metric(semantic_audit['cached_score'], semantic_report['best'])
    files = [MANIFEST, annotations, Path(__file__), Path('tests/test_yolo_semantic_upgrade.py'),
             R34 / 'inference_config.json', ENSEMBLE / 'inference_config.json', ENSEMBLE / 'ensemble_state.json',
             ENSEMBLE_AUDIT / 'native_audit.json', ENSEMBLE_AUDIT / 'per_image.json',
             YOLO / 'inference_config.json', YOLO / 'run_report.json', Path(yolo_report['checkpoint'])]
    files.extend(Path(__file__).with_name(name) for name in ('yolo_boundary.py', 'yolo_refined_quality.py',
        'candidate_fusion.py', 'quality.py', 'quality_pool.py', 'ensemble.py', 'metrics.py', 'postprocess.py',
        'inference.py', 'infer_yolo.py', 'train_yolo.py'))
    for directory in (R34_QUALITY, ENSEMBLE_QUALITY, old_quality):
        files.extend(directory / name for name in ('features.joblib', 'oof_quality.npy', 'report.json'))
    files.extend(boundary / name for name in ('identity.json', 'predictions.json', 'baseline.csv', 'refined.csv', 'report.json'))
    files.append(old_quality / 'selected_oof.csv')
    files.extend(Path(p) for p in metadata['checkpoint'])
    for source in metadata['ensemble']['sources']:
        files.append(Path(source['prob_dir']) / 'inference_config.json')
        files.extend(Path(source['prob_dir']) / f'{stem}.npz' for stem in records)
    image_paths = {}
    for stem, record in records.items():
        path = Path(record['image_path'])
        image_paths[stem] = path if path.is_absolute() else MANIFEST.parent / path
        files.extend([image_paths[stem], YOLO / 'raw' / f'{stem}.json', R34 / f'{stem}.npz', ENSEMBLE / f'{stem}.npz'])
    plan = {'spec': SPEC, 'original_source_identity': original, 'ensemble_metadata': metadata,
        'group_folds': group_folds, 'file_sha256': pin_files(files),
        'review_gate': {'required_file': 'independent_review.json', 'required_true': 'approved_for_fixed_fit',
            'required_hashes': ['predeclaration_sha256', 'prepared_sha256', 'source_sha256']},
        'fit_has_not_started': True, 'no_final_model': True}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'predeclaration.json', plan)
    old_predictions = json.loads((boundary / 'predictions.json').read_text())
    jobs = [(e['image_id'], image_paths[e['image_id']], metadata, state['completed'][e['image_id']],
             old_predictions[e['image_id']]['post_nms']) for e in r34_cache['entries']]
    entries = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        for index, entry in enumerate(pool.map(geometry_features, jobs), 1):
            entries.append(entry)
            if index % 20 == 0 or index == len(jobs):
                print(json.dumps({'geometry_images': index, 'total': len(jobs)}), flush=True)
    # Seal the annotation-free geometric/feature output before reading targets.
    atomic_write(output / 'geometry_features.joblib', lambda stream: joblib.dump(entries, stream))
    semantic_score_by_image, offset = {}, 0
    for entry in semantic_cache['entries']:
        count = len(entry['target'])
        semantic_score_by_image[entry['image_id']] = semantic_oof[offset:offset + count]
        offset += count
    semantic_predictions = []
    semantic_entries = {e['image_id']: e for e in semantic_cache['entries']}
    for entry in entries:
        stem, rles = entry['image_id'], entry['rles']
        record = records[stem]
        semantic = entry.pop('semantic_rles')
        scores = semantic_score_by_image[stem]
        if len(scores) != len(semantic):
            raise ValueError('Rebuilt semantic baseline count differs')
        native_matrices = [rle_iou(semantic, annotation_rles(a)).T for a in record['annotators']]
        if rounded_matrix_multiset(native_matrices) != rounded_matrix_multiset(semantic_entries[stem]['iou']):
            raise ValueError('Rebuilt native semantic IoUs do not exactly round to cached float32 matrices')
        semantic_predictions.append((stem, [r for r, q in zip(semantic, scores) if q >= .20]))
        matrices = [rle_iou(rles, annotation_rles(a)).T for a in record['annotators']]
        entry.update(group=record['date_group'], weight=len(matrices), iou=matrices,
            target=(sum(np.where(m > .5, m, 0).sum(axis=0) for m in matrices) / len(matrices)).astype(np.float32))
    if sum(len(e['target']) for e in entries) > 5000:
        raise ValueError('Refined pool exceeds fixed 5000-candidate budget')
    atomic_write(output / 'features.joblib', lambda stream: joblib.dump({'entries': entries, 'group_folds': group_folds}, stream))
    atomic_json(output / 'fold_contract.json', fold_contract(entries, group_folds))
    write_submission(semantic_predictions, output / 'semantic_baseline.csv', records)
    baselines, per_image = {}, {}
    baselines['semantic'], per_image['semantic'] = rescore(output / 'semantic_baseline.csv', records)
    assert_metric(baselines['semantic'], semantic_audit['native_score'])
    original_native_rows = json.loads((ENSEMBLE_AUDIT / 'per_image.json').read_text())
    for stem, score in per_image['semantic'].items():
        assert_metric(score, original_native_rows[stem]['candidate'])
    boundary_report = json.loads((boundary / 'report.json').read_text())
    for name, path, expected in [('old_raw', boundary / 'baseline.csv', boundary_report['scores']['baseline']),
        ('old_boundary', boundary / 'refined.csv', boundary_report['scores']['post_nms']),
        ('old_quality', old_quality / 'selected_oof.csv', old_report['best'])]:
        baselines[name], per_image[name] = rescore(path, records)
        assert_metric(baselines[name], expected)
    confidence = np.concatenate([e['confidence'] for e in entries])
    fixed, fixed_images = evaluate(entries, confidence, .20)
    fixed_predictions = [(e['image_id'], [r for r, score in zip(e['rles'], e['confidence']) if score >= .20]) for e in entries]
    write_submission(fixed_predictions, output / 'new_boundary_conf20.csv', records)
    baselines['new_boundary'], per_image['new_boundary'] = rescore(output / 'new_boundary_conf20.csv', records)
    assert_metric(baselines['new_boundary'], fixed)
    check_pins(plan)
    atomic_json(output / 'baseline_per_image.json', per_image)
    receipt = {'baselines': baselines, 'pool_count': sum(len(e['target']) for e in entries), 'images': len(entries),
        'features_sha256': digest_file(output / 'features.joblib'), 'geometry_features_sha256': digest_file(output / 'geometry_features.joblib'),
        'baseline_per_image_sha256': digest_file(output / 'baseline_per_image.json'),
        'fold_contract_sha256': digest_file(output / 'fold_contract.json'),
        'predeclaration_sha256': digest_file(output / 'predeclaration.json'),
        'source_sha256': digest_file(__file__), 'source_hashes_unchanged': True,
        'old_r34_geometry_exact': True, 'ensemble_weighted_pixels_exact': True,
        'native_semantic_iou_matrices_round_exactly_to_cached_float32': True,
        'semantic_cached_float32_score': semantic_report['best'],
        'native_csv_baselines_exact': True, 'quality_fit_started': False, 'seconds': time.monotonic() - started}
    atomic_json(output / 'prepared.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def fit(output):
    cpu_only()
    plan = json.loads((output / 'predeclaration.json').read_text())
    prepared = json.loads((output / 'prepared.json').read_text())
    review = json.loads((output / 'independent_review.json').read_text())
    expected = {'predeclaration_sha256': digest_file(output / 'predeclaration.json'),
        'prepared_sha256': digest_file(output / 'prepared.json'), 'source_sha256': digest_file(__file__)}
    if review.get('approved_for_fixed_fit') is not True or any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('Require independent exact-hash review before the one fixed fit')
    if (output / 'fit_started.json').exists():
        raise ValueError('Single-fit guard: this experiment has already begun fitting')
    check_pins(plan)
    for name in ('features', 'baseline_per_image', 'fold_contract', 'geometry_features'):
        suffix = '.joblib' if name in ('features', 'geometry_features') else '.json'
        if digest_file(output / (name + suffix)) != prepared[name + '_sha256']:
            raise ValueError('Prepared artifact changed')
    data = joblib.load(output / 'features.joblib')
    entries, group_folds = data['entries'], data['group_folds']
    if group_folds != plan['group_folds'] or fold_contract(entries, group_folds) != json.loads((output / 'fold_contract.json').read_text()):
        raise ValueError('Meta-fold assignment changed')
    atomic_json(output / 'fit_started.json', {**expected, 'unix_time': time.time(), 'threshold': .20})
    scores = crossfit(entries, group_folds)
    result, per_image = evaluate(entries, scores, .20)
    predictions, offset, selected_ids = [], 0, {}
    for entry in entries:
        count = len(entry['target'])
        keep = scores[offset:offset + count] >= .20
        offset += count
        rles = [r for r, retained in zip(entry['rles'], keep) if retained]
        selected_ids[entry['image_id']] = [i for i, retained in zip(entry['source_indices'], keep) if retained]
        for rle in rles:
            if not np.array_equal(mu.decode(rle), mu.decode(mu.encode(np.asfortranarray(mu.decode(rle))))):
                raise ValueError('Selected native RLE failed identity roundtrip')
        if len(rles) > 1 and np.triu(mu.iou(rles, rles, [0] * len(rles)), 1).max() > .5:
            raise ValueError('Mask-NMS duplicate survived')
        predictions.append((entry['image_id'], rles))
    export = write_submission(predictions, output / 'selected_oof.csv', [e['image_id'] for e in entries])
    manifest = json.loads(MANIFEST.read_text())
    records = {r['image_id']: r for r in manifest['train'] if r['fold'] == 0}
    direct, direct_images = rescore(output / 'selected_oof.csv', records)
    assert_metric(direct, result)
    for stem in per_image:
        assert_metric(direct_images[stem], per_image[stem])
    baselines = json.loads((output / 'baseline_per_image.json').read_text())
    groups = {e['image_id']: e['group'] for e in entries}
    confidence = np.concatenate([e['confidence'] for e in entries])
    diagnostics = proposal_diagnostics(entries, confidence, scores, .20)
    check_pins(plan)
    cpu_only()
    report = {'spec': SPEC, 'quality': result, 'native_csv_metric': direct, 'prepared_baselines': prepared['baselines'],
        'comparisons': {name: paired_bootstrap(rows, direct_images, groups) for name, rows in baselines.items()},
        'proposal_diagnostics': diagnostics, 'export': {**export, 'sha256': digest_file(output / 'selected_oof.csv')},
        'checks': {'native_csv_exact': True, 'original_date_meta_folds': True, 'source_hashes_unchanged': True,
                   'quality_thresholds': [.20], 'final_model_fitted': False, 'cuda_initialized': False},
        'interpretation': SPEC['interpretation'] + ' Mask NMS is not pixel-disjointness; see measured overlap. Conditional paired intervals exclude development selection uncertainty.',
        'decision': 'stop_negative' if direct['pq'] <= prepared['baselines']['old_quality']['pq'] else 'positive_development_only_root_review_required'}
    atomic_write(output / 'oof_quality.npy', lambda stream: np.save(stream, scores))
    atomic_json(output / 'selected_source_ids.json', selected_ids)
    atomic_json(output / 'per_image.json', direct_images)
    atomic_json(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'fit'))
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else fit)(OUTPUT)

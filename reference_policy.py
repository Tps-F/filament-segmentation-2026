"""Reference equality policy only; never changes inference or output values."""
from __future__ import annotations
import re

MODES = ('strict', 'report-only')
POLICY = {'default': 'strict', 'choices': list(MODES),
    'report_only_changes': 'Only equality to canonical GB10 output hashes becomes nonfatal.',
    'never_relaxed': ['bundle/source/weights/input hashes', 'native BF16/backend/precision',
        'native shapes/finiteness', 'actual artifact hashes', 'source identity/confidence/R12',
        'fixed q20/disjoint ownership', 'zero pixel overlap'],
    'cross_hardware_equality_claim': False}


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('Reference mode must be strict or report-only')
    return mode


def check_reference(mode, actual, expected, label, checks):
    validate_mode(mode)
    if not all(isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) for x in (actual, expected)):
        raise ValueError('Missing or malformed actual/reference hash')
    exact = actual == expected
    checks.append({'label': label, 'actual_sha256': actual, 'reference_sha256': expected, 'exact': exact})
    if mode == 'strict' and not exact:
        raise ValueError('Reference mismatch: '+label+'; actual='+actual+'; reference='+expected)
    return exact


def summarize(mode, checks):
    validate_mode(mode)
    return {'mode': mode, 'checks': checks, 'checked': len(checks),
            'mismatches': sum(not row['exact'] for row in checks),
            'all_reference_hashes_exact': bool(checks) and all(row['exact'] for row in checks)}


def validate_probability(value, dtype):
    import numpy as np
    if value.shape != (2, 2048, 2048) or value.dtype != np.dtype(dtype):
        raise ValueError('Expected native two-channel '+dtype+' probability')
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError('Nonfinite or out-of-range native probability')


def validate_raw_rows(rows):
    import numpy as np
    from pycocotools import mask as mu
    previous = 1.0
    for row in rows:
        score = row['score']; box = np.asarray(row['box'])
        if not np.isfinite(score) or not 0 <= score <= previous or box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid original detector confidence/box/order')
        previous = score
        mask = mu.decode(row['rle'])
        if mask.shape != (2048, 2048) or row['area'] <= 0 or row['area'] != int(mask.sum()):
            raise ValueError('Invalid native raw mask/area')

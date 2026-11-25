"""Verify the exact original-row B v9 archive; seed equality is insufficient."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .core import SEED, SPLIT_NAMES, load_dataset, sha256_file, validate_split_indices
from .paths import SHARED_SPLIT_ROOT

CANONICAL_DATASET_SHA256 = '19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0'
CANONICAL_ARCHIVE_SHA256 = '26f95eee418ae5510ce7e2f27369f02cf3385104dadb0b327715ef924c11bae3'
CANONICAL_ROWS = 253680
CANONICAL_COUNTS = dict(train=177576, validation=38052, test=38052)
SHARED_MODE = 'shared_b_v9_original_rows'
LEGACY_QA_MODE = 'qa_legacy_test_first_sorted'
RECIPE = 'train_test_split(test_size=0.30), then temporary test_size=0.50; stratify=y; random_state=42; preserve returned order'


def array_sha256(array):
    """Portable digest of source positions, including their order."""
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes(order='C')).hexdigest()


def npy_sha256(array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def b_v9_indices(y):
    y = np.asarray(y)
    train, temporary = train_test_split(np.arange(len(y)), test_size=0.30,
                                       stratify=y, random_state=SEED)
    validation, test = train_test_split(temporary, test_size=0.50,
                                       stratify=y[temporary], random_state=SEED)
    return dict(train=train, validation=validation, test=test)


def validate_b_v9_recipe(splits, y):
    """Check complete/disjoint membership AND order against the two-stage recipe."""
    checked = validate_split_indices(splits, len(y))
    expected = b_v9_indices(y)
    for name in SPLIT_NAMES:
        if not np.array_equal(checked[name], expected[name]):
            raise ValueError(f'{name} differs from B v9 stratified recipe or source-row order')
    return checked


def verify_shared_split(csv_path, archive_path=None, manifest_path=None):
    csv_path = Path(csv_path).expanduser().resolve()
    archive_path = Path(archive_path or SHARED_SPLIT_ROOT / 'shared_split_indices.npz').resolve()
    manifest_path = Path(manifest_path or SHARED_SPLIT_ROOT / 'shared_split_manifest.json').resolve()
    for path in (csv_path, archive_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required shared-split input unavailable: {path}; no generated fallback')
    manifest_digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Shared-split manifest must be an object')
    expected_hash = manifest.get('expected_dataset_full_sha256_from_user_terminal')
    if (expected_hash != CANONICAL_DATASET_SHA256
            or manifest.get('sizes') != CANONICAL_COUNTS
            or manifest.get('original_n_rows') != CANONICAL_ROWS
            or manifest.get('uses_original_row_positions') is not True):
        raise ValueError('Shared-split manifest differs from the frozen dataset/population identity')
    source_hash = sha256_file(csv_path)
    if source_hash != CANONICAL_DATASET_SHA256:
        raise ValueError(f'Dataset SHA256 mismatch: got {source_hash}, expected {CANONICAL_DATASET_SHA256}')
    archive_hash = sha256_file(archive_path)
    if (archive_hash != CANONICAL_ARCHIVE_SHA256
            or manifest.get('archive_sha256', CANONICAL_ARCHIVE_SHA256) != archive_hash):
        raise ValueError('Shared split archive SHA256 differs from the approved exact archive')
    X, y = load_dataset(csv_path)
    if len(X) != CANONICAL_ROWS:
        raise ValueError('Canonical shared split requires all original source rows')
    with np.load(archive_path, allow_pickle=False) as archive:
        splits = validate_split_indices(
            {key: archive[key] for key in archive.files}, len(X), CANONICAL_COUNTS)
    validate_b_v9_recipe(splits, y)
    normalized = {key: array_sha256(splits[key]) for key in SPLIT_NAMES}
    original_npy = {key: npy_sha256(splits[key]) for key in SPLIT_NAMES}
    if (manifest.get('arrays_in_zip_sha256') != original_npy
            or manifest.get('normalized_array_sha256', normalized) != normalized):
        raise ValueError('Shared split array hashes differ from original B v9 source positions')
    if (sha256_file(csv_path) != source_hash or sha256_file(archive_path) != archive_hash
            or sha256_file(manifest_path) != manifest_digest):
        raise ValueError('Shared-split inputs changed during verification')
    counts = {}
    for name, ix in splits.items():
        positive = int(y.iloc[ix].sum())
        if not 0 < positive < len(ix):
            raise ValueError(f'{name} split must contain both original target classes')
        counts[name] = dict(rows=len(ix), positive_count=positive,
                            negative_count=len(ix)-positive, positive_fraction=positive/len(ix))
    return splits, dict(
        status='VERIFIED_EXACT_CSV_AND_RECIPE', mode=SHARED_MODE,
        csv_sha256=source_hash, shape=[len(X), len(X.columns)+1],
        archive_sha256=archive_hash, input_manifest_sha256=manifest_digest,
        archive_path=str(archive_path), input_manifest_path=str(manifest_path),
        normalized_array_sha256=normalized, source_npy_sha256=original_npy,
        recipe=RECIPE, seed=SEED, index_space='original_csv_positions',
        verified_B_v9_recipe_and_order=True, full_coverage=True, cross_split_overlap=0,
        class_counts=counts, split_ratios=[0.70, 0.15, 0.15],
        interpretation='Primary original-row split; identical X does not identify a respondent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--indices', type=Path, default=SHARED_SPLIT_ROOT / 'shared_split_indices.npz')
    parser.add_argument('--manifest', type=Path, default=SHARED_SPLIT_ROOT / 'shared_split_manifest.json')
    args = parser.parse_args()
    _, result = verify_shared_split(args.csv, args.indices, args.manifest)
    print(json.dumps(result, indent=2))
    print('SHARED SPLIT FULL VERIFICATION PASSED')


if __name__ == '__main__':
    main()

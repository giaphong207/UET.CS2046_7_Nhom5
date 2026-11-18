"""Run snapshots and checksum bindings; no credentials or signing keys are used."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

import yaml
from sklearn.model_selection import ParameterGrid

from .core import FEATURE_COUNTS, FINAL_MODEL_TIE_RULE, SEED, TARGET, sha256_file, versions, write_json

COMPONENT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = 2
MODEL_ENVIRONMENT_PACKAGES = (
    'numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib',
)
A4_CRITICAL_ARTIFACTS = (
    'config.yaml', 'runtime_config.json', 'source_identity.json', 'versions.json',
    'dataset_sha256.txt', 'dataset_audit.json', 'source_row_indices.npy',
    'split_indices.npz', 'cv_results.csv', 'validation_metrics.csv',
    'test_metrics.csv', 'test_predictions.csv', 'ablation.csv',
    'feature_sets.json', 'selection.json', 'fitted_models.joblib',
)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read valid JSON artifact: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'JSON artifact must be an object: {path}')
    return value


def _git_value(arguments):
    try:
        result = subprocess.run(
            ['git', *arguments], cwd=COMPONENT_ROOT, capture_output=True,
            text=True, check=False, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_identity():
    """Hash executed modules/scripts and dependency/protocol definitions, including untracked files."""
    paths = [
        *sorted((COMPONENT_ROOT / 'src' / 'diabetes_research').glob('*.py')),
        *sorted((COMPONENT_ROOT / 'scripts').glob('*.py')),
        COMPONENT_ROOT / 'config' / 'A2_protocol_v2.yaml',
        *sorted(COMPONENT_ROOT.glob('requirements*.txt')),
    ]
    hashes = {str(p.relative_to(COMPONENT_ROOT)): sha256_file(p) for p in paths}
    encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'file_hashes': hashes,
        'source_files_sha256': hashlib.sha256(encoded).hexdigest(),
        'git_head': _git_value(['rev-parse', 'HEAD']),
        'git_branch': _git_value(['branch', '--show-current']),
        'git_worktree_dirty': bool(_git_value(['status', '--porcelain'])),
        'identity_basis': 'file hashes; Git HEAD alone does not identify untracked code',
    }


def start_execution(stage):
    started = datetime.now(timezone.utc)
    return {
        'run_id': f'{stage}_{started.strftime("%Y%m%dT%H%M%S%fZ")}_{uuid.uuid4().hex[:12]}',
        'started_at_utc': started.isoformat(),
    }, time.perf_counter()


def finish_execution(metadata, started):
    return {
        **metadata,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': time.perf_counter() - started,
    }


def snapshot_protocol(output_dir):
    """Copy the available research protocol verbatim; runtime overrides are recorded separately."""
    path = COMPONENT_ROOT / 'config' / 'A2_protocol_v2.yaml'
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    if not isinstance(protocol, dict):
        raise ValueError('Research protocol must be a YAML mapping')
    validate_protocol(protocol)
    (Path(output_dir) / 'config.yaml').write_bytes(raw)
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_protocol(protocol):
    """Fail on drift in fixed scientific choices; the existing CLI exposes CV/budget overrides."""
    required = {
        'project': {'target': TARGET},
        'split': {'train': 0.70, 'validation': 0.15, 'test': 0.15,
                  'stratify': True, 'seed': SEED, 'test_locked': True},
        'cross_validation': {'type': 'StratifiedKFold', 'seed': SEED,
                             'selection_metric': 'average_precision'},
        'threshold': {'reference': 0.5, 'selection': 'maximize_f1_validation',
                      'tie_rule': 'highest_threshold'},
        'feature_reduction': {'k_values': list(FEATURE_COUNTS),
                              'fit_ranking_on_training_only': True, 'use_test_for_selection': False},
        'metrics': {'primary': 'average_precision'},
        'uncertainty': {'method': 'paired_test_bootstrap', 'n_resamples': 1000,
                        'confidence_level': 0.95, 'paired_same_indices': True,
                        'conditional_on_fitted_models': True},
    }
    for section, fields in required.items():
        saved = protocol.get(section, {})
        if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in fields.items()):
            raise ValueError(f'Protocol differs from the implemented scientific choices: {section}')


def artifact_hashes(output_dir, names=None):
    """Manifests are excluded from their own hash lists to avoid circular hashes."""
    root = Path(output_dir)
    if names is None:
        names = sorted(
            p.name for p in root.iterdir()
            if p.is_file() and p.name not in ('manifest.json', 'manifest_a5.json')
        )
    hashes = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError(f'Artifact name must be a basename: {name}')
        path = root / name
        if not path.is_file():
            raise ValueError(f'Missing critical artifact: {name}')
        hashes[name] = sha256_file(path)
    return hashes


def verify_a4_provenance(a4_dir, manifest):
    """Verify bytes and compatible source/environment before loading the trusted model pickle."""
    root = Path(a4_dir)
    if manifest.get('manifest_version') != MANIFEST_VERSION:
        raise ValueError(
            'Unsupported A4 provenance version. Rerun A4 with this component; '
            'legacy artifacts without integrity metadata are not accepted.'
        )
    if not isinstance(manifest.get('run_id'), str) or not manifest['run_id'].startswith('a4_'):
        raise ValueError('Invalid A4 run identifier')
    recorded = manifest.get('artifact_hashes')
    if not isinstance(recorded, dict) or not set(A4_CRITICAL_ARTIFACTS).issubset(recorded):
        raise ValueError('A4 provenance is missing critical artifact hashes')
    actual = artifact_hashes(root, list(recorded))
    for name, digest in recorded.items():
        if actual[name] != digest:
            raise ValueError(f'Critical artifact checksum mismatch: {name}')

    identity = read_json(root / 'source_identity.json')
    current_identity = source_identity()
    if (identity.get('file_hashes') != current_identity['file_hashes']
            or identity.get('source_files_sha256') != current_identity['source_files_sha256']
            or manifest.get('source_files_sha256') != identity.get('source_files_sha256')):
        raise ValueError('A4 source-code identity differs from current component')

    recorded_versions = read_json(root / 'versions.json')
    current_versions = versions()
    if str(recorded_versions.get('python', '')).split('.')[:2] != current_versions['python'].split('.')[:2]:
        raise ValueError('A4 Python major/minor version is incompatible')
    for package in MODEL_ENVIRONMENT_PACKAGES:
        if recorded_versions.get(package) != current_versions[package]:
            raise ValueError(f'A4 model environment is incompatible: {package}')

    if (manifest.get('protocol_sha256') != actual['config.yaml']
            or identity['file_hashes'].get('config/A2_protocol_v2.yaml') != actual['config.yaml']):
        raise ValueError('Protocol snapshot hash differs from A4 manifest')
    validate_protocol(yaml.safe_load((root / 'config.yaml').read_bytes()))
    if (root / 'dataset_sha256.txt').read_text(encoding='ascii').strip() != manifest.get('dataset_sha256'):
        raise ValueError('Dataset checksum artifact differs from A4 manifest')

    runtime = read_json(root / 'runtime_config.json')
    if runtime.get('run_id') != manifest['run_id']:
        raise ValueError('Runtime configuration belongs to a different A4 run')
    effective = runtime.get('effective', {})
    if (manifest.get('target') != TARGET or effective.get('target') != TARGET
            or manifest.get('seed') != SEED
            or runtime.get('seeds') != dict(split=SEED, cv=SEED, search=SEED, model=SEED)):
        raise ValueError('A4 target/random seeds differ from this research protocol')
    split = effective.get('split', {})
    if (split.get('ratios') != [0.70, 0.15, 0.15] or split.get('stratify') is not True
            or manifest.get('split_ratios') != split.get('ratios')):
        raise ValueError('A4 split protocol differs from runtime configuration')
    cv = effective.get('cross_validation', {})
    if (isinstance(cv.get('folds'), bool) or not isinstance(cv.get('folds'), int)
            or cv['folds'] < 2):
        raise ValueError('A4 CV fold count must be an integer >=2')
    if (cv.get('type') != 'StratifiedKFold' or cv.get('scoring') != 'average_precision'
            or cv.get('shuffle') is not True or cv.get('folds') != manifest.get('cv_folds')):
        raise ValueError('A4 cross-validation protocol differs from runtime configuration')
    if (effective.get('search_backend') != 'RandomizedSearchCV'
            or effective.get('weighting') != 'A1/A3 balanced on full train; B3 search unweighted'
            or effective.get('final_model_tie_rule') != FINAL_MODEL_TIE_RULE
            or effective.get('xgboost_fixed_configuration') != dict(tree_method='hist', eval_metric='logloss')
            or effective.get('threshold') != dict(reference=0.5, selection='max F1 on validation', tie_rule='highest threshold')
            or effective.get('feature_selection') != dict(method='SelectKBest(f_classif)', k_values=list(FEATURE_COUNTS),
                    fit_scope='full train; k selected on validation; no A3 CV',
                    ordinal_representation='numeric codes; LR scales all inputs')):
        raise ValueError('A4 effective modeling methodology is incompatible')
    counts = runtime.get('search_counts', {})
    spaces = effective.get('search_spaces', {})
    requested = runtime.get('requested', {})
    if set(counts) != {'B1', 'B2', 'B3'} or set(spaces) != set(counts):
        raise ValueError('A4 search counts are incomplete')
    if (isinstance(requested.get('trials'), bool) or not isinstance(requested.get('trials'), int)
            or not 1 <= requested['trials'] <= 100
            or not isinstance(requested.get('quick'), bool) or not isinstance(requested.get('demo'), bool)):
        raise ValueError('A4 requested runtime configuration is invalid')
    expected_status = ('SMOKE_NOT_RESEARCH_RESULT' if requested['quick'] else
                       ('SYNTHETIC_QA_ONLY' if requested['demo'] else 'RESEARCH_TEST_COMPLETE'))
    if effective.get('mode') != expected_status or manifest.get('status') != expected_status:
        raise ValueError('A4 mode differs from requested runtime configuration')
    effective_budget = 1 if requested['quick'] else requested['trials']
    if requested['quick'] and cv.get('folds') != 2:
        raise ValueError('Quick A4 must use two CV folds')
    if not requested['quick'] and cv.get('folds') != requested.get('cv_folds'):
        raise ValueError('A4 effective CV folds differ from the requested configuration')
    for key, value in counts.items():
        cardinality = len(ParameterGrid(spaces[key]))
        actual_count = min(effective_budget, cardinality)
        expected = dict(requested_configurations=effective_budget,
                        actual_configurations=actual_count, finite_space_size=cardinality,
                        space_exhausted=(actual_count == cardinality),
                        budget_capped_to_finite_space=(effective_budget > cardinality),
                        cv_fits=actual_count * cv['folds'], refit_fits=1)
        if not isinstance(value, dict) or any(value.get(field) != number for field, number in expected.items()):
            raise ValueError(f'A4 search budget/count metadata is inconsistent: {key}')
    actual_counts = {key: value.get('actual_configurations') for key, value in counts.items()}
    if (actual_counts != manifest.get('actual_trials_per_family')
            or actual_counts != manifest.get('trials_per_family')
            or manifest.get('requested_trials_per_family') != {key: requested['trials'] for key in counts}
            or manifest.get('effective_requested_trials_per_family') != {key: effective_budget for key in counts}):
        raise ValueError('A4 actual search counts differ from runtime configuration')
    if effective.get('mode') != manifest.get('status'):
        raise ValueError('A4 mode differs from runtime configuration')
    return {
        'runtime': runtime,
        'versions': recorded_versions,
        'source_identity': identity,
        'verified_artifact_hashes': actual,
        'source_manifest_sha256': sha256_file(root / 'manifest.json'),
    }


def save_analysis_provenance(output_dir, runtime):
    environment, identity = versions(), source_identity()
    write_json(Path(output_dir) / 'analysis_config.json', runtime)
    write_json(Path(output_dir) / 'versions_a5.json', environment)
    write_json(Path(output_dir) / 'source_identity_a5.json', identity)
    return dict(versions=environment, source_identity=identity)


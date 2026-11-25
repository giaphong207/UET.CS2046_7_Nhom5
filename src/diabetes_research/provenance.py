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

from .core import (CORE_BOOTSTRAP_PAIRS, FEATURE_COUNTS, FINAL_MODEL_TIE_RULE, SEED, TARGET,
                   feature_selection_policy, sha256_file, threshold_policy, versions, write_json)
from .paths import AMENDMENTS_PATH, COMPONENT_ROOT, PROTOCOL_PATH, REPOSITORY_ROOT, SHARED_SPLIT_ROOT
from .shared_splits import (CANONICAL_ARCHIVE_SHA256, CANONICAL_DATASET_SHA256,
                            LEGACY_QA_MODE, SHARED_MODE)
from .sensitivity_contract import validate_frozen_sensitivity

MANIFEST_VERSION = 3
PROTOCOL_IDENTITY_KEY = str(PROTOCOL_PATH.relative_to(REPOSITORY_ROOT))
MODEL_ENVIRONMENT_PACKAGES = (
    'numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib',
)
A4_CRITICAL_ARTIFACTS = (
    'config.yaml', 'runtime_config.json', 'source_identity.json', 'versions.json',
    'dataset_sha256.txt', 'dataset_audit.json', 'source_row_indices.npy',
    'split_indices.npz', 'cv_results.csv', 'validation_metrics.csv',
    'test_metrics.csv', 'test_predictions.csv', 'ablation.csv',
    'feature_sets.json', 'selection.json', 'fitted_models.joblib',
    'split_manifest.json', 'cv_fold_indices.npz', 'protocol_amendments.json',
    'test_exposure_log.json', 'threshold_search.json', 'selection_lock.json',
    'validation_reference_metrics.csv', 'test_reference_metrics.csv',
    'feature_selection_cv.csv', 'feature_selection_folds.json', 'feature_rankings.json',
    'feature_selection_models.joblib', 'secondary_anova.json', 'a2_reference_lock.json',
)
SELECTION_LOCK_ARTIFACTS = (
    'config.yaml', 'protocol_amendments.json', 'test_exposure_log.json',
    'source_identity.json', 'dataset_sha256.txt', 'source_row_indices.npy',
    'split_indices.npz', 'split_manifest.json', 'cv_fold_indices.npz', 'cv_results.csv',
    'validation_metrics.csv', 'validation_reference_metrics.csv', 'threshold_search.json',
    'feature_sets.json', 'selection.json', 'fitted_models.joblib',
    'feature_selection_cv.csv', 'feature_selection_folds.json', 'feature_rankings.json',
    'feature_selection_models.joblib', 'secondary_anova.json', 'a2_reference_lock.json',
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
            ['git', *arguments], cwd=REPOSITORY_ROOT, capture_output=True,
            text=True, check=False, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_identity():
    """Hash executed modules/scripts and dependency/protocol definitions, including untracked files."""
    paths = [
        *sorted((REPOSITORY_ROOT / 'src' / 'diabetes_research').glob('*.py')),
        *sorted((COMPONENT_ROOT / 'scripts').glob('*.py')),
        *sorted((COMPONENT_ROOT / 'notebooks').glob('*.ipynb')),
        *sorted(p for p in SHARED_SPLIT_ROOT.glob('*') if p.suffix in ('.py', '.npz', '.json')),
        PROTOCOL_PATH, AMENDMENTS_PATH, REPOSITORY_ROOT / 'pyproject.toml',
        *sorted(REPOSITORY_ROOT.glob('requirements*.txt')),
        *sorted(COMPONENT_ROOT.glob('requirements*.txt')),
    ]
    hashes = {str(p.relative_to(REPOSITORY_ROOT)): sha256_file(p) for p in paths if p.is_file()}
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
    path = PROTOCOL_PATH
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    if not isinstance(protocol, dict):
        raise ValueError('Research protocol must be a YAML mapping')
    validate_protocol(protocol)
    (Path(output_dir) / 'config.yaml').write_bytes(raw)
    (Path(output_dir) / 'protocol_amendments.json').write_bytes(AMENDMENTS_PATH.read_bytes())
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_protocol(protocol):
    """Fail on drift in fixed scientific choices; the existing CLI exposes CV/budget overrides."""
    required = {
        'project': {'target': TARGET},
        'split': {'train': 0.70, 'validation': 0.15, 'test': 0.15,
                  'stratify': True, 'seed': SEED, 'test_locked': True,
                  'primary_mode': SHARED_MODE, 'dataset_sha256': CANONICAL_DATASET_SHA256,
                  'archive_sha256': CANONICAL_ARCHIVE_SHA256},
        'cross_validation': {'type': 'StratifiedKFold', 'seed': SEED, 'folds': 3,
                             'selection_metric': 'average_precision'},
        'threshold': threshold_policy(),
        'feature_reduction': {'k_values': list(FEATURE_COUNTS),
                              'fit_ranking_on_training_only': True, 'use_test_for_selection': False,
                              'primary_status': 'IMPLEMENTED',
                              'primary_method': 'fold-local SelectFromModel',
                              'secondary_qa_method': 'SelectKBest(f_classif)',
                              'primary_configuration': feature_selection_policy()},
        'models': {'B3': {'family': 'XGBClassifier', 'primary_status': 'APPROVED'}},
        'metrics': {'primary': 'average_precision'},
        'uncertainty': {'method': 'paired_test_bootstrap', 'n_resamples': 1000,
                        'confidence_level': 0.95, 'paired_same_indices': True,
                        'conditional_on_fitted_models': True, 'seed': SEED,
                        'comparison_pairs': [dict(id=key, left=left, right=right)
                                             for key,left,right in CORE_BOOTSTRAP_PAIRS]},
    }
    for section, fields in required.items():
        saved = protocol.get(section, {})
        if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in fields.items()):
            raise ValueError(f'Protocol differs from the implemented scientific choices: {section}')
    shap = protocol.get('explanations', {}).get('shap', {})
    for key in ('n', 'background_n', 'chunk_size', 'prediction_batch_size'):
        value = shap.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Protocol SHAP budgets must be positive integers')
    if (shap.get('algorithm') != 'permutation' or shap.get('seed') != SEED
            or protocol.get('explanations', {}).get('lime', {}).get('num_samples') != 1000
            or protocol.get('explanations', {}).get('lime', {}).get('seed') != SEED):
        raise ValueError('Protocol explanation algorithm/seeds or distinct LIME budget differ')
    sensitivity=protocol.get('sensitivity', {})
    expected_sensitivity=dict(cv_folds=3,seed=SEED,cv_scope='primary training rows only',
        cv_methods=['StratifiedKFold','StratifiedGroupKFold'],use_for_pipeline_selection=False,
        outer_heldout_evaluation=False,outer_design_id='exact-x-group-outer-v3',
        cv_design_id='exact-x-primary-train-cv-v3')
    if (any(sensitivity.get(key)!=value for key,value in expected_sensitivity.items())
            or sensitivity.get('cv_std_ddof') != 0):
        raise ValueError('Protocol group sensitivity scope/frozen family differs')
    validate_frozen_sensitivity(protocol, read_json(AMENDMENTS_PATH))


def require_official_release():
    """A ready selector does not authorize official training or test evaluation."""
    validate_protocol(yaml.safe_load(PROTOCOL_PATH.read_bytes()))
    decision=read_json(AMENDMENTS_PATH).get('decisions',{}).get('official_run_release',{})
    if decision.get('status') != 'APPROVED':
        raise ValueError('Official evaluation authorization remains PENDING; primary SelectFromModel '
                         'implementation is READY. Separate official release is required before fitting/scoring.')


def artifact_hashes(output_dir, names=None):
    """Manifests are excluded from their own hash lists to avoid circular hashes."""
    root = Path(output_dir)
    if names is None:
        names = sorted(
            p.name for p in root.iterdir()
            if p.is_file() and p.name not in (
                'manifest.json', 'manifest_a5.json', 'manifest_group_sensitivity.json')
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
            or identity['file_hashes'].get(PROTOCOL_IDENTITY_KEY) != actual['config.yaml']):
        raise ValueError('Protocol snapshot hash differs from A4 manifest')
    validate_protocol(yaml.safe_load((root / 'config.yaml').read_bytes()))
    if actual['protocol_amendments.json'] != current_identity['file_hashes'].get(
            str(AMENDMENTS_PATH.relative_to(REPOSITORY_ROOT))):
        raise ValueError('Protocol amendment snapshot differs from current source identity')
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
    if (split.get('mode') not in (SHARED_MODE, LEGACY_QA_MODE)
            or manifest.get('split_mode') != split.get('mode')
            or split.get('archive_sha256') != actual['split_indices.npz']
            or manifest.get('split_archive_sha256') != actual['split_indices.npz']):
        raise ValueError('A4 split archive identity differs from runtime configuration')
    if split.get('mode') == SHARED_MODE and (
            split.get('shared_archive_sha256') != CANONICAL_ARCHIVE_SHA256
            or manifest.get('dataset_sha256') != CANONICAL_DATASET_SHA256):
        raise ValueError('A4 canonical shared split/dataset identity differs')
    cv = effective.get('cross_validation', {})
    if (isinstance(cv.get('folds'), bool) or not isinstance(cv.get('folds'), int)
            or cv['folds'] < 2):
        raise ValueError('A4 CV fold count must be an integer >=2')
    if (cv.get('type') != 'StratifiedKFold' or cv.get('scoring') != 'average_precision'
            or cv.get('shuffle') is not True or cv.get('folds') != manifest.get('cv_folds')):
        raise ValueError('A4 cross-validation protocol differs from runtime configuration')
    if (effective.get('search_backend') != 'RandomizedSearchCV'
            or effective.get('weighting') != 'A1/A2/A3 balanced on labels available to each fit; B3 search unweighted'
            or effective.get('final_model_tie_rule') != FINAL_MODEL_TIE_RULE
            or effective.get('xgboost_fixed_configuration') != dict(tree_method='hist', eval_metric='logloss')
            or effective.get('threshold') != threshold_policy()
            or manifest.get('threshold_policy') != threshold_policy()
            or effective.get('feature_selection') != feature_selection_policy()
            or manifest.get('feature_selection_status') != 'IMPLEMENTED'):
        raise ValueError('A4 effective modeling methodology is incompatible')
    saved_protocol=yaml.safe_load((root/'config.yaml').read_bytes())
    shap_config=saved_protocol['explanations']['shap']
    if (effective.get('explanations')!=saved_protocol['explanations']
            or manifest.get('shap_configuration')!=shap_config
            or manifest.get('shap_samples')!=shap_config['n']
            or manifest.get('shap_background_samples')!=shap_config['background_n']):
        raise ValueError('A4 SHAP/LIME effective settings differ from the protocol/manifest')
    if (effective.get('group_sensitivity_mode')!='separate_train_only_cv_and_outer_diagnostics'
            or manifest.get('group_sensitivity_mode')!=effective['group_sensitivity_mode']
            or manifest.get('group_sensitivity_performed') is not False):
        raise ValueError('A4 group sensitivity mode differs from the separate-run protocol')
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
    if expected_status == 'RESEARCH_TEST_COMPLETE':
        require_official_release()
        if cv['folds'] != 3:
            raise ValueError('Official A4 requires stratified 3-fold CV')
    if split.get('mode') == LEGACY_QA_MODE and expected_status == 'RESEARCH_TEST_COMPLETE':
        raise ValueError('Legacy split generation is a specifically labeled QA mode only')
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
    pairs = [dict(id=key, left=left, right=right) for key,left,right in CORE_BOOTSTRAP_PAIRS]
    if manifest.get('bootstrap_comparisons') != pairs or effective.get('bootstrap_comparisons') != pairs:
        raise ValueError('A4 prespecified bootstrap comparison family differs')
    lock = read_json(root / 'selection_lock.json')
    body = {key:value for key,value in lock.items() if key != 'lock_id'}
    lock_id = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (lock.get('run_id') != manifest['run_id'] or lock.get('lock_id') != lock_id
            or manifest.get('selection_lock_id') != lock_id
            or lock.get('threshold_policy') != threshold_policy()
            or lock.get('bootstrap_comparisons') != pairs
            or lock.get('feature_selection_status') != 'IMPLEMENTED'
            or lock.get('dataset_sha256') != manifest['dataset_sha256']
            or lock.get('artifact_hashes') != artifact_hashes(root,SELECTION_LOCK_ARTIFACTS)):
        raise ValueError('A4 selection lock differs from the frozen pre-test artifacts')
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


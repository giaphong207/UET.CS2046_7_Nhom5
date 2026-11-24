"""Shared safeguards for C's DEV/FINAL runs. No model selection on test."""
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def validate_split(indices, n, synthetic=False):
    expected = (177576, 38052, 38052)
    if not synthetic and (n != 253680 or tuple(map(len, indices)) != expected):
        raise ValueError('Wrong dataset or split size; expected full BRFSS 2015.')
    for a in indices:
        if a.ndim != 1 or a.dtype.kind not in 'iu' or len(a) == 0:
            raise ValueError('Indices must be non-empty 1-D integer arrays.')
        if a.min() < 0 or a.max() >= n:
            raise ValueError('Index out of range.')
    combined = np.concatenate(indices)
    if len(combined) != n or len(np.unique(combined)) != n:
        raise ValueError('Indices overlap, repeat or do not cover all rows.')


def json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_safe, allow_nan=False) + '\n', encoding='utf-8')


def provenance(args, B, script, raw_path, features, indices, options):
    if B.SEED != 42 or B.N_FOLDS != 3:
        raise ValueError('Unexpected seed or fold count in B pipeline.')
    packages = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'xgboost', 'scipy', 'threadpoolctl'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(dataset_sha256=None if args.synthetic else file_sha(raw_path),
                synthetic=args.synthetic, features=list(features), seed=B.SEED, n_folds=B.N_FOLDS,
                index_sha256={k: array_sha(v) for k,v in zip(('train','val','test'),indices)},
                source_sha256={'C':file_sha(script), 'guard':file_sha(__file__), 'B':file_sha(B.__file__)},
                b_version=getattr(B,'CODE_VERSION','unknown'), python=platform.python_version(),
                packages=packages, options=options)


class RunGuard:
    def __init__(self, args, meta, model_names):
        self.final = args.final
        self.root = Path(args.out_dir)
        self.meta = meta
        self.model_names = list(model_names)
        self.entries = {}
        self.scores = {}
        self.lock_path = self.root / 'c_locked_config.json'
        if args.final:
            if not args.approved_final:
                raise ValueError('FINAL blocked. Explicit group approval required (--approved-final).')
            if not args.synthetic and (args.quick or getattr(args, 'skip_tune', False) or args.beta != 2):
                raise ValueError('Real FINAL requires full F2 DEV configuration, not smoke/sensitivity.')
            lock = json.loads(self.lock_path.read_text(encoding='utf-8'))
            if lock['provenance'] != meta or list(lock['models']) != self.model_names:
                raise ValueError('Locked dataset, indices, code, environment, options or models differ.')
            self.entries = lock['models']
            score_path = self.root / 'val_predictions_C.npz'
            if file_sha(score_path) != lock['val_predictions_file_sha256']:
                raise ValueError('Locked validation prediction file changed.')
            with np.load(score_path, allow_pickle=False) as z:
                self.locked_scores = {n: z[f'proba_{n}'].copy() for n in self.model_names}
            self.out = self.root / 'final'
            if self.out.exists():
                raise ValueError('FINAL directory already exists; do not repeat test without group review.')
        else:
            if args.approved_final:
                raise ValueError('--approved-final is only for FINAL.')
            self.root.mkdir(parents=True, exist_ok=True)
            if any(self.root.iterdir()):
                raise ValueError('DEV output directory must be empty; preserve older evidence.')
            self.out = self.root

    def params(self, name, select):
        return self.entries[name]['params'] if self.final else select()

    def register(self, name, params, prob, select_threshold):
        p = np.asarray(prob)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid scores.')
        if self.final:
            entry = self.entries[name]
            threshold = entry['threshold']
            ref = self.locked_scores[name]
            if array_sha(ref) != entry['val_scores_sha256']:
                raise ValueError('Locked validation scores do not match manifest.')
            if p.shape != ref.shape or not np.allclose(p, ref, rtol=0, atol=1e-12):
                raise ValueError(f'Validation scores changed for {name}; stop before accessing test.')
            # Parallel floating-point summation can differ in the last bits.
            for t in (0.5, threshold):
                if not np.array_equal(p >= t, ref >= t):
                    raise ValueError(f'Validation decisions changed for {name}.')
            # Also reject any change in exact score ordering/ties (AP depends on ranking).
            order = np.argsort(ref, kind='stable')
            if not np.array_equal(np.sign(np.diff(ref[order])), np.sign(np.diff(p[order]))):
                raise ValueError(f'Validation ranking changed for {name}; stop before test.')
        else:
            threshold = float(select_threshold())
            self.entries[name] = dict(params=params, threshold=threshold, val_scores_sha256=array_sha(p))
        self.scores[name] = p
        return threshold

    def finish_dev(self, idx, y):
        np.savez_compressed(self.root/'val_predictions_C.npz', idx_val=idx, y_val=y,
                            **{f'proba_{n}':p for n,p in self.scores.items()})
        write_json(self.lock_path, dict(provenance=self.meta, models=self.entries,
                   val_predictions_file_sha256=file_sha(self.root/'val_predictions_C.npz')))

    def begin_test(self):
        # Reserve before any test prediction; a failed run also requires review.
        self.out.mkdir(exist_ok=False)
        write_json(self.out/'test_started.json', {'lock_sha256':file_sha(self.lock_path), 'synthetic':self.meta['synthetic']})

    def finish_test(self, filename, idx, y, predictions):
        for name,p in predictions.items():
            if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f'Invalid test scores for {name}')
        np.savez_compressed(self.out/filename, idx_test=idx, y_test=y,
                            **{f'proba_{n}':p for n,p in predictions.items()})
        write_json(self.out/'test_completed.json', {'lock_sha256':file_sha(self.lock_path), 'synthetic':self.meta['synthetic'],
                    'predictions_sha256':file_sha(self.out/filename)})

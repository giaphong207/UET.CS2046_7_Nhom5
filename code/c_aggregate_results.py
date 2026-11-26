"""
S9.5 - Historical CV and validation aggregation.

Generated from the six notebook cells used
for source auditing and result aggregation.

Scope:
- Existing CV and validation results only.
- No model training.
- No raw BRFSS data.
- No test or FINAL performance evaluation.

The historical source package is verified
using its pinned SHA-256 hashes.
"""

import argparse
from pathlib import Path


def aggregate(input_root, out_dir):

    from pathlib import Path

    if not __debug__:
        raise RuntimeError(
            "Run without Python optimization (-O). "
            "The pipeline requires assertions."
        )

    input_root = Path(input_root).resolve()
    out_dir = Path(out_dir).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    if out_dir.is_relative_to(input_root):
        raise ValueError(
            "Output must be outside input directory."
        )

    if out_dir.exists():

        if not out_dir.is_dir():
            raise ValueError(
                "Output path is not a directory."
            )

        if any(out_dir.iterdir()):
            raise FileExistsError(
                "Output directory must be empty."
            )



    # BEGIN NOTEBOOK CELL 1


    # ==========================================
    # S9.5 - CELL 1: ENVIRONMENT SETUP
    # ==========================================

    from pathlib import Path
    import hashlib
    import zipfile
    import json
    import platform

    import numpy as np
    import pandas as pd

    def display(obj):
        if hasattr(obj, 'to_string'):
            print(obj.to_string(index=False, max_rows=12))
        else:
            print(obj)

    SEED = 42

    INPUT_ROOT = Path(input_root).resolve()

    OUT = Path(out_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    PRIMARY_METRIC = "AP"

    EXPECTED_SOURCE_FILES = 34
    EXPECTED_CSV_FILES = 29

    EXPECTED_REGISTRY_HASH = (
        "6056188ba850e34c9d6c81d5ea39dbcfabd154b782a15abc2aa43dc3480284a4"
    )

    EXPECTED_ZIP_HASH = (
        "da99c86e5bd1756f14c2fa6297733ee0aa79b4597ce8f1a987123f2d477c32e3"
    )


    def sha256_file(path):
        return hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest()


    print("=== S9.5 ENVIRONMENT ===")

    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)

    print("\nSeed:", SEED)
    print("Primary metric:", PRIMARY_METRIC)
    print("Scope: Existing CV and validation results")

    print("\nInput root:", INPUT_ROOT)
    print("Output:", OUT)

    print("\n=== SAFETY PROTOCOL ===")

    print("Model training: NO")
    print("Raw BRFSS required: NO")
    print("Test metrics aggregation: NO")
    print("FINAL evaluation: NO")

    assert INPUT_ROOT.is_dir()
    assert OUT.is_dir()

    print("\nS9.5 CELL 1: PASS")

    # END NOTEBOOK CELL 1


    # BEGIN NOTEBOOK CELL 2


    # ==========================================
    # S9.5 - CELL 2: SOURCE INTEGRITY AUDIT
    # ==========================================

    print("=== LOCATE INPUT DATA ===")

    registry_paths = list(
        INPUT_ROOT.rglob("source_registry.csv")
    )

    zip_paths = list(
        INPUT_ROOT.rglob("S9_5_Aggregation_Inputs.zip")
    )

    # 1. Locate or extract package

    if len(registry_paths) == 1:

        REGISTRY_PATH = registry_paths[0]
        SOURCE_ROOT = REGISTRY_PATH.parent

        print("Input mode: Extracted directory")

    elif len(registry_paths) == 0:

        assert len(zip_paths) == 1, zip_paths

        source_zip = zip_paths[0]

        assert sha256_file(source_zip) == EXPECTED_ZIP_HASH

        extract_dir = OUT / "input_snapshot"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(source_zip) as z:

            assert z.testzip() is None

            for name in z.namelist():

                destination = (extract_dir / name).resolve()

                assert destination.is_relative_to(
                    extract_dir.resolve()
                )

            z.extractall(extract_dir)

        SOURCE_ROOT = extract_dir / "S9_5_inputs"
        REGISTRY_PATH = SOURCE_ROOT / "source_registry.csv"

        print("Input mode: ZIP archive")
        print("[PASS] Input ZIP SHA-256")

    else:

        raise ValueError(
            f"Multiple source registries found: {registry_paths}"
        )

    print("Source root:", SOURCE_ROOT)

    # 2. Verify registry

    assert REGISTRY_PATH.is_file()

    assert sha256_file(REGISTRY_PATH) == EXPECTED_REGISTRY_HASH

    registry = pd.read_csv(
        REGISTRY_PATH,
        dtype={"source_id": str}
    )

    required_columns = {
        "source_id",
        "owner",
        "archive",
        "archive_sha256",
        "archive_member",
        "relative_path",
        "category",
        "sha256",
        "bytes",
        "data_rows"
    }

    assert required_columns.issubset(registry.columns)

    assert len(registry) == EXPECTED_SOURCE_FILES
    assert registry["source_id"].is_unique
    assert registry["relative_path"].is_unique
    assert set(registry["owner"]) == {"B", "C"}

    print("[PASS] Source registry")

    # 3. Verify source files

    audit_rows = []

    for row in registry.itertuples(index=False):

        path = (SOURCE_ROOT / row.relative_path).resolve()

        assert path.is_relative_to(SOURCE_ROOT.resolve())
        assert path.is_file(), path

        actual_hash = sha256_file(path)

        assert actual_hash == row.sha256, (
            f"Hash mismatch: {row.relative_path}"
        )

        assert path.stat().st_size == row.bytes

        actual_rows = None

        if path.suffix == ".csv":

            df = pd.read_csv(path)
            actual_rows = len(df)

            assert actual_rows == int(row.data_rows)

        audit_rows.append({
            "source_id": row.source_id,
            "owner": row.owner,
            "relative_path": row.relative_path,
            "category": row.category,
            "sha256_match": True,
            "size_match": True,
            "data_rows": actual_rows,
            "status": "PASS"
        })

        print(
            f"[PASS] {row.source_id} | "
            f"{row.owner} | {path.name}"
        )

    # 4. Consolidated audit

    source_audit = pd.DataFrame(audit_rows)

    assert len(source_audit) == 34
    assert (source_audit["status"] == "PASS").all()

    csv_count = int(
        registry["relative_path"].str.endswith(".csv").sum()
    )

    assert csv_count == EXPECTED_CSV_FILES

    total_csv_rows = int(registry["data_rows"].sum())

    assert total_csv_rows == 230

    # 5. Save results

    source_audit.to_csv(
        OUT / "source_integrity_audit.csv",
        index=False
    )

    registry.to_csv(
        OUT / "source_registry.csv",
        index=False
    )

    # 6. Display inventory

    print("\n=== SOURCE INVENTORY BY OWNER ===")

    display(
        registry.groupby("owner")
        .size()
        .rename("n_files")
        .reset_index()
    )

    print("\n=== SOURCE INVENTORY BY CATEGORY ===")

    display(
        registry.groupby("category")
        .size()
        .rename("n_files")
        .reset_index()
    )

    print("\n=== SOURCE AUDIT SUMMARY ===")

    print("Source files:", len(registry))
    print("CSV files:", csv_count)
    print("Total CSV rows:", total_csv_rows)
    print("Verified:", len(source_audit), "/ 34")
    print("Output:", OUT)

    print("\nS9.5 CELL 2: PASS")

    # END NOTEBOOK CELL 2


    # BEGIN NOTEBOOK CELL 3


    # ==========================================
    # S9.5 - CELL 3: MODEL REGISTRY
    # ==========================================

    import json

    # 1. Source utilities

    SOURCE_MAP = registry.set_index("source_id")


    def get_source(source_id):

        info = SOURCE_MAP.loc[source_id]

        path = SOURCE_ROOT / info["relative_path"]

        assert path.is_file()
        assert sha256_file(path) == info["sha256"]

        return path


    def read_csv_source(source_id):

        return pd.read_csv(get_source(source_id))


    def read_json_source(source_id):

        return json.loads(
            get_source(source_id).read_text(encoding="utf-8")
        )


    # 2. Load source tables

    b_grid = read_csv_source("SRC002")
    b_val = read_csv_source("SRC004")

    b_lock = read_json_source("SRC012")
    b4_lock = read_json_source("SRC013")

    b4_val = read_csv_source("SRC008")

    assert len(b_grid) == 26
    assert len(b_val) == 14
    assert len(b4_val) == 6

    print("[PASS] Source tables loaded")

    # 3. Normalize model families

    B_FAMILY_MAP = {
        "dummy": "Dummy",
        "logistic": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "primary": "XGBoost",
        "stacking": "Stacking"
    }

    C_FAMILY_MAP = {
        "B0_dummy": "Dummy",
        "B1_logistic": "Logistic Regression",
        "B2_random_forest": "Random Forest",
        "APPX_decision_tree": "Decision Tree",
        "X1_gaussian_nb": "Gaussian Naive Bayes",
        "X2_mlp": "MLP",
        "X3_svm_linear": "Linear SVM",
        "X4_logistic_SMOTE": "Logistic Regression",
        "X5_random_forest_SMOTE": "Random Forest"
    }

    # 4. Registry helper

    model_rows = []


    def add_model(
        model_id,
        owner,
        experiment_family,
        model_family,
        original_config,
        strategy,
        n_features,
        params_recorded,
        source_id,
        source_row,
        has_validation,
        params_source_id=""
    ):

        source = SOURCE_MAP.loc[source_id]

        model_rows.append({
            "model_id": model_id,
            "owner": owner,
            "experiment_family": experiment_family,
            "model_family": model_family,
            "original_config": original_config,
            "strategy": strategy,
            "n_features": int(n_features),
            "params_recorded": params_recorded,
            "params_source_id": params_source_id,
            "source_id": source_id,
            "source_row": int(source_row),
            "source_path": source["relative_path"],
            "source_sha256": source["sha256"],
            "has_validation": bool(has_validation),
            "status": "HISTORICAL_NON_TEST"
        })


    # 5. B: Imbalance configurations

    print("\n=== B: IMBALANCE CONFIGURATIONS ===")

    assert not b_grid.duplicated(["model", "strategy"]).any()

    config_map = {}

    for config, model, strategy in b_lock["configs"]:

        key = (model, strategy)

        assert key not in config_map

        config_map[key] = config

    val_keys = set(zip(b_val["model"], b_val["strategy"]))

    assert set(config_map) == val_keys

    for i, row in b_grid.iterrows():

        model = row["model"]
        strategy = row["strategy"]
        key = (model, strategy)

        params = b_lock["hyperparams"].get(model)

        if params is None:

            params_text = "Not recorded in lock"
            params_source = ""

        else:

            params_text = json.dumps(params, sort_keys=True)
            params_source = "SRC012"

        add_model(
            model_id=f"B__{model}__{strategy}",
            owner="B",
            experiment_family="imbalance_grid",
            model_family=B_FAMILY_MAP[model],
            original_config=config_map.get(key, model),
            strategy=strategy,
            n_features=21,
            params_recorded=params_text,
            source_id="SRC002",
            source_row=i + 1,
            has_validation=key in val_keys,
            params_source_id=params_source
        )

    print("B imbalance configurations:", 26)

    # 6. B: Feature selection

    print("\n=== B: FEATURE CONFIGURATIONS ===")

    assert set(b4_val["feature_set"]) == {
        "all_features",
        "selected",
        "k15_xgb_gain"
    }

    b4_unique = b4_val.drop_duplicates(
        subset=["feature_set"]
    )

    assert len(b4_unique) == 3

    for i, row in b4_unique.iterrows():

        feature_set = row["feature_set"]

        feature_list = b4_lock["feature_sets"][feature_set]

        assert len(feature_list) == row["n_features"]
        assert row["strategy"] == b4_lock["strategy"]

        add_model(
            model_id=f"B__B4__{feature_set}",
            owner="B",
            experiment_family="feature_selection",
            model_family="XGBoost",
            original_config=feature_set,
            strategy=row["strategy"],
            n_features=row["n_features"],
            params_recorded=json.dumps(
                b4_lock["xgb_params"],
                sort_keys=True
            ),
            source_id="SRC008",
            source_row=i + 1,
            has_validation=True,
            params_source_id="SRC013"
        )

    print("B feature configurations:", 3)

    # 7. C: Baseline and Xie

    print("\n=== C: BASELINE AND XIE ===")

    c_sources = {
        "SRC014": "baseline",
        "SRC015": "baseline_appendix",
        "SRC020": "xie_appendix",
        "SRC021": "xie_appendix"
    }

    for source_id, family in c_sources.items():

        df = read_csv_source(source_id)

        assert set(df["split"]) == {"val"}
        assert set(df["seed"]) == {42}
        assert set(df["n_folds"]) == {3}
        assert not df["model"].duplicated().any()

        for i, row in df.iterrows():

            original_model = row["model"]

            add_model(
                model_id=f"C__{original_model}",
                owner="C",
                experiment_family=family,
                model_family=C_FAMILY_MAP[original_model],
                original_config=original_model,
                strategy=row["strategy"],
                n_features=21,
                params_recorded=str(row["params"]),
                source_id=source_id,
                source_row=i + 1,
                has_validation=True,
                params_source_id=source_id
            )

        print(source_id, "|", len(df), "models")

    # 8. Consolidate registry

    model_registry = pd.DataFrame(model_rows)

    assert len(model_registry) == 38
    assert model_registry["model_id"].is_unique
    assert set(model_registry["owner"]) == {"B", "C"}
    assert (model_registry["n_features"] > 0).all()

    assert model_registry["source_id"].isin(
        registry["source_id"]
    ).all()

    print("\n[PASS] Unique model identifiers")

    # 9. Verify counts

    assert (
        model_registry["owner"].value_counts().to_dict()
        == {"B": 29, "C": 9}
    )

    assert model_registry["has_validation"].sum() == 19

    print("[PASS] Model registry counts")

    # 10. F1/F2 links

    c_f2 = read_csv_source("SRC014")
    c_f1 = read_csv_source("SRC016")

    assert set(c_f2["model"]) == set(c_f1["model"])

    assert set(c_f2["threshold_rule"]) == {
        "thr_val_maxF2"
    }

    assert set(c_f1["threshold_rule"]) == {
        "thr_val_maxF1"
    }

    paired_threshold = c_f2.merge(
        c_f1,
        on=["model", "strategy", "params"],
        suffixes=("_F2", "_F1"),
        validate="one_to_one"
    )

    assert len(paired_threshold) == 3

    for metric in ["AP", "auc_roc"]:

        np.testing.assert_allclose(
            paired_threshold[f"{metric}_F2"],
            paired_threshold[f"{metric}_F1"],
            rtol=0,
            atol=1e-12
        )

    print("[PASS] F1/F2 configuration links")

    # 11. Verify source-row provenance

    for row in model_registry.itertuples(index=False):

        original = read_csv_source(row.source_id)

        assert 1 <= row.source_row <= len(original)

        record = original.iloc[row.source_row - 1]

        assert record["strategy"] == row.strategy

        if row.source_id == "SRC002":

            expected_id = (
                f"B__{record['model']}__{record['strategy']}"
            )

        elif row.source_id == "SRC008":

            expected_id = f"B__B4__{record['feature_set']}"

        else:

            expected_id = f"C__{record['model']}"

        assert expected_id == row.model_id

    print("[PASS] Source-row provenance: 38/38")

    # 12. Save

    model_registry.to_csv(
        OUT / "model_registry.csv",
        index=False
    )

    assert (OUT / "model_registry.csv").is_file()

    # 13. Display results

    print("\n=== REGISTRY BY OWNER ===")

    display(
        model_registry.groupby("owner")
        .size()
        .rename("n_configurations")
        .reset_index()
    )

    print("\n=== REGISTRY BY EXPERIMENT ===")

    display(
        model_registry.groupby(["owner", "experiment_family"])
        .size()
        .rename("n_configurations")
        .reset_index()
    )

    print("\n=== MODEL REGISTRY ===")

    display(
        model_registry[[
            "model_id",
            "owner",
            "model_family",
            "strategy",
            "n_features",
            "has_validation",
            "source_id",
            "source_row"
        ]]
    )

    print("\n=== REGISTRY SUMMARY ===")

    print("Total configurations:", len(model_registry))

    print(
        "Configurations with validation:",
        int(model_registry["has_validation"].sum())
    )

    print("Unique model IDs:", model_registry["model_id"].nunique())

    print("Source provenance verified: 38/38")
    print("Model training: NO")
    print("Test evaluation: NO")
    print("Output:", OUT / "model_registry.csv")

    print("\nS9.5 CELL 3: PASS")

    # END NOTEBOOK CELL 3


    # BEGIN NOTEBOOK CELL 4


    # ==========================================
    # S9.5 - CELL 4: CROSS-VALIDATION AGGREGATION
    # ==========================================

    import ast
    import re

    # 1. C tuning source map

    C_TUNING = {
        "C__B1_logistic": "SRC017",
        "C__B2_random_forest": "SRC018",
        "C__APPX_decision_tree": "SRC019",
        "C__X1_gaussian_nb": "SRC022",
        "C__X2_mlp": "SRC023",
        "C__X3_svm_linear": "SRC024",
        "C__X4_logistic_SMOTE": "SRC025",
        "C__X5_random_forest_SMOTE": "SRC026"
    }

    M = model_registry.set_index("model_id")

    cv_rows = []
    tuning_rows = []

    # 2. Provenance helper


    def provenance(sid, i):

        src = SOURCE_MAP.loc[sid]

        return {
            "source_id": sid,
            "source_row": int(i) + 1,
            "source_path": src["relative_path"],
            "source_sha256": src["sha256"]
        }


    # 3. Standardize CV records


    def cv_record(mid, sid, i, row, kind):

        rec = {
            "model_id": mid,
            "owner": M.loc[mid, "owner"],
            "model_family": M.loc[mid, "model_family"],
            "strategy": M.loc[mid, "strategy"],
            "cv_kind": kind,
            "split": "train_cv",
            "cv_AP": float(row["cv_AP"]),
            "cv_AP_std": float(row["cv_AP_std"]),
            "cv_ROC_AUC": row.get("cv_auc", np.nan),
            "cv_recall_1": row.get("cv_recall_1", np.nan),
            "cv_f1_macro": row.get("cv_f1_macro", np.nan),
            **provenance(sid, i)
        }

        assert 0 <= rec["cv_AP"] <= 1
        assert np.isfinite(rec["cv_AP_std"])
        assert rec["cv_AP_std"] >= 0

        return rec


    # 4. B imbalance CV

    print("=== B: IMBALANCE GRID CV ===")

    for i, row in read_csv_source("SRC002").iterrows():

        mid = f"B__{row['model']}__{row['strategy']}"

        cv_rows.append(
            cv_record(mid, "SRC002", i, row, "imbalance_grid")
        )

    assert len(cv_rows) == 26

    print("[PASS] B imbalance CV: 26/26")

    # 5. C recorded tuning selections

    print("\n=== C: HYPERPARAMETER TUNING ===")

    for mid, sid in C_TUNING.items():

        t = read_csv_source(sid)

        assert len(t) == 10

        raw = str(M.loc[mid, "params_recorded"])

        cleaned = re.sub(
            r"np\.float64\(([-+0-9.eE]+)\)",
            r"\1",
            raw
        )

        params = ast.literal_eval(cleaned)

        keys = [
            k for k in t.columns
            if k not in {"cv_AP", "cv_AP_std", "seconds"}
        ]

        assert set(keys) == set(params)

        mask = np.ones(len(t), dtype=bool)

        for k in keys:

            v = params[k]

            if isinstance(v, tuple):

                same = (
                    t[k]
                    .map(ast.literal_eval)
                    .map(lambda x: x == v)
                    .to_numpy()
                )

            elif v is None:

                same = t[k].isna().to_numpy()

            else:

                same = np.isclose(
                    t[k].to_numpy(dtype=float),
                    float(v),
                    rtol=1e-9,
                    atol=0
                )

            mask &= same

        assert mask.sum() == 1, mid

        selected_i = t.index[mask][0]

        row = t.loc[selected_i]

        max_ap = t["cv_AP"].max()

        assert np.isclose(
            row["cv_AP"],
            max_ap,
            rtol=0,
            atol=1e-12
        ), mid

        rec = cv_record(
            mid,
            sid,
            selected_i,
            row,
            "hyperparameter_tuning"
        )

        rec.update({
            "params_recorded": raw,
            "params_source_id": M.loc[mid, "source_id"],
            "params_source_row": int(M.loc[mid, "source_row"]),
            "n_tied_max_AP_rounded": int(
                (t["cv_AP"] == max_ap).sum()
            )
        })

        cv_rows.append(rec)

        for i, candidate in t.iterrows():

            tuning_rows.append({
                "model_id": mid,
                **candidate.to_dict(),
                "is_recorded_selection": bool(i == selected_i),
                **provenance(sid, i)
            })

        print(
            f"[PASS] {mid}: "
            f"AP={row['cv_AP']:.4f}, "
            f"row={selected_i + 1}, "
            f"ties={rec['n_tied_max_AP_rounded']}"
        )

    # 6. Consolidate CV

    cv_summary = pd.DataFrame(cv_rows)

    c_tuning_candidates = pd.DataFrame(tuning_rows)

    assert len(cv_summary) == 34
    assert cv_summary["model_id"].is_unique
    assert len(c_tuning_candidates) == 80

    assert c_tuning_candidates[
        "is_recorded_selection"
    ].sum() == 8

    print("\n[PASS] Consolidated CV tables")

    # 7. B tuning parameters

    print("\n=== B: TUNING PARAMETERS ===")

    b_tuning_recorded = read_csv_source(
        "SRC001"
    ).rename(columns={"Unnamed: 0": "model"})

    assert set(b_tuning_recorded["model"]) == {
        "primary",
        "random_forest"
    }

    for i, row in b_tuning_recorded.iterrows():

        params = b_lock["hyperparams"][row["model"]]

        for k, v in params.items():

            assert np.isclose(row[k], v)

        for k, v in provenance("SRC001", i).items():

            b_tuning_recorded.loc[i, k] = v

        b_tuning_recorded.loc[i, "model_id"] = (
            f"B__{row['model']}__none"
        )

    print("[PASS] B recorded tuning parameters: 2/2")

    # 8. Missing CV coverage

    missing_cv = model_registry.loc[
        ~model_registry["model_id"].isin(cv_summary["model_id"]),
        ["model_id", "owner", "experiment_family"]
    ].copy()

    assert set(missing_cv["model_id"]) == {
        "C__B0_dummy",
        "B__B4__all_features",
        "B__B4__selected",
        "B__B4__k15_xgb_gain"
    }

    missing_cv["reason"] = np.where(
        missing_cv["owner"] == "B",
        "Feature CV handled separately",
        "No C dummy CV source in package"
    )

    # 9. Save outputs

    tables = {
        "cv_summary": cv_summary,
        "c_tuning_candidates": c_tuning_candidates,
        "b_tuning_recorded": b_tuning_recorded,
        "cv_coverage_pending": missing_cv
    }

    for name, df in tables.items():

        path = OUT / f"{name}.csv"

        df.to_csv(path, index=False)

        assert path.is_file()

        print("[SAVED]", path.name, "| Rows:", len(df))

    # 10. Display results

    print("\n=== CV SUMMARY ===")

    display(
        cv_summary[[
            "model_id",
            "cv_kind",
            "cv_AP",
            "cv_AP_std",
            "source_id",
            "source_row"
        ]]
    )

    print("\n=== COVERAGE ===")

    display(missing_cv)

    print("\n=== FINAL SUMMARY ===")

    print("CV summary:", len(cv_summary))
    print("C tuning candidates:", len(c_tuning_candidates))
    print("B recorded tuning rows:", len(b_tuning_recorded))

    print(
        "Recorded C selections:",
        int(c_tuning_candidates["is_recorded_selection"].sum())
    )

    print("Model training: NO")
    print("Test evaluation: NO")

    print("\nS9.5 CELL 4: PASS")

    # END NOTEBOOK CELL 4


    # BEGIN NOTEBOOK CELL 5


    # ==========================================
    # S9.5 - CELL 5: VALIDATION AGGREGATION
    # ==========================================

    VAL_SOURCES = [
        "SRC004",
        "SRC008",
        "SRC014",
        "SRC015",
        "SRC016",
        "SRC020",
        "SRC021"
    ]

    METRIC_COLUMNS = [
        "AP",
        "auc_roc",
        "accuracy",
        "balanced_acc",
        "f1_macro",
        "f1_class1",
        "f2_class1",
        "precision_1",
        "recall_1",
        "specificity",
        "mcc"
    ]

    CM_COLUMNS = ["n_tp", "n_fp", "n_fn", "n_tn"]

    ROUNDING_TOL = 5.01e-5

    EXPECTED_VAL_SIZE = 38052
    EXPECTED_POSITIVE = 5302
    EXPECTED_NEGATIVE = 32750

    validation_rows = []
    recomputed_rows = []

    # 1. Safe division


    def safe_div(a, b):

        return float(a / b) if b != 0 else 0.0


    # 2. Recompute metrics


    def recompute_metrics(tp, fp, fn, tn):

        precision = safe_div(tp, tp + fp)

        recall = safe_div(tp, tp + fn)

        specificity = safe_div(tn, tn + fp)

        f1_positive = safe_div(
            2 * tp,
            2 * tp + fp + fn
        )

        f1_negative = safe_div(
            2 * tn,
            2 * tn + fp + fn
        )

        f2_positive = safe_div(
            5 * tp,
            5 * tp + 4 * fn + fp
        )

        mcc_denominator = np.sqrt(
            float(tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
        )

        return {
            "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
            "balanced_acc": (recall + specificity) / 2,
            "f1_macro": (f1_positive + f1_negative) / 2,
            "f1_class1": f1_positive,
            "f2_class1": f2_positive,
            "precision_1": precision,
            "recall_1": recall,
            "specificity": specificity,
            "mcc": safe_div(
                tp * tn - fp * fn,
                mcc_denominator
            )
        }


    # 3. Model identification


    def get_validation_model_id(sid, row):

        if sid == "SRC004":

            return f"B__{row['model']}__{row['strategy']}"

        elif sid == "SRC008":

            return f"B__B4__{row['feature_set']}"

        else:

            return f"C__{row['model']}"


    # 4. Read and verify validation

    print("=== VALIDATION SOURCE AUDIT ===")

    for sid in VAL_SOURCES:

        df = read_csv_source(sid)

        required = set(
            METRIC_COLUMNS + CM_COLUMNS
            + ["threshold", "threshold_rule"]
        )

        assert required.issubset(df.columns)

        if "split" in df.columns:

            assert set(df["split"]) == {"val"}

        if "seed" in df.columns:

            assert set(df["seed"]) == {42}

        if "n_folds" in df.columns:

            assert set(df["n_folds"]) == {3}

        for i, row in df.iterrows():

            mid = get_validation_model_id(sid, row)

            assert mid in M.index

            model_info = M.loc[mid]

            assert bool(model_info["has_validation"])
            assert row["strategy"] == model_info["strategy"]

            if sid == "SRC004":

                assert row["config"] == model_info["original_config"]

            if sid == "SRC008":

                assert row["n_features"] == model_info["n_features"]

            if model_info["owner"] == "C":

                assert str(row["params"]) == str(
                    model_info["params_recorded"]
                )

            # Verify metric domains

            for metric in METRIC_COLUMNS:

                value = float(row[metric])

                assert np.isfinite(value)

                if metric == "mcc":

                    assert -1 <= value <= 1

                else:

                    assert 0 <= value <= 1

            # Verify confusion matrix

            counts = row[CM_COLUMNS].to_numpy(dtype=float)

            assert np.isfinite(counts).all()
            assert (counts >= 0).all()
            assert np.equal(counts, np.floor(counts)).all()

            tp, fp, fn, tn = map(int, counts)

            assert tp + fp + fn + tn == EXPECTED_VAL_SIZE

            assert tp + fn == EXPECTED_POSITIVE
            assert fp + tn == EXPECTED_NEGATIVE

            threshold = float(row["threshold"])

            assert 0 <= threshold <= 1

            rule = row["threshold_rule"]

            assert rule in {
                "thr_0.5",
                "thr_val_maxF1",
                "thr_val_maxF2"
            }

            if rule == "thr_0.5":

                assert np.isclose(threshold, 0.5)

            # Recompute nine metrics

            recalculated = recompute_metrics(tp, fp, fn, tn)

            for metric, value in recalculated.items():

                recorded = float(row[metric])

                error = abs(recorded - value)

                assert error <= ROUNDING_TOL, (
                    sid, i + 1, metric, error
                )

                recomputed_rows.append({
                    "model_id": mid,
                    "threshold_rule": rule,
                    "metric": metric,
                    "recorded_value": recorded,
                    "recomputed_value": value,
                    "absolute_error": error,
                    "status": "PASS",
                    **provenance(sid, i)
                })

            record = {
                "model_id": mid,
                "owner": model_info["owner"],
                "experiment_family": model_info["experiment_family"],
                "model_family": model_info["model_family"],
                "original_config": model_info["original_config"],
                "strategy": model_info["strategy"],
                "n_features": int(model_info["n_features"]),
                "params_recorded": model_info["params_recorded"],
                "split": "validation",
                "split_evidence": (
                    "explicit_csv_column"
                    if "split" in df.columns
                    else "source_filename_and_context"
                ),
                "threshold_rule": rule,
                "threshold": threshold,
                "n_validation": EXPECTED_VAL_SIZE,
                "n_positive": EXPECTED_POSITIVE,
                "n_negative": EXPECTED_NEGATIVE,
                "status": "HISTORICAL_NON_TEST",
                **{
                    metric: float(row[metric])
                    for metric in METRIC_COLUMNS
                },
                "n_tp": tp,
                "n_fp": fp,
                "n_fn": fn,
                "n_tn": tn,
                **provenance(sid, i)
            }

            validation_rows.append(record)

        print(f"[PASS] {sid}: {len(df)} rows")

    # 5. Consolidate

    validation_summary = pd.DataFrame(validation_rows)

    validation_metrics_recomputed = pd.DataFrame(
        recomputed_rows
    )

    assert len(validation_summary) == 32
    assert len(validation_metrics_recomputed) == 288

    assert not validation_summary.duplicated(
        ["model_id", "threshold_rule"]
    ).any()

    expected_ids = set(
        model_registry.loc[
            model_registry["has_validation"],
            "model_id"
        ]
    )

    assert set(validation_summary["model_id"]) == expected_ids

    print("\n[PASS] Validation coverage: 19/19")
    print("[PASS] Confusion matrix metrics: 288/288")

    # 6. Separate threshold rules

    validation_main = validation_summary.loc[
        validation_summary["threshold_rule"] == "thr_val_maxF2"
    ].copy()

    validation_default = validation_summary.loc[
        validation_summary["threshold_rule"] == "thr_0.5"
    ].copy()

    validation_f1 = validation_summary.loc[
        validation_summary["threshold_rule"] == "thr_val_maxF1"
    ].copy()

    assert len(validation_main) == 19
    assert len(validation_default) == 10
    assert len(validation_f1) == 3

    assert validation_main["model_id"].is_unique
    assert set(validation_main["model_id"]) == expected_ids

    print("[PASS] Threshold rule separation")

    # 7. Threshold-independent metrics

    for mid, group in validation_summary.groupby("model_id"):

        for metric in ["AP", "auc_roc"]:

            np.testing.assert_allclose(
                group[metric],
                group[metric].iloc[0],
                rtol=0,
                atol=1e-12
            )

    print(
        "[PASS] AP and ROC-AUC invariant "
        "across recorded thresholds"
    )

    # 8. Compare F2/F1 rules

    for reference, metric in [
        (validation_default, "f2_class1"),
        (validation_f1, "f1_class1")
    ]:

        paired = validation_main.merge(
            reference,
            on="model_id",
            suffixes=("_F2", "_reference"),
            validate="one_to_one"
        )

        if metric == "f2_class1":

            assert (
                paired[f"{metric}_F2"] + ROUNDING_TOL
                >= paired[f"{metric}_reference"]
            ).all()

        else:

            assert (
                paired[f"{metric}_reference"] + ROUNDING_TOL
                >= paired[f"{metric}_F2"]
            ).all()

    print("[PASS] Recorded threshold comparisons")

    # 9. Document overlapping B records

    overlap_audit = []

    for rule in ["thr_0.5", "thr_val_maxF2"]:

        a = validation_summary.loc[
            (
                validation_summary["model_id"]
                == "B__primary__class_weight"
            )
            & (validation_summary["threshold_rule"] == rule)
        ].iloc[0]

        b = validation_summary.loc[
            (
                validation_summary["model_id"]
                == "B__B4__all_features"
            )
            & (validation_summary["threshold_rule"] == rule)
        ].iloc[0]

        compare_columns = (
            METRIC_COLUMNS + CM_COLUMNS + ["threshold"]
        )

        np.testing.assert_allclose(
            a[compare_columns].to_numpy(dtype=float),
            b[compare_columns].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12
        )

        overlap_audit.append({
            "model_id_A": a["model_id"],
            "model_id_B": b["model_id"],
            "threshold_rule": rule,
            "source_id_A": a["source_id"],
            "source_row_A": a["source_row"],
            "source_id_B": b["source_id"],
            "source_row_B": b["source_row"],
            "recorded_metrics_match": True,
            "prediction_identity_verified": False,
            "interpretation": (
                "Matching reported metrics; "
                "not independent evidence"
            )
        })

    validation_overlap_audit = pd.DataFrame(overlap_audit)

    print("[PASS] B all-feature reference overlap documented")

    # 10. Save tables

    tables = {
        "validation_summary": validation_summary,
        "validation_main_F2": validation_main,
        "validation_default_05": validation_default,
        "validation_sensitivity_F1": validation_f1,
        "validation_metrics_recomputed":
            validation_metrics_recomputed,
        "validation_overlap_audit": validation_overlap_audit
    }

    for name, df in tables.items():

        path = OUT / f"{name}.csv"

        df.to_csv(path, index=False)

        print("[SAVED]", path.name, "| Rows:", len(df))

    # 11. Display results

    print("\n=== MAIN VALIDATION RESULTS: MAX F2 ===")

    display(
        validation_main[[
            "model_id",
            "AP",
            "auc_roc",
            "threshold",
            "f2_class1",
            "precision_1",
            "recall_1",
            "specificity",
            "source_id",
            "source_row"
        ]]
    )

    print("\n=== THRESHOLD RULE COVERAGE ===")

    display(
        validation_summary.groupby("threshold_rule")
        .size()
        .rename("n_rows")
        .reset_index()
    )

    print("\n=== METRIC RECOMPUTATION ===")

    display(
        validation_metrics_recomputed.groupby("metric")[
            "absolute_error"
        ]
        .agg(["count", "max"])
        .reset_index()
    )

    max_error = float(
        validation_metrics_recomputed["absolute_error"].max()
    )

    print("\n=== FINAL SUMMARY ===")

    print("Validation records:", len(validation_summary))
    print("Max F2 records:", len(validation_main))
    print("Default 0.5 records:", len(validation_default))
    print("Max F1 records:", len(validation_f1))
    print("Unique configurations:", len(expected_ids))

    print(
        "Recomputed metrics:",
        len(validation_metrics_recomputed),
        "/ 288"
    )

    print("Maximum absolute error:", f"{max_error:.10f}")

    print("Documented overlap rows:", len(validation_overlap_audit))

    print("Model training: NO")
    print("Test evaluation: NO")

    print("\nS9.5 CELL 5: PASS")

    # END NOTEBOOK CELL 5


    # BEGIN NOTEBOOK CELL 6


    # ==========================================
    # S9.5 - CELL 6: SENSITIVITY ANALYSIS
    # ==========================================

    import time

    cell6_start = time.perf_counter()

    sensitivity_checks = {}


    def sensitivity_pass(name):

        sensitivity_checks[name] = "PASS"

        elapsed = time.perf_counter() - cell6_start

        print(
            f"[PASS] {name} | Elapsed: {elapsed:.2f}s",
            flush=True
        )


    def source_table(sid):

        df = read_csv_source(sid).copy()

        for i in df.index:

            for key, value in provenance(sid, i).items():

                df.loc[i, key] = value

        return df


    # ==========================================
    # 1. IMBALANCE STRATEGY
    # ==========================================

    print("=== IMBALANCE ANALYSIS ===", flush=True)

    imbalance_summary = cv_summary.loc[
        cv_summary["cv_kind"] == "imbalance_grid"
    ].copy()

    assert len(imbalance_summary) == 26

    baseline_ap = (
        imbalance_summary.loc[
            imbalance_summary["strategy"] == "none"
        ]
        .set_index("model_family")["cv_AP"]
    )

    imbalance_summary["reference_AP_none"] = (
        imbalance_summary["model_family"].map(baseline_ap)
    )

    imbalance_summary["delta_AP_vs_none"] = (
        imbalance_summary["cv_AP"]
        - imbalance_summary["reference_AP_none"]
    )

    assert imbalance_summary["reference_AP_none"].notna().all()

    sensitivity_pass("S01_imbalance_coverage")

    # Verify recorded strategy

    strategy_selection = source_table("SRC003")

    chosen = strategy_selection.iloc[0]

    assert chosen["chosen_strategy"] == b_lock["best_strategy"]

    primary_grid = read_csv_source("SRC002")

    primary_grid = primary_grid.loc[
        primary_grid["model"] == "primary"
    ].copy()

    max_macro_f1 = primary_grid["cv_f1_macro"].max()

    eligible = primary_grid.loc[
        primary_grid["cv_f1_macro"]
        >= max_macro_f1 - 0.01 - 1e-12
    ]

    selected_strategy = eligible.sort_values(
        ["cv_AP", "cv_recall_1"],
        ascending=[False, False]
    ).iloc[0]

    assert selected_strategy["strategy"] == chosen["chosen_strategy"]

    assert np.isclose(
        selected_strategy["cv_AP"],
        chosen["cv_AP"]
    )

    assert np.isclose(
        selected_strategy["cv_recall_1"],
        chosen["cv_recall_1"]
    )

    sensitivity_pass("S02_recorded_strategy_rule")


    # ==========================================
    # 2. FEATURE SELECTION CV
    # ==========================================

    print("\n=== FEATURE SELECTION ===", flush=True)

    feature_selection_summary = source_table("SRC006")

    assert len(feature_selection_summary) == 25

    assert not feature_selection_summary.duplicated(
        ["method", "k"]
    ).any()

    assert set(feature_selection_summary["k"]) == {
        5, 10, 15, 21
    }

    assert (
        feature_selection_summary["k"]
        == feature_selection_summary["n_features_kept"]
    ).all()

    full = feature_selection_summary.loc[
        (feature_selection_summary["method"] == "baseline")
        & (feature_selection_summary["k"] == 21)
    ].iloc[0]

    full_AP = float(full["cv_AP"])

    feature_selection_summary["delta_AP_vs_full21"] = (
        feature_selection_summary["cv_AP"] - full_AP
    )

    feature_selection_summary["within_AP_tolerance"] = (
        feature_selection_summary["delta_AP_vs_full21"]
        >= -0.01 - 1e-12
    )

    sensitivity_pass("S03_feature_cv_coverage")

    # Verify recorded feature selection

    feature_selected_config = source_table("SRC007")

    selected = feature_selected_config.iloc[0]

    assert selected["selected_method"] == (
        b4_lock["selected_method"]
    )

    assert selected["k"] == b4_lock["k"]
    assert selected["strategy"] == b4_lock["strategy"]

    matched = feature_selection_summary.loc[
        (
            feature_selection_summary["method"]
            == selected["selected_method"]
        )
        & (feature_selection_summary["k"] == selected["k"])
    ]

    assert len(matched) == 1

    matched = matched.iloc[0]

    for metric in ["cv_AP", "cv_recall_1", "cv_f1_macro"]:

        assert np.isclose(
            matched[metric],
            selected[metric]
        )

    assert np.isclose(selected["cv_AP_full21"], full_AP)

    assert np.isclose(
        selected["delta_AP_vs_full"],
        matched["delta_AP_vs_full21"]
    )

    assert matched["within_AP_tolerance"]

    eligible_k = feature_selection_summary.loc[
        feature_selection_summary["within_AP_tolerance"],
        "k"
    ]

    assert eligible_k.min() == selected["k"]

    sensitivity_pass("S04_recorded_feature_selection")


    # ==========================================
    # 3. FEATURE SELECTION VALIDATION
    # ==========================================

    feature_validation_summary = validation_main.loc[
        validation_main["experiment_family"]
        == "feature_selection"
    ].copy()

    assert len(feature_validation_summary) == 3

    assert set(feature_validation_summary["n_features"]) == {
        10, 15, 21
    }

    full_val_AP = float(
        feature_validation_summary.loc[
            feature_validation_summary["n_features"] == 21,
            "AP"
        ].iloc[0]
    )

    feature_validation_summary["delta_AP_vs_full21"] = (
        feature_validation_summary["AP"] - full_val_AP
    )

    sensitivity_pass("S05_feature_validation")


    # ==========================================
    # 4. SELECTED FEATURES AND IMPORTANCE
    # ==========================================

    f10 = source_table("SRC009")
    f15 = source_table("SRC010")

    assert len(f10) == 10
    assert len(f15) == 15

    assert f10["feature"].is_unique
    assert f15["feature"].is_unique

    assert f10["feature"].tolist() == (
        b4_lock["feature_sets"]["selected"]
    )

    assert f15["feature"].tolist() == (
        b4_lock["feature_sets"]["k15_xgb_gain"]
    )

    assert set(f10["feature"]).issubset(f15["feature"])

    selected_feature_registry = pd.concat(
        [f10, f15],
        ignore_index=True
    )

    sensitivity_pass("S06_selected_feature_lists")

    permutation_importance = source_table("SRC011")

    assert len(permutation_importance) == 10
    assert permutation_importance["feature"].is_unique

    assert set(permutation_importance["feature"]) == (
        set(f10["feature"])
    )

    assert (permutation_importance["imp_std"] >= 0).all()

    assert np.isfinite(
        permutation_importance[
            ["imp_mean", "imp_std"]
        ].to_numpy()
    ).all()

    sensitivity_pass("S07_permutation_importance")


    # ==========================================
    # 5. FEATURE METHOD OVERLAP
    # ==========================================

    feature_overlap_rows = []

    for method_a, method_b in [
        ("xgb_gain", "rfe_xgb"),
        ("variance", "corr_dedup")
    ]:

        left = feature_selection_summary.loc[
            feature_selection_summary["method"] == method_a
        ]

        right = feature_selection_summary.loc[
            feature_selection_summary["method"] == method_b
        ]

        paired = left.merge(
            right,
            on="k",
            suffixes=("_A", "_B"),
            validate="one_to_one"
        )

        assert len(paired) == 4

        for _, row in paired.iterrows():

            assert np.isclose(
                row["cv_AP_A"],
                row["cv_AP_B"],
                rtol=0,
                atol=1e-12
            )

            feature_overlap_rows.append({
                "method_A": method_a,
                "method_B": method_b,
                "k": row["k"],
                "cv_AP_A": row["cv_AP_A"],
                "cv_AP_B": row["cv_AP_B"],
                "source_id": "SRC006",
                "source_row_A": row["source_row_A"],
                "source_row_B": row["source_row_B"],
                "recorded_AP_match": True,
                "prediction_identity_verified": False
            })

    feature_method_overlap = pd.DataFrame(
        feature_overlap_rows
    )

    assert len(feature_method_overlap) == 8

    sensitivity_pass("S08_feature_method_overlap")


    # ==========================================
    # 6. THRESHOLD SENSITIVITY
    # ==========================================

    print("\n=== THRESHOLD SENSITIVITY ===", flush=True)

    threshold_rows = []

    for ref_table, reference_name in [
        (validation_default, "thr_0.5"),
        (validation_f1, "thr_val_maxF1")
    ]:

        paired = validation_main.merge(
            ref_table,
            on="model_id",
            suffixes=("_F2", "_ref"),
            validate="one_to_one"
        )

        for _, row in paired.iterrows():

            record = {
                "model_id": row["model_id"],
                "owner": row["owner_F2"],
                "comparison": f"{reference_name}_to_maxF2",
                "threshold_reference": row["threshold_ref"],
                "threshold_F2": row["threshold_F2"],
                "source_id_reference": row["source_id_ref"],
                "source_row_reference": row["source_row_ref"],
                "source_id_F2": row["source_id_F2"],
                "source_row_F2": row["source_row_F2"]
            }

            for metric in [
                "AP",
                "auc_roc",
                "f1_class1",
                "f2_class1",
                "precision_1",
                "recall_1",
                "specificity",
                "mcc",
                "n_tp",
                "n_fp",
                "n_fn",
                "n_tn"
            ]:

                record[f"{metric}_reference"] = row[f"{metric}_ref"]

                record[f"{metric}_F2"] = row[f"{metric}_F2"]

                record[f"delta_{metric}"] = (
                    row[f"{metric}_F2"] - row[f"{metric}_ref"]
                )

            threshold_rows.append(record)

    threshold_sensitivity_summary = pd.DataFrame(
        threshold_rows
    )

    assert len(threshold_sensitivity_summary) == 13

    assert np.allclose(
        threshold_sensitivity_summary["delta_AP"],
        0,
        rtol=0,
        atol=1e-12
    )

    assert np.allclose(
        threshold_sensitivity_summary["delta_auc_roc"],
        0,
        rtol=0,
        atol=1e-12
    )

    sensitivity_pass("S09_threshold_sensitivity")


    # ==========================================
    # 7. GROUP-VECTOR CV
    # ==========================================

    print("\n=== GROUP-VECTOR CV ===", flush=True)

    b_group = source_table("SRC005")
    c_group = source_table("SRC027")
    c_folds = source_table("SRC028")
    c_comparison = source_table("SRC029")

    s7_config = read_json_source("SRC034")

    assert s7_config["scope"] == "train_only"
    assert s7_config["seed"] == 42
    assert s7_config["n_folds"] == 3

    assert not s7_config["retuning"]
    assert not s7_config["holdout_test_used_for_cv"]

    assert len(b_group) == 2
    assert len(c_group) == 4
    assert len(c_folds) == 12
    assert len(c_comparison) == 2

    group_rows = []

    # B: recorded summary

    method_map = {
        "stratified (mac dinh)": "StratifiedKFold",
        "group_split (theo nhom vector trung)": "GroupKFold"
    }

    for _, row in b_group.iterrows():

        group_rows.append({
            "model_id": "B__primary__class_weight",
            "owner": "B",
            "method": method_map[row["split_type"]],
            "cv_AP": row["cv_AP"],
            "cv_AP_std": row["cv_AP_std"],
            "cv_AUC": np.nan,
            "cv_AUC_std": np.nan,
            "n_groups": row["n_groups"],
            "fold_level_available": False,
            "source_id": row["source_id"],
            "source_row": row["source_row"],
            "source_sha256": row["source_sha256"]
        })

    # C: recompute from fold-level results

    for _, row in c_group.iterrows():

        subset = c_folds.loc[
            (c_folds["model"] == row["model"])
            & (c_folds["method"] == row["method"])
        ]

        assert len(subset) == 3
        assert set(subset["fold"]) == {1, 2, 3}

        assert (
            subset["n_train"] + subset["n_eval"]
            == s7_config["n_train"]
        ).all()

        actual = [
            subset["AP"].mean(),
            subset["AP"].std(ddof=0),
            subset["ROC_AUC"].mean(),
            subset["ROC_AUC"].std(ddof=0)
        ]

        expected = [
            row["cv_AP"],
            row["cv_AP_std"],
            row["cv_AUC"],
            row["cv_AUC_std"]
        ]

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=1e-9
        )

        group_rows.append({
            "model_id": f"C__{row['model']}",
            "owner": "C",
            "method": row["method"],
            "cv_AP": row["cv_AP"],
            "cv_AP_std": row["cv_AP_std"],
            "cv_AUC": row["cv_AUC"],
            "cv_AUC_std": row["cv_AUC_std"],
            "n_groups": s7_config["n_groups"],
            "fold_level_available": True,
            "source_id": row["source_id"],
            "source_row": row["source_row"],
            "source_sha256": row["source_sha256"]
        })

    group_cv_summary = pd.DataFrame(group_rows)

    assert len(group_cv_summary) == 6

    sensitivity_pass("S10_group_cv_summary")


    # ==========================================
    # 8. GROUP CV DELTAS
    # ==========================================

    group_delta_rows = []

    for mid, subset in group_cv_summary.groupby("model_id"):

        assert len(subset) == 2

        strat = subset.loc[
            subset["method"] == "StratifiedKFold"
        ].iloc[0]

        group = subset.loc[
            subset["method"] == "GroupKFold"
        ].iloc[0]

        delta = group["cv_AP"] - strat["cv_AP"]

        if strat["owner"] == "C":

            original = c_comparison.loc[
                c_comparison["model"] == mid.removeprefix("C__")
            ].iloc[0]

            assert np.isclose(
                delta,
                original["delta_AP_group_minus_stratified"],
                rtol=0,
                atol=1e-9
            )

            historical = cv_summary.loc[
                cv_summary["model_id"] == mid,
                "cv_AP"
            ].iloc[0]

            assert np.isclose(
                round(strat["cv_AP"], 4),
                historical
            )

        group_delta_rows.append({
            "model_id": mid,
            "owner": strat["owner"],
            "stratified_cv_AP": strat["cv_AP"],
            "group_cv_AP": group["cv_AP"],
            "delta_AP": delta,
            "delta_AUC": group["cv_AUC"] - strat["cv_AUC"],
            "source_id_stratified": strat["source_id"],
            "source_row_stratified": strat["source_row"],
            "source_id_group": group["source_id"],
            "source_row_group": group["source_row"]
        })

    group_cv_comparison = pd.DataFrame(group_delta_rows)

    assert len(group_cv_comparison) == 3

    sensitivity_pass("S11_group_cv_deltas")


    # ==========================================
    # 9. GROUP OVERLAP AUDIT
    # ==========================================

    group_overlap_summary = source_table("SRC030")
    group_fold_audit = source_table("SRC031")

    assert len(group_overlap_summary) == 2
    assert len(group_fold_audit) == 6

    for _, row in group_overlap_summary.iterrows():

        subset = group_fold_audit.loc[
            group_fold_audit["method"] == row["method"]
        ]

        assert len(subset) == 3

        assert subset["n_eval"].sum() == s7_config["n_train"]

        assert subset["n_eval"].sum() == row["n_eval_total"]

        assert subset["shared_eval_rows"].sum() == (
            row["shared_eval_rows_total"]
        )

        assert subset["index_overlap"].sum() == (
            row["index_overlap_total"]
        )

        assert np.isclose(
            subset["shared_groups"].mean(),
            row["shared_groups_mean"]
        )

        assert np.isclose(
            subset["shared_eval_rows"].sum()
            / subset["n_eval"].sum(),
            row["shared_eval_rate"],
            rtol=0,
            atol=1e-9
        )

        assert row["index_overlap_total"] == 0

        if row["method"] == "GroupKFold":

            assert row["shared_eval_rows_total"] == 0

            assert (subset["shared_groups"] == 0).all()

    sensitivity_pass("S12_group_overlap_audit")


    # ==========================================
    # 10. SAVE OUTPUTS
    # ==========================================

    tables = {
        "imbalance_summary":
            imbalance_summary,

        "imbalance_selected_strategy":
            strategy_selection,

        "feature_selection_summary":
            feature_selection_summary,

        "feature_selected_config":
            feature_selected_config,

        "feature_validation_summary":
            feature_validation_summary,

        "selected_feature_registry":
            selected_feature_registry,

        "permutation_importance":
            permutation_importance,

        "feature_method_overlap":
            feature_method_overlap,

        "threshold_sensitivity_summary":
            threshold_sensitivity_summary,

        "group_cv_summary":
            group_cv_summary,

        "group_cv_comparison":
            group_cv_comparison,

        "group_overlap_summary":
            group_overlap_summary,

        "group_fold_audit":
            group_fold_audit
    }

    print("\n=== SAVED TABLES ===", flush=True)

    for name, df in tables.items():

        df.to_csv(
            OUT / f"{name}.csv",
            index=False
        )

        print(
            "[SAVED]",
            f"{name}.csv",
            "| Rows:",
            len(df),
            flush=True
        )

    assert len(sensitivity_checks) == 12

    with open(
        OUT / "sensitivity_checks.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sensitivity_checks,
            f,
            indent=2
        )


    # ==========================================
    # 11. DISPLAY KEY RESULTS
    # ==========================================

    print("\n=== FEATURE SELECTION SUMMARY ===")

    display(
        feature_selection_summary[[
            "method",
            "k",
            "cv_AP",
            "delta_AP_vs_full21",
            "within_AP_tolerance"
        ]]
    )

    print("\n=== THRESHOLD SENSITIVITY ===")

    display(
        threshold_sensitivity_summary[[
            "model_id",
            "comparison",
            "delta_AP",
            "delta_f2_class1",
            "delta_precision_1",
            "delta_recall_1"
        ]]
    )

    print("\n=== GROUP CV COMPARISON ===")

    display(group_cv_comparison)

    print("\n=== GROUP OVERLAP ===")

    display(group_overlap_summary)

    print("\n=== FINAL SUMMARY ===")

    print("Imbalance configurations:", len(imbalance_summary))

    print("Feature CV records:", len(feature_selection_summary))

    print(
        "Feature validation records:",
        len(feature_validation_summary)
    )

    print(
        "Threshold comparisons:",
        len(threshold_sensitivity_summary)
    )

    print("Group CV records:", len(group_cv_summary))

    print("Group CV comparisons:", len(group_cv_comparison))

    print("Tables saved:", len(tables))

    print("Sensitivity QA:", len(sensitivity_checks), "/ 12")

    print("Model training: NO")
    print("Test evaluation: NO")

    print(
        "Cell 6 execution time:",
        f"{time.perf_counter() - cell6_start:.2f} seconds"
    )

    print("\nS9.5 CELL 6: PASS")

    # END NOTEBOOK CELL 6



    return OUT


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate historical CV and "
            "validation results."
        )
    )

    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing verified input package"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="New or empty output directory"
    )

    args = parser.parse_args()

    result_dir = aggregate(
        input_root=args.input_root,
        out_dir=args.out_dir
    )

    print("\nAggregation output:", result_dir)
    print("S9.5 AGGREGATION: PASS")


if __name__ == "__main__":
    main()

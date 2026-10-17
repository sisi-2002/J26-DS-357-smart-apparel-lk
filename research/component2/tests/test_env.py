"""Compatibility smoke test for the pinned research stack."""
import io

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import pyarrow
import scipy
import shap
import sklearn
import xgboost
import yaml
from xgboost import XGBRegressor

N_ROWS = 200
N_FEATURES = 6


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] ** 2 + rng.normal(scale=0.1, size=N_ROWS)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    df["target"] = y
    return df


def test_versions(capsys):
    mods = [pd, np, sklearn, xgboost, shap, pyarrow, openpyxl, scipy, matplotlib, yaml]
    with capsys.disabled():
        print()
        for m in mods:
            print(f"{m.__name__:<12} {m.__version__}")
    assert all(m.__version__ for m in mods)


def test_xgboost_shap_tree_explainer():
    df = _synthetic_frame()
    X, y = df.drop(columns="target"), df["target"]
    model = XGBRegressor(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
    shap_values = shap.TreeExplainer(model).shap_values(X)
    assert shap_values.shape == (N_ROWS, X.shape[1])


def test_parquet_roundtrip(tmp_path):
    df = _synthetic_frame()
    path = tmp_path / "frame.parquet"
    df.to_parquet(path, engine="pyarrow")
    pd.testing.assert_frame_equal(pd.read_parquet(path, engine="pyarrow"), df)


def test_openpyxl_in_memory_sheet_read():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "orders"
    ws.append(["style", "waste_pct"])
    ws.append(["A1", 4.5])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="orders", engine="openpyxl")
    assert list(df.columns) == ["style", "waste_pct"]
    assert df.loc[0, "waste_pct"] == 4.5

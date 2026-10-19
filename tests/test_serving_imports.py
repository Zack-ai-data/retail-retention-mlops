"""A saved model must load without the training stack.

Models are pickled, and a pickle refers to its functions and classes by module path, so loading
one imports those modules. If a model referred to src.model, the API that serves it would have to
import matplotlib, MLflow logging and the database driver just to unpickle it (review finding F10).
"""

import subprocess
import sys

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from src import calibration, config, hurdle, model_search

# Importing any of these while loading a model means the serving image needs the training stack
TRAINING_ONLY_MODULES = [
    "matplotlib",
    "psycopg",
    "mlflow",
    "src.model",
    "src.model_search",
    "src.reporting",
    "src.segmentation",
    "src.backtesting",
    "src.database",
    "src.tracking",
]

LOAD_AND_PREDICT = """
import sys
import cloudpickle
import pandas as pd

with open(sys.argv[1], "rb") as handle:
    model = cloudpickle.load(handle)
X = pd.DataFrame({"a": [1.0, 5.0, 40.0], "b": [0.0, 3.0, -2.0]})
prediction = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
print("PREDICTIONS", len(prediction))
forbidden, forbidden_src = {FORBIDDEN}, {FORBIDDEN_SRC}
loaded = sorted(m for m in sys.modules if m.split(".")[0] in forbidden or m in forbidden_src)
print("LOADED", ",".join(loaded))
"""


def _data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 50, 200), "b": rng.normal(size=200)})
    y = (X["a"] + rng.normal(0, 10, 200) > 25).astype(int)
    return X, y


def _linear(name):
    candidate = next(c for c in model_search.CHURN_CANDIDATES if c.name == name)
    return candidate.factory({"C": 1.0})


def _models():
    X, y = _data()
    churn = _linear("logistic_regression").fit(X, y)  # contains the signed_log1p transform
    calibrated = calibration.PlattCalibratedClassifier(churn, 0.8, 0.1)
    spend = next(c for c in model_search.SPEND_CANDIDATES if c.name == "gamma_glm")
    spend_model = spend.factory({"alpha": 0.1}).fit(X[y == 1], 100 + X.loc[y == 1, "a"])
    return {
        "linear model with the signed_log1p transform": churn,
        "calibrated churn model": calibrated,
        "two-part value model": hurdle.HurdleValueModel(calibrated, spend_model),
    }


@pytest.fixture(scope="module")
def pickles(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pickles")
    paths = {}
    for name, fitted in _models().items():
        path = folder / f"{len(paths)}.pkl"
        path.write_bytes(cloudpickle.dumps(fitted))
        paths[name] = path
    return paths


def _load_in_a_fresh_interpreter(path):
    """Unpickle a model in a new process; return the training-only modules that got imported."""
    forbidden = {m.split(".")[0] for m in TRAINING_ONLY_MODULES if "." not in m}
    forbidden_src = {m for m in TRAINING_ONLY_MODULES if "." in m}
    code = LOAD_AND_PREDICT.replace("{FORBIDDEN}", repr(forbidden)).replace(
        "{FORBIDDEN_SRC}", repr(forbidden_src)
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "PREDICTIONS 3" in result.stdout
    loaded = next(line for line in result.stdout.splitlines() if line.startswith("LOADED"))
    return loaded.removeprefix("LOADED").strip()


@pytest.mark.parametrize("name", list(_models()))
def test_a_saved_model_loads_and_predicts_without_the_training_stack(pickles, name):
    assert _load_in_a_fresh_interpreter(pickles[name]) == "", name


def test_the_check_notices_a_model_that_refers_to_the_training_code(tmp_path):
    # control: a model whose transform lives in the training code must be flagged, otherwise the
    # test above could pass for the wrong reason
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from src import reporting

    X, y = _data()
    entangled = make_pipeline(FunctionTransformer(lambda frame: frame), LogisticRegression()).fit(
        X, y
    )
    entangled.helper = reporting.data_fingerprint  # a reference into the training stack
    path = tmp_path / "entangled.pkl"
    path.write_bytes(cloudpickle.dumps(entangled))

    loaded = _load_in_a_fresh_interpreter(path)

    assert "src.reporting" in loaded and "mlflow" in loaded


def test_the_transform_is_referenced_from_its_own_numpy_only_module(pickles):
    blob = pickles["linear model with the signed_log1p transform"].read_bytes()

    assert b"src.transforms" in blob
    assert b"src.model" not in blob.replace(b"src.model_search", b"")


def test_the_transforms_module_needs_only_numpy():
    code = (
        "import sys; import src.transforms; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'pandas','sklearn','scipy','matplotlib','mlflow','psycopg'}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"

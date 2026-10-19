"""Regression tests on the real dataset, run against the CURRENT cleaning and feature code.

The fixture copies the loaded raw table (`raw.online_retail`) into a scratch database, then runs
the current `sql/transform/*.sql` and quality checks there. So an accidental edit to the SQL, or to
the model code, changes the results below and fails these tests, without anyone having to re-run
the pipeline first. Nothing in the real warehouse is modified.

The expected values were verified by hand and independently (row counts, line types, snapshot
sizes, churn rates, recorded model metrics). If a change to the results is intentional, update the
values here in the same commit and say why in the commit message.

The tests need the real data loaded (`docker compose run --rm app`). Without it they are skipped;
pass `--require-warehouse` to make them fail instead of skip (for example in CI).
"""

import uuid

import psycopg
import pytest

from src import config, database, features, model, processor, tracking
from src.errors import PipelineError

pytestmark = pytest.mark.warehouse

RAW_ROWS = 1_067_371
STAGING_ROWS = 1_044_848
OVERLAP_ROWS_REMOVED = 22_523

LINE_TYPE_ROWS = {
    "sale": 1_015_088,
    "return": 17_974,
    "stock_adjustment": 5_973,
    "non_product": 5_813,
}

# snapshot_date: (split, customers, churn rate rounded to 4 decimals; None for the score snapshot)
# Re-pinned when money became NET of returns: churn means "no net-positive spend in the window".
# Customer counts are unchanged. Churn rose by exactly the customers who were "retained" only by
# an order they returned (+8, +6, +5, +8 customers), confirmed with an independent query:
#   2010-12-13: 0.6955 -> 0.6973   2011-03-13: 0.6472 -> 0.6485
#   2011-06-11: 0.6797 -> 0.6807   2011-09-09: 0.5663 -> 0.5679
# Re-pinned again when three earlier snapshots were added (step 2b); the other five are unchanged.
# 2010-03-18 has only 107 days of history, which is what history_days_available records.
SNAPSHOTS = {
    "2010-03-18": ("train", 1_925, 0.4016),
    "2010-06-16": ("train", 2_832, 0.5258),
    "2010-09-14": ("train", 3_364, 0.4260),
    "2010-12-13": ("train", 4_292, 0.6973),
    "2011-03-13": ("train", 4_581, 0.6485),
    "2011-06-11": ("train", 4_951, 0.6807),
    "2011-09-09": ("test", 5_253, 0.5679),
    "2011-12-10": ("score", 5_852, None),
}

# Customers who bought in the 180 days before each snapshot (the population the churn and value
# models train, are gated and report on; added in step 2c). Share of all customers at the test
# snapshot: 2,775 of 5,253 (53%).
ACTIONABLE = {
    "2010-03-18": 1_925,
    "2010-06-16": 2_625,
    "2010-09-14": 2_816,
    "2010-12-13": 3_468,
    "2011-03-13": 3_340,
    "2011-06-11": 2_640,
    "2011-09-09": 2_775,
    "2011-12-10": 3_471,
}

# Results of training on the full data. Retraining is deterministic (fixed seeds), so these should
# reproduce to many decimals; the tolerance only absorbs floating-point noise. If a change alters
# them, that is a change in model behaviour.
# Re-pinned when money became NET of returns (labels and features). Churn barely moved
# (test PR-AUC 0.8082 -> 0.8078). The value model changed a lot: a 77,183.60 order cancelled the
# same day had been counted as spend, so removing it changed which model won the cross-validation
# (gradient boosting -> Poisson regression) and the test RMSE fell from 2341 to 1907 against a
# naive baseline of 2070.
# Re-pinned for step 2c (models train and are evaluated on the actionable population only):
# churn test PR-AUC 0.8206 -> 0.6521 and recency baseline 0.7985 -> 0.5345, because the lapsed
# customers who made the old figures look good are no longer in the test set. The margin over the
# baseline grew from 0.022 to 0.118. The value model now LOSES the RMSE gate (3,036 vs naive
# 2,825) while still ranking customers better (Spearman 0.59 vs 0.44).
# Earlier, step 2b (three earlier snapshots, 365-day window features, seasonal terms): the
# churn model is now gradient boosting (test PR-AUC 0.8078 -> 0.8206, recency baseline 0.7985
# unchanged); the value model is Poisson gradient boosting again (RMSE 1907 -> 1906 vs naive 2070).
# Re-pinned for Phase 3 (evaluation and gate). Model fits are unchanged, so the model figures
# above did not move. What changed, and why:
#   - the naive value forecast is clipped at zero (spend cannot be negative; money is now net of
#     returns, so last-90-day spend can be): baseline RMSE 2824.794 -> 2824.558
#   - the churn model is calibrated only if that helps on the latest cross-validation fold; it
#     did not, so test_ece/brier are those of the uncalibrated model (calibration_applied = 0)
#   - new pins: the bootstrap interval of the gain over recency (seeded, so reproducible), the
#     backtest (4 origins, gain significant at all of them), and the gate decisions.
EXPECTED_CHURN = {
    "algorithm": "hist_gradient_boosting",
    "test_pr_auc": 0.652068,
    "test_roc_auc": 0.772371,
    "test_brier": 0.195024,
    "test_ece": 0.086674,
    "baseline_recency_test_pr_auc": 0.534534,
    "test_band_le90_pr_auc": 0.554387,
    "baseline_recency_test_band_le90_pr_auc": 0.390919,
    "calibration_applied": 0.0,
    "test_pr_auc_vs_recency_gain_ci_low": 0.088806,
    "test_pr_auc_vs_recency_gain_ci_high": 0.144048,
    "backtest_origins": 4.0,
    "backtest_share_significant": 1.0,
}
# Re-pinned when the value model became two-part (P(buy) from the churn model x gamma regressor on
# buyers, review F14). It did not fix the bias: -31.8% (single stage: -30.9%) against -27.0% for
# the naive forecast, and the ranking is unchanged (Spearman 0.5923 vs 0.5922). The single-stage
# figures stay pinned so the comparison is visible. Buyers spent 1,423 on average in the test
# window but the model predicts 1,068, and it expects 53% to buy against 61% who did.
EXPECTED_VALUE = {
    "algorithm": "hurdle_hist_gradient_boosting_gamma",
    "test_rmse": 3112.674358,
    "test_top_decile_capture": 0.558426,
    "test_spearman": 0.592275,
    "test_bias_pct": -31.816829,
    "baseline_naive_test_rmse": 2824.558442,
    "baseline_naive_test_bias_pct": -27.033315,
    "single_stage_test_rmse": 3036.189355,
    "single_stage_test_bias_pct": -30.925470,
    "test_buy_probability_mean": 0.525939,
    "test_buy_rate": 0.612613,
    "test_spend_if_buying_predicted_mean": 1067.740086,
    "test_spend_if_buying_actual_mean": 1423.312341,
}
# What the promotion gate decides on this data: both pass. The value model's bias (-31.8%) is
# within the owner's +-40% bound (it had been rejected when the rule was "no more biased than the
# naive forecast", which is -27.0%). Re-pinned for that rule change; the model itself is unchanged.
EXPECTED_GATE = {"churn": (True, "passed:"), "value": (True, "passed:")}
TOLERANCE = 1e-5  # relative


def _unavailable(request, reason):
    """Skip, or fail when the run was told that the real data must be present."""
    if request.config.getoption("--require-warehouse"):
        pytest.fail(f"--require-warehouse was given, but {reason}")
    pytest.skip(reason)


def _copy_table(source, target, table):
    """Stream a table between two databases with COPY (binary), without building Python rows."""
    with (
        source.cursor() as read,
        target.cursor() as write,
        read.copy(f"COPY {table} TO STDOUT (FORMAT BINARY)") as out,
        write.copy(f"COPY {table} FROM STDIN (FORMAT BINARY)") as into,
    ):
        for block in out:
            into.write(block)


@pytest.fixture(scope="module")
def warehouse(request):
    """A scratch database holding the real raw data and the output of the CURRENT transform SQL."""
    try:
        real = database.get_connection(retries=1)
    except PipelineError as exc:
        _unavailable(request, f"the warehouse is not reachable: {exc}")
    try:
        loaded = database.fetch_scalar(real, f"SELECT count(*) FROM {config.RAW_TABLE}")
        real.rollback()
    except psycopg.Error:
        real.close()
        _unavailable(request, "the warehouse has not been loaded")
    if loaded != RAW_ROWS:
        real.close()
        _unavailable(request, f"the warehouse does not hold the full dataset ({loaded:,} raw rows)")

    admin = database.get_connection(dbname="postgres", retries=1, autocommit=True)
    name = f"test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    scratch = database.get_connection(dbname=name, retries=1)
    try:
        database.run_sql_dir(scratch, config.SCHEMA_DIR)
        _copy_table(real, scratch, config.RAW_TABLE)
        scratch.commit()
        real.close()
        processor.rebuild(scratch)  # the current SQL plus the quality checks
        yield scratch
    finally:
        scratch.close()
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.close()


def test_the_scratch_copy_holds_all_the_raw_rows(warehouse):
    assert database.fetch_scalar(warehouse, f"SELECT count(*) FROM {config.RAW_TABLE}") == RAW_ROWS


def test_row_counts_reconcile_through_the_layers(warehouse):
    staging = database.fetch_scalar(warehouse, "SELECT count(*) FROM staging.transactions")

    assert staging == STAGING_ROWS
    assert RAW_ROWS - staging == OVERLAP_ROWS_REMOVED


def test_line_types_are_unchanged(warehouse):
    rows = warehouse.execute(
        "SELECT line_type, count(*) FROM staging.transactions GROUP BY 1"
    ).fetchall()
    warehouse.rollback()

    assert dict(rows) == LINE_TYPE_ROWS


def test_snapshot_sizes_and_churn_rates_are_unchanged(warehouse):
    rows = warehouse.execute(
        """
        SELECT snapshot_date::text, split, count(*), round(avg(label_churned::int), 4)::float8
        FROM marts.customer_snapshots GROUP BY snapshot_date, split ORDER BY snapshot_date
        """
    ).fetchall()
    warehouse.rollback()

    assert {date: (split, n, churn) for date, split, n, churn in rows} == SNAPSHOTS


def test_actionable_population_sizes_are_unchanged(warehouse):
    rows = warehouse.execute(
        """
        SELECT snapshot_date::text, count(*) FILTER (WHERE is_actionable)
        FROM marts.customer_snapshots GROUP BY snapshot_date ORDER BY snapshot_date
        """
    ).fetchall()
    warehouse.rollback()

    assert dict(rows) == ACTIONABLE


def test_the_data_quality_checks_pass_on_the_real_data(warehouse):
    assert processor.run_quality_checks(warehouse) == []


def test_model_training_reproduces_the_recorded_results(warehouse, tmp_path):
    """Retrains the churn and value models on the rebuilt tables (about 90 seconds) into a
    scratch MLflow store, so nothing in the real registry changes."""
    client = tracking.configure(
        f"sqlite:///{tmp_path / 'mlflow.db'}", (tmp_path / "artifacts").as_uri()
    )
    snapshots = features.load_snapshots(warehouse)
    labelled = snapshots[snapshots["split"].isin(["train", "test"])]

    churn = model.train_churn(labelled, client)
    results = {
        "churn": (churn, EXPECTED_CHURN),
        "value": (model.train_value(labelled, client, churn["estimator"]), EXPECTED_VALUE),
    }

    for name, (trained, expected) in results.items():
        promoted, reason_start = EXPECTED_GATE[name]
        assert trained["promoted"] is promoted, f"{name}: the gate decided differently"
        assert trained["reason"].startswith(reason_start), f"{name}: {trained['reason']}"
        assert trained["algorithm"] == expected["algorithm"], f"{name}: a different model won"
        for metric, value in expected.items():
            if metric != "algorithm":
                assert trained["metrics"][metric] == pytest.approx(value, rel=TOLERANCE), (
                    f"{name}.{metric} changed"
                )

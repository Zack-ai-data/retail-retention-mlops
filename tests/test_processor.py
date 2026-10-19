import math
from datetime import date, timedelta
from decimal import Decimal

import pytest

from src import features, processor
from src.errors import DataQualityError
from tests.conftest import SHEET_1, SHEET_2

UK = "United Kingdom"


# ---------------------------------------------------------------- staging


def staging_rows(conn, where="true"):
    return conn.execute(
        f"""
        SELECT invoice, stock_code, line_type, line_total, is_repeated_line, source_sheet
        FROM staging.transactions WHERE {where} ORDER BY source_sheet, source_row
        """
    ).fetchall()


def test_sheet_overlap_keeps_each_invoice_once_from_the_first_sheet(scratch_conn, insert_raw):
    copy = [("500001", "85123A", "MUG", 2, "2010-12-02 10:00", 3, 17850, UK)]
    insert_raw([(SHEET_1, *row) for row in copy] + [(SHEET_2, *row) for row in copy])
    insert_raw([(SHEET_2, "500002", "85123A", "MUG", 1, "2010-12-20 10:00", 3, 17850, UK)])

    processor.build_staging(scratch_conn)

    rows = staging_rows(scratch_conn)
    assert [(r[0], r[5]) for r in rows] == [("500001", SHEET_1), ("500002", SHEET_2)]


def test_identical_lines_inside_one_sheet_are_kept_and_flagged(scratch_conn, insert_raw):
    line = (SHEET_2, "500001", "22698", "TOWER", 1, "2011-06-05 11:37", 2.95, 14000, UK)
    insert_raw([line, line, line])

    processor.build_staging(scratch_conn)

    flags = [r[4] for r in staging_rows(scratch_conn)]
    assert flags == [False, True, True]


def test_text_is_normalised_and_line_total_computed(scratch_conn, insert_raw):
    insert_raw([(SHEET_1, "500001", " 85123a ", "  ", 3, "2010-01-04 09:00", 2.5, 1, UK)])

    processor.build_staging(scratch_conn)

    row = scratch_conn.execute(
        "SELECT stock_code, description, line_total FROM staging.transactions"
    ).fetchone()
    assert row == ("85123A", None, Decimal("7.5"))


@pytest.mark.parametrize(
    "invoice, stock, qty, price, expected",
    [
        ("500001", "85123A", 1, 2.5, "sale"),
        ("C500002", "85123A", -1, 2.5, "return"),
        ("500003", "POST", 1, 18, "non_product"),
        ("500004", "gift_0001_20", 1, 20, "non_product"),
        ("A500005", "B", 1, -100, "non_product"),
        ("A500006", "85123A", 1, -5, "adjustment"),
        ("500007", "85123A", -3, 0, "stock_adjustment"),
        ("500008", "85123A", 2, 0, "stock_adjustment"),
        ("500009", "PADS", 5, 0.001, "sale"),
    ],
)
def test_line_type(scratch_conn, insert_raw, invoice, stock, qty, price, expected):
    insert_raw([(SHEET_1, invoice, stock, "X", qty, "2010-01-04 09:00", price, None, UK)])

    processor.build_staging(scratch_conn)

    assert staging_rows(scratch_conn)[0][2] == expected


# ------------------------------------------------------------------ marts
# Snapshot dates are fixed in the SQL; these tests use 2011-03-13 (horizon 90 days).

SNAP = date(2011, 3, 13)


def snapshot_rows(conn, snapshot=SNAP):
    cur = conn.execute(
        "SELECT * FROM marts.customer_snapshots WHERE snapshot_date = %s ORDER BY customer_id",
        (snapshot,),
    )
    columns = [c.name for c in cur.description]
    return {
        row[columns.index("customer_id")]: dict(zip(columns, row, strict=True))
        for row in cur.fetchall()
    }


@pytest.fixture
def built_marts(scratch_conn, insert_raw):
    s = SHEET_2
    insert_raw(
        [
            # customer 1: two purchases and a return before the snapshot, one purchase after it
            (s, "600001", "A", "ITEM A", 2, "2011-01-10 10:00", 5, 1, UK),
            (s, "600001", "22633", "ITEM B", 1, "2011-01-10 10:00", 3, 1, UK),
            (s, "600002", "A", "ITEM A", 1, "2011-02-20 11:00", 20, 1, UK),
            (s, "C600003", "A", "ITEM A", -1, "2011-02-25 12:00", 20, 1, UK),
            (s, "600004", "A", "ITEM A", 1, "2011-04-01 09:00", 7, 1, UK),
            # customer 2: bought once, never again -> churned
            (s, "600010", "A", "ITEM A", 1, "2011-01-05 09:00", 4, 2, "France"),
            # customer 3: first purchase exactly at the snapshot -> no history yet, excluded
            (s, "600020", "A", "ITEM A", 1, "2011-03-13 00:00", 4, 3, UK),
            # customer 4: purchase one minute before the snapshot -> included, recency 1 day
            (s, "600030", "A", "ITEM A", 1, "2011-03-12 23:59", 4, 4, UK),
            # customer 5: only buys long after the horizon -> excluded at this snapshot
            (s, "600040", "A", "ITEM A", 1, "2011-08-01 09:00", 4, 5, UK),
            # no customer id and a non-product line must never reach the features
            (s, "600050", "A", "ITEM A", 1, "2011-01-05 09:00", 4, None, UK),
            (s, "600060", "POST", "POSTAGE", 1, "2011-02-01 09:00", 18, 2, "France"),
        ]
    )
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    return scratch_conn


def test_only_customers_with_prior_purchases_are_included(built_marts):
    assert set(snapshot_rows(built_marts)) == {1, 2, 4}


def test_features_use_only_data_before_the_snapshot(built_marts):
    c1 = snapshot_rows(built_marts)[1]

    assert c1["frequency"] == 2
    assert c1["gross_sales"] == Decimal("33")  # 2*5 + 1*3 + 20; the 04-01 order is excluded
    assert c1["monetary"] == Decimal("13")  # net of the 20 returned: money is sales minus returns
    assert c1["avg_order_value"] == Decimal("6.5")  # net per order
    assert c1["total_units"] == 4
    assert c1["distinct_products"] == 2
    assert c1["first_purchase_date"] == date(2011, 1, 10)
    assert c1["last_purchase_date"] == date(2011, 2, 20)
    assert c1["recency_days"] == 21
    assert c1["tenure_days"] == 62
    assert c1["avg_days_between_orders"] == Decimal("41")
    assert c1["orders_last_90d"] == 2
    assert c1["orders_prev_90d"] == 0
    assert c1["spend_last_90d"] == Decimal("13")  # the 02-25 return falls inside the 90 days
    assert c1["return_invoices"] == 1
    assert c1["return_value"] == Decimal("20")
    assert float(c1["return_rate"]) == pytest.approx(20 / 33)  # share of GROSS sales returned
    assert c1["is_uk"] is True


def test_purchase_one_minute_before_snapshot_counts_as_history(built_marts):
    assert snapshot_rows(built_marts)[4]["recency_days"] == 1


def test_labels_describe_the_following_90_days(built_marts):
    rows = snapshot_rows(built_marts)

    assert rows[1]["label_churned"] is False
    assert rows[1]["label_future_orders"] == 1
    assert rows[1]["label_future_spend"] == Decimal("7")

    assert rows[2]["label_churned"] is True
    assert rows[2]["label_future_orders"] == 0
    assert rows[2]["label_future_spend"] == 0
    assert rows[2]["avg_days_between_orders"] is None  # single purchase
    assert rows[2]["is_uk"] is False


def test_postage_lines_do_not_count_as_purchases(built_marts):
    assert snapshot_rows(built_marts)[2]["frequency"] == 1  # the POST line is ignored


def test_score_snapshot_describes_customers_as_of_now_without_labels(built_marts):
    rows = snapshot_rows(built_marts, snapshot=date(2011, 12, 10))

    # every customer with at least one purchase (customer 3 and 5 included), none without
    assert set(rows) == {1, 2, 3, 4, 5}
    assert rows[1]["frequency"] == 3  # includes the 2011-04-01 order, which is history by now
    for row in rows.values():
        assert row["label_churned"] is None
        assert row["label_future_orders"] is None
        assert row["label_future_spend"] is None


# ---------------------------------------------------------- quality checks


def test_quality_checks_pass_on_clean_data_and_catch_lost_rows(built_marts):
    assert processor.run_quality_checks(built_marts, require_label_coverage=False) == []

    # simulate a bug that silently drops a real transaction during cleaning
    built_marts.execute("DELETE FROM staging.transactions WHERE invoice = '600010'")
    built_marts.commit()

    failures = processor.run_quality_checks(built_marts, require_label_coverage=False)
    assert len(failures) == 1
    assert "identical copy" in failures[0]


def test_label_coverage_check_flags_snapshots_without_a_full_label_window(built_marts):
    # the synthetic data ends 2011-08-01, so the later snapshots have incomplete label windows
    failures = processor.run_quality_checks(built_marts, require_label_coverage=True)

    assert any("label window" in f for f in failures)


# ------------------------------------------------- rebuild: publish only if checks pass

HEALTHY = [
    (SHEET_1, "700001", "85123A", "MUG", 2, "2011-01-10 10:00", 5, 1, UK),
    (SHEET_1, "700002", "85123A", "MUG", 1, "2011-02-20 10:00", 5, 2, UK),
]
LATER = [(SHEET_2, "700003", "85123A", "MUG", 1, "2011-04-01 10:00", 5, 3, UK)]


def _counts(conn):
    return conn.execute(
        "SELECT (SELECT count(*) FROM staging.transactions), "
        "(SELECT count(*) FROM marts.customer_snapshots)"
    ).fetchone()


def test_rebuild_publishes_both_layers_when_the_checks_pass(scratch_conn, insert_raw):
    insert_raw(HEALTHY)

    processor.rebuild(scratch_conn, require_label_coverage=False)

    assert _counts(scratch_conn)[0] == 2


def test_failed_checks_leave_the_previous_tables_exactly_as_they_were(scratch_conn, insert_raw):
    insert_raw(HEALTHY)
    processor.rebuild(scratch_conn, require_label_coverage=False)
    before = _counts(scratch_conn)
    scratch_conn.rollback()
    insert_raw(LATER)  # a rebuild would now produce a different staging table

    # the synthetic data ends in 2011, so the label-coverage check fails
    with pytest.raises(DataQualityError, match="previous staging and marts tables were left"):
        processor.rebuild(scratch_conn)

    assert _counts(scratch_conn) == before


def test_an_error_while_building_marts_rolls_back_the_staging_rebuild(
    scratch_conn, insert_raw, monkeypatch
):
    insert_raw(HEALTHY)
    processor.rebuild(scratch_conn, require_label_coverage=False)
    before = _counts(scratch_conn)
    scratch_conn.rollback()
    insert_raw(LATER)

    def fail(conn, commit=True):
        raise RuntimeError("marts script failed")

    monkeypatch.setattr(processor, "build_marts", fail)

    with pytest.raises(RuntimeError, match="marts script failed"):
        processor.rebuild(scratch_conn, require_label_coverage=False)

    assert (
        _counts(scratch_conn) == before
    )  # staging was rebuilt inside the transaction, then undone


# ------------------------------------------- money is net of returns (features and labels)

# Snapshot 2011-03-13, label window [2011-03-13, 2011-06-11). Every customer has history first.
NET_ROWS = [
    # 10: buys 100 and cancels all of it the same day (the case that motivated net labels)
    (SHEET_2, "710001", "A", "ITEM A", 1, "2011-01-05 09:00", 10, 10, UK),
    (SHEET_2, "710002", "A", "ITEM A", 1, "2011-03-20 09:00", 100, 10, UK),
    (SHEET_2, "C710003", "A", "ITEM A", -1, "2011-03-20 15:00", 100, 10, UK),
    # 11: buys 100 and returns 40: still retained, with 60 of net spend
    (SHEET_2, "710011", "A", "ITEM A", 1, "2011-01-05 09:00", 10, 11, UK),
    (SHEET_2, "710012", "A", "ITEM A", 1, "2011-03-20 09:00", 100, 11, UK),
    (SHEET_2, "C710013", "A", "ITEM A", -1, "2011-03-25 09:00", 40, 11, UK),
    # 12: buys 50 but returns 80 (goods from an earlier order): net below zero, so no spend
    (SHEET_2, "710021", "A", "ITEM A", 1, "2011-01-05 09:00", 80, 12, UK),
    (SHEET_2, "710022", "A", "ITEM A", 1, "2011-03-20 09:00", 50, 12, UK),
    (SHEET_2, "C710023", "A", "ITEM A", -1, "2011-03-22 09:00", 80, 12, UK),
    # 13: only a return in the window: nothing was bought
    (SHEET_2, "710031", "A", "ITEM A", 1, "2011-01-05 09:00", 30, 13, UK),
    (SHEET_2, "C710032", "A", "ITEM A", -1, "2011-04-01 09:00", 30, 13, UK),
    # 14: buys 100 in the window and returns it only after the window ends: still retained
    (SHEET_2, "710041", "A", "ITEM A", 1, "2011-01-05 09:00", 10, 14, UK),
    (SHEET_2, "710042", "A", "ITEM A", 1, "2011-03-20 09:00", 100, 14, UK),
    (SHEET_2, "C710043", "A", "ITEM A", -1, "2011-06-15 09:00", 100, 14, UK),
    # 15: returns more than they ever bought before the snapshot: negative net money to date
    (SHEET_2, "710051", "A", "ITEM A", 1, "2011-01-05 09:00", 10, 15, UK),
    (SHEET_2, "C710052", "A", "ITEM A", -1, "2011-02-01 09:00", 30, 15, UK),
]


@pytest.fixture
def net_marts(scratch_conn, insert_raw):
    insert_raw(NET_ROWS)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    return scratch_conn


@pytest.mark.parametrize(
    "customer, churned, spend",
    [
        (10, True, Decimal("0")),  # cancelled the same day: not a retained customer
        (11, False, Decimal("60")),  # partial return: still retained
        (12, True, Decimal("0")),  # returns exceed purchases: floored at zero
        (13, True, Decimal("0")),  # a return and nothing bought
        (14, False, Decimal("100")),  # the return happens after the window
    ],
)
def test_labels_count_money_net_of_returns_in_the_window(net_marts, customer, churned, spend):
    row = snapshot_rows(net_marts)[customer]

    assert row["label_churned"] is churned
    assert row["label_future_spend"] == spend
    assert row["label_future_spend"] >= 0


def test_a_cancelled_order_still_counts_as_an_order_placed(net_marts):
    # the order count is informational; spend and churn are what decide retention
    assert snapshot_rows(net_marts)[10]["label_future_orders"] == 1


def test_money_to_date_can_be_negative_when_returns_exceed_purchases(net_marts):
    row = snapshot_rows(net_marts)[15]

    assert row["gross_sales"] == Decimal("10")
    assert row["monetary"] == Decimal("-20")
    assert row["spend_last_90d"] == Decimal("-20")
    assert row["return_value"] == Decimal("30")


def test_no_customer_is_retained_without_positive_spend(net_marts):
    assert processor.run_quality_checks(net_marts, require_label_coverage=False) == []

    # simulate a labelling bug: mark a zero-spend customer as retained
    net_marts.execute(
        "UPDATE marts.customer_snapshots SET label_churned = false "
        "WHERE customer_id = 10 AND snapshot_date = '2011-03-13'"
    )
    net_marts.commit()

    failures = processor.run_quality_checks(net_marts, require_label_coverage=False)

    assert any("retained without positive net spend" in failure for failure in failures)


# ------------------------------------------- 365-day window, history length, time of year

WINDOW_ROWS = [
    # customer 21, snapshot 2011-03-13 (the 365-day window starts 2010-03-13)
    (SHEET_2, "720001", "A", "ITEM A", 1, "2010-02-01 09:00", 10, 21, UK),  # before the window
    (SHEET_2, "C720002", "A", "ITEM A", -1, "2010-03-01 09:00", 5, 21, UK),  # return, before it
    (SHEET_2, "720003", "A", "ITEM A", 1, "2010-06-01 09:00", 20, 21, UK),  # inside
    (SHEET_2, "720004", "22633", "ITEM B", 1, "2011-01-10 09:00", 30, 21, UK),  # inside
    (SHEET_2, "C720005", "22633", "ITEM B", -1, "2011-02-01 09:00", 8, 21, UK),  # return, inside
    # customer 22: one purchase long before the window, then silence
    (SHEET_2, "720011", "A", "ITEM A", 1, "2010-02-15 09:00", 40, 22, UK),
]


@pytest.fixture
def window_marts(scratch_conn, insert_raw):
    insert_raw(WINDOW_ROWS)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    return scratch_conn


def test_window_features_count_only_the_365_days_before_the_snapshot(window_marts):
    c = snapshot_rows(window_marts)[21]

    assert c["frequency"] == 3 and c["frequency_365d"] == 2  # lifetime vs window
    assert c["gross_sales"] == Decimal("60") and c["gross_sales_365d"] == Decimal("50")
    assert c["return_value"] == Decimal("13") and c["return_value_365d"] == Decimal("8")
    assert c["monetary"] == Decimal("47") and c["monetary_365d"] == Decimal("42")  # net
    assert c["avg_order_value_365d"] == Decimal("21")  # 42 net over 2 orders
    assert c["return_invoices_365d"] == 1
    assert c["total_units_365d"] == 2
    assert c["distinct_products_365d"] == 2
    expected_gap = (date(2011, 1, 10) - date(2010, 6, 1)).days  # between the two in-window orders
    assert float(c["avg_days_between_orders_365d"]) == pytest.approx(expected_gap)


def test_a_customer_silent_for_over_a_year_has_empty_window_features(window_marts):
    c = snapshot_rows(window_marts)[22]

    assert c["frequency"] == 1  # still a customer with history
    assert c["frequency_365d"] == 0
    assert c["monetary_365d"] == 0
    assert c["avg_order_value_365d"] is None  # undefined with no orders
    assert c["avg_days_between_orders_365d"] is None


def test_history_available_is_capped_at_a_year(window_marts):
    c = snapshot_rows(window_marts)[21]

    # the data starts 2010-02-01, so 405 days exist before 2011-03-13: capped at 365
    assert c["history_days_available"] == 365
    assert c["tenure_days_capped"] == 365  # first purchase 2010-02-01 is 405 days before


def test_short_history_snapshots_say_how_little_history_exists(scratch_conn, insert_raw):
    insert_raw(
        [
            (SHEET_2, "730001", "A", "ITEM A", 1, "2010-01-10 09:00", 10, 31, UK),
            (SHEET_2, "730002", "A", "ITEM A", 1, "2010-02-20 09:00", 10, 31, UK),
        ]
    )
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)

    c = snapshot_rows(scratch_conn, snapshot=date(2010, 3, 18))[31]

    assert c["history_days_available"] == 67  # 2010-01-10 to 2010-03-18
    assert c["tenure_days_capped"] == 67
    assert c["frequency_365d"] == 2


def test_season_terms_encode_the_time_of_year_of_the_snapshot(window_marts):
    c = snapshot_rows(window_marts)[21]
    day_of_year = date(2011, 3, 13).timetuple().tm_yday

    assert c["season_sin"] == pytest.approx(math.sin(2 * math.pi * day_of_year / 365.25))
    assert c["season_cos"] == pytest.approx(math.cos(2 * math.pi * day_of_year / 365.25))


def test_window_features_never_exceed_the_lifetime_ones(window_marts):
    violations = window_marts.execute(
        """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE frequency_365d > frequency OR gross_sales_365d > gross_sales
           OR return_value_365d > return_value OR history_days_available NOT BETWEEN 0 AND 365
        """
    ).fetchone()[0]

    assert violations == 0


# ------------------------------------------------------------ actionable population


@pytest.fixture
def actionable_marts(scratch_conn, insert_raw):
    snapshot = date(2011, 3, 13)
    limit = features.ACTIONABLE_RECENCY_DAYS
    rows = []
    for customer, days_ago in ((41, limit), (42, limit + 1), (43, 1)):
        when = snapshot - timedelta(days=days_ago)
        rows.append((SHEET_2, f"74{customer}", "A", "ITEM A", 1, f"{when} 10:00", 10, customer, UK))
    insert_raw(rows)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    return scratch_conn


def test_actionable_means_bought_within_the_threshold_and_agrees_with_the_python_constant(
    actionable_marts,
):
    rows = snapshot_rows(actionable_marts)

    assert rows[41]["recency_days"] == features.ACTIONABLE_RECENCY_DAYS
    assert rows[41]["is_actionable"] is True  # exactly at the threshold
    assert rows[42]["recency_days"] == features.ACTIONABLE_RECENCY_DAYS + 1
    assert rows[42]["is_actionable"] is False  # one day past it
    assert rows[43]["is_actionable"] is True


def test_the_python_horizon_matches_the_one_in_the_warehouse(built_marts):
    horizons = built_marts.execute(
        "SELECT DISTINCT horizon_days FROM marts.snapshot_dates"
    ).fetchall()

    assert horizons == [(features.HORIZON_DAYS,)]

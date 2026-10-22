"""The demo traffic script: labelled as synthetic, safe for the API, and well behaved."""

import logging
import random
import re
from types import SimpleNamespace

import pytest
import requests

from src import demo_traffic, metrics

API = "http://api:8000"
CUSTOMERS = [11, 22, 33, 44, 55]
KNOWN_PATHS = re.compile(
    r"^(/customers/(11|22|33|44|55)/(score|prediction|features)"
    r"|/health|/outreach/latest"
    r"|/customers/\d{8}/score"  # a customer that does not exist
    r"|/customers/not-a-number/score|/demo/no/such/route)$"
)


class FakeClock:
    """Time that only moves when the script sleeps."""

    def __init__(self):
        self.now = 0.0
        self.reads = 0

    def __call__(self):
        self.reads += 1
        if self.reads > 100_000:  # a script that never sleeps would loop for ever
            raise AssertionError("the script is not letting time pass (it never sleeps)")
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSession:
    """Answers like the API: client errors for the requests that are meant to fail."""

    def __init__(self, down=False, reply=None):
        self.calls, self.down, self.reply = [], down, reply

    def request(self, method, url, params=None, headers=None, timeout=None):
        self.calls.append((method, url, params, headers))
        if self.down:
            raise requests.ConnectionError("refused")
        path = url.removeprefix(API)
        if "not-a-number" in path:
            return SimpleNamespace(status_code=422)
        if "no/such/route" in path or re.search(r"/customers/\d{8}/", path):
            return SimpleNamespace(status_code=404)
        return SimpleNamespace(status_code=200)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, params, headers))
        if self.down:
            raise requests.ConnectionError("refused")
        return self.reply


def _session_replying(status, body):
    return FakeSession(reply=SimpleNamespace(status_code=status, json=lambda: body))


# ------------------------------------------------------------------------ labelled as demo


def test_the_script_says_it_is_synthetic_and_names_the_header_and_the_label():
    assert "not real usage" in demo_traffic.BANNER
    assert metrics.DEMO_TRAFFIC_HEADER in demo_traffic.BANNER
    assert 'traffic="demo"' in demo_traffic.BANNER


def test_the_header_is_the_one_the_api_counts_as_demo():
    assert demo_traffic.DEMO_TRAFFIC_HEADER is metrics.DEMO_TRAFFIC_HEADER


def test_every_request_it_sends_carries_the_demo_header():
    session, clock = FakeSession(), FakeClock()

    demo_traffic.run(session, API, CUSTOMERS, 20, 4, random.Random(1), clock, clock.sleep)

    assert session.calls
    assert all(headers == {metrics.DEMO_TRAFFIC_HEADER: "true"} for *_, headers in session.calls)


def test_the_script_announces_itself_when_it_starts(caplog, monkeypatch):
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(down=True))

    with caplog.at_level(logging.INFO, logger="demo_traffic"):
        demo_traffic.main(["--api-url", API])

    assert "DEMO TRAFFIC" in caplog.text and "not real usage" in caplog.text


# ---------------------------------------------------------------------- what it asks for


def test_it_only_asks_for_things_that_cannot_cause_a_server_error():
    rng = random.Random(0)
    calls = [demo_traffic.pick_call(rng, CUSTOMERS) for _ in range(3000)]

    assert all(KNOWN_PATHS.match(call.path) for call in calls), {c.path for c in calls}
    assert {call.method for call in calls} == {"GET"}


def test_the_mix_adds_up_to_a_hundred_and_the_biggest_share_is_what_is_sent_most():
    assert sum(share for _, share in demo_traffic.MIX) == 100
    rng = random.Random(0)
    n = 20000
    paths = [demo_traffic.pick_call(rng, CUSTOMERS).path for _ in range(n)]

    scores = sum(bool(re.fullmatch(r"/customers/(11|22|33|44|55)/score", p)) for p in paths)
    malformed = sum("not-a-number" in p for p in paths)
    assert 0.30 < scores / n < 0.40  # the "score" share is 35%
    assert 0.01 < malformed / n < 0.06  # the "malformed id" share is 3%


def test_the_same_seed_gives_the_same_traffic():
    first, second = random.Random(9), random.Random(9)

    assert [demo_traffic.pick_call(first, CUSTOMERS) for _ in range(50)] == [
        demo_traffic.pick_call(second, CUSTOMERS) for _ in range(50)
    ]


def test_it_only_asks_about_customers_that_exist_or_clearly_unknown_ones():
    rng = random.Random(3)
    ids = {
        int(m.group(1))
        for _ in range(3000)
        if (m := re.search(r"/customers/(\d+)/", demo_traffic.pick_call(rng, CUSTOMERS).path))
    }

    unknown = set(range(demo_traffic.NO_SUCH_CUSTOMER, demo_traffic.NO_SUCH_CUSTOMER + 1000))
    assert ids <= set(CUSTOMERS) | unknown


# -------------------------------------------------------------------------------- the wave


def test_the_rate_rises_and_falls_around_the_average_and_is_never_negative():
    rng = random.Random(0)
    per_second = [demo_traffic.requests_this_second(rng, 5, t) for t in range(120)]

    assert min(per_second) >= 0 and min(per_second) < 5 < max(per_second)
    assert abs(sum(per_second) / len(per_second) - 5) < 0.6  # about the average over whole waves


def test_a_low_rate_sometimes_sends_nothing_rather_than_going_negative():
    rng = random.Random(0)

    assert {demo_traffic.requests_this_second(rng, 0.3, t) for t in range(120)} >= {0, 1}


# ---------------------------------------------------------------------------------- running


def test_it_runs_for_the_time_asked_and_tallies_the_answers():
    session, clock = FakeSession(), FakeClock()

    outcomes = demo_traffic.run(
        session, API, CUSTOMERS, 60, 4, random.Random(2), clock, clock.sleep
    )

    assert 55 <= clock.now <= 70  # about a minute of (simulated) time
    assert 150 <= sum(outcomes.values()) <= 330  # about 4 a second on average
    assert set(outcomes) <= {"200", "404", "422"}  # never a server error
    assert outcomes["200"] > outcomes["404"] > 0 and outcomes["422"] > 0


def test_when_the_api_stops_answering_it_gives_up_with_a_clear_message():
    session, clock = FakeSession(down=True), FakeClock()

    with pytest.raises(demo_traffic.DemoTrafficError, match="stopped answering"):
        demo_traffic.run(session, API, CUSTOMERS, 600, 5, random.Random(0), clock, clock.sleep)

    assert len(session.calls) == demo_traffic.MAX_FAILURES_IN_A_ROW
    # the failing requests were paced like any others (about 5 a second), not fired in a burst
    assert clock.now >= demo_traffic.MAX_FAILURES_IN_A_ROW / 8


def test_a_brief_failure_is_not_fatal_and_is_counted():
    session, clock = FakeSession(), FakeClock()
    real_request = session.request
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if 20 <= state["n"] < 25:  # five in a row, well under the limit
            raise requests.Timeout("slow")
        return real_request(*args, **kwargs)

    session.request = flaky

    outcomes = demo_traffic.run(
        session, API, CUSTOMERS, 30, 5, random.Random(4), clock, clock.sleep
    )

    assert outcomes["no answer"] == 5 and outcomes["200"] > 0


# ------------------------------------------------------------------------ finding customers


def test_the_customers_come_from_the_latest_outreach_plan():
    session = _session_replying(200, {"customers": [{"customer_id": 7}, {"customer_id": 9}]})

    assert demo_traffic.fetch_customers(session, API) == [7, 9]
    assert session.calls[0][3] == {metrics.DEMO_TRAFFIC_HEADER: "true"}


def test_no_plan_yet_says_what_to_run_first():
    with pytest.raises(demo_traffic.DemoTrafficError, match="--stage plan"):
        demo_traffic.fetch_customers(_session_replying(404, {}), API)


def test_a_plan_with_no_customers_is_an_error_not_an_endless_loop_of_nothing():
    with pytest.raises(demo_traffic.DemoTrafficError, match="no customers"):
        demo_traffic.fetch_customers(_session_replying(200, {"customers": []}), API)


def test_an_unreachable_api_says_how_to_start_it():
    with pytest.raises(demo_traffic.DemoTrafficError, match="docker compose up -d api"):
        demo_traffic.fetch_customers(FakeSession(down=True), API)


# ------------------------------------------------------------------------------ the command


def test_main_reports_an_unreachable_api_and_exits_with_a_failure(monkeypatch, caplog):
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(down=True))

    with caplog.at_level(logging.ERROR, logger="demo_traffic"):
        code = demo_traffic.main(["--api-url", API, "--minutes", "1"])

    assert code == 1 and "Cannot reach the API" in caplog.text


@pytest.mark.parametrize("bad", [["--minutes", "0"], ["--rate", "-1"]])
def test_a_duration_or_rate_of_zero_or_less_is_refused(bad):
    with pytest.raises(SystemExit):
        demo_traffic.main(bad)


def test_main_runs_end_to_end_and_ends_by_saying_it_was_synthetic(monkeypatch, caplog):
    body = {"customers": [{"customer_id": c} for c in CUSTOMERS]}
    session = _session_replying(200, body)
    clock = FakeClock()
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(demo_traffic.time, "monotonic", clock)
    monkeypatch.setattr(demo_traffic.time, "sleep", clock.sleep)

    with caplog.at_level(logging.INFO, logger="demo_traffic"):
        code = demo_traffic.main(
            ["--api-url", API + "/", "--minutes", "0.5", "--rate", "3", "--seed", "1"]
        )

    assert code == 0
    assert "Done:" in caplog.text and "synthetic traffic, not real usage" in caplog.text
    assert session.calls[0][1] == f"{API}/outreach/latest"  # the trailing slash was dropped


def test_ctrl_c_stops_the_demo_cleanly_and_is_not_an_error(monkeypatch, caplog):
    body = {"customers": [{"customer_id": c} for c in CUSTOMERS]}
    monkeypatch.setattr(requests, "Session", lambda: _session_replying(200, body))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(demo_traffic, "run", interrupted)

    with caplog.at_level(logging.INFO, logger="demo_traffic"):
        code = demo_traffic.main(["--api-url", API])

    assert code == 0 and "Stopped." in caplog.text

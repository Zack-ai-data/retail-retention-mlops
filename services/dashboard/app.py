"""Retail retention dashboard (Streamlit).

Everything on screen comes from the API (src/api.py); this file only lays it out. The logic
behind each page lives in src/dashboard_client.py, where it is tested.
"""

from typing import Any

import pandas as pd
import streamlit as st

from src import dashboard_client as dc

st.set_page_config(page_title="Retail retention", page_icon="🛍️", layout="wide")

client = dc.ApiClient()

# The inputs worth changing in a what-if. The rest stay as the customer's own values.
WHAT_IF_INPUTS = [
    "recency_days",
    "frequency_365d",
    "orders_last_90d",
    "orders_prev_90d",
    "monetary_365d",
    "spend_last_90d",
    "return_value_365d",
]


def call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, dc.ApiError | None]:
    """Run an API call; return its result or the explained error, never an exception."""
    try:
        return function(*args, **kwargs), None
    except dc.ApiError as exc:
        return None, exc


def score_metrics(score: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Chance of churning", dc.percent(score["p_churn"]))
    columns[1].metric("Spend if they buy (90 days)", dc.money(score["spend_if_retained"]))
    columns[2].metric("Margin at risk", dc.money(score["margin_at_risk"]))
    columns[3].metric("Worth contacting", "Yes" if score["worth_contacting"] else "No")


# ------------------------------------------------------------------ sidebar: the API

st.title("Retail retention")
health, problem = call(client.health)
with st.sidebar:
    st.header("Service")
    if problem is not None:
        st.error(str(problem))
    elif health["status"] == "ok":
        st.success(
            f"API ok. Churn model v{health['churn_model_version']}, "
            f"value model v{health['value_model_version']}."
        )
        run = health.get("latest_scoring_run")
        if run:
            st.caption(
                f"Latest scoring: {run['scored_at'][:16]}, customers as of {run['snapshot_date']}"
            )
        else:
            st.caption("No scoring run yet: python -m src.main --stage score")
    else:
        st.warning(f"The API is degraded: {health.get('problem', 'unknown problem')}")
    if st.button("Reload models", key="reload"):
        _, reload_problem = call(client.reload_models)
        if reload_problem is not None:
            st.error(str(reload_problem))
        else:
            st.rerun()
    st.divider()
    st.caption(
        "The scores rank customers by risk. The project has no campaign data, so it does not "
        "claim that contacting someone saves them."
    )

if problem is not None:
    st.error(f"Nothing can be shown until the API is reachable. {problem}")
    st.stop()

plan_tab, customer_tab, what_if_tab, assumptions_tab = st.tabs(
    ["Who to contact", "Customer", "What-if", "Assumptions"]
)

# ------------------------------------------------------------- who to contact

with plan_tab:
    plan, plan_problem = call(client.outreach_plan)
    if plan_problem is not None:
        st.info(str(plan_problem))
    else:
        totals = dc.plan_totals(plan)
        columns = st.columns(4)
        columns[0].metric("Customers to contact", f"{totals['contact']:,}")
        columns[1].metric("Held out (do not contact)", f"{totals['holdout']:,}")
        columns[2].metric("Contact cost", dc.money(totals["cost"]))
        columns[3].metric("Margin at risk covered", dc.money(totals["margin_at_risk_contacted"]))
        st.caption(
            f"Plan {plan['plan_id']} from scoring run {plan['scoring_run_id']}: top {plan['k']} "
            f"by margin at risk, assuming {dc.percent(plan['margin'])} margin and "
            f"{dc.money(plan['contact_cost'])} per contact. Holdout "
            f"{dc.percent(plan['holdout_fraction'])}, random seed {plan['seed']}."
        )
        if not totals["complete"]:
            st.warning("Only part of the plan is shown, so the margin figure is understated.")
        st.info(
            "The held-out customers are a random sample of the selected ones, deliberately not "
            "contacted. Comparing them with the contacted group later is the only way to find out "
            "whether contact changes anything. Do not contact them."
        )
        choice = st.selectbox(
            "Show", ["contact", "holdout", "not_selected", "all"], key="plan_group"
        )
        table = pd.DataFrame(plan["customers"])
        if choice != "all":
            table = table[table["group"] == choice]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download the contact list (CSV)",
            dc.contact_list_csv(plan),
            file_name="contact_list.csv",
            mime="text/csv",
            key="download_contact",
        )

# ------------------------------------------------------------------- customer

with customer_tab:
    st.write(
        "Look up a customer. The live models score them now, and the answer is compared with what "
        "batch scoring stored. They should agree for the same model versions."
    )
    default_id = (
        plan["customers"][0]["customer_id"] if plan_problem is None and plan["customers"] else 12347
    )
    customer_id = st.number_input(
        "Customer id", min_value=1, step=1, value=default_id, key="customer_id"
    )
    if st.button("Look up", key="lookup"):
        st.session_state["looked_up"] = int(customer_id)
    if "looked_up" in st.session_state:
        wanted = st.session_state["looked_up"]
        online, online_problem = call(client.score, wanted)
        if online_problem is not None:
            st.warning(str(online_problem))
        else:
            st.subheader(f"Customer {wanted}")
            if online.get("segment"):
                st.caption(f"Segment: {online['segment']}")
            score_metrics(online)
            stored, stored_problem = call(client.prediction, wanted)
            if stored_problem is not None:
                st.info(f"No stored score to compare with. {stored_problem}")
            else:
                comparison = dc.compare_scores(online, stored)
                if comparison["agree"].all():
                    st.success("The live score and the stored score agree.")
                else:
                    st.warning(
                        "The live and stored scores differ. That is expected if the models were "
                        "retrained since the last scoring run; the versions below say."
                    )
                # numbers and model versions share these columns, which a table cannot hold as one
                # type; show them all as text
                st.dataframe(
                    comparison.astype({"online": "string", "stored": "string"}), hide_index=True
                )

# -------------------------------------------------------------------- what-if

with what_if_tab:
    st.write(
        "Start from a customer's own inputs, change some, and see how the score moves. The model "
        "sees exactly what you enter: changing one input does not update the ones related to it."
    )
    what_if_id = st.number_input(
        "Customer id", min_value=1, step=1, value=default_id, key="what_if_id"
    )
    if st.button("Load their inputs", key="load_inputs"):
        loaded, load_problem = call(client.customer_features, int(what_if_id))
        if load_problem is not None:
            st.session_state.pop("what_if_inputs", None)
            st.warning(str(load_problem))
        else:
            st.session_state["what_if_inputs"] = loaded
    loaded = st.session_state.get("what_if_inputs")
    if loaded:
        original = loaded["features"]
        st.subheader(f"Customer {loaded['customer_id']}")
        edited = dict(original)
        columns = st.columns(3)
        for position, name in enumerate(WHAT_IF_INPUTS):
            edited[name] = columns[position % 3].number_input(
                name, value=float(original[name]), key=f"what_if_{name}"
            )
        if st.button("Score the changed inputs", key="score_what_if"):
            before, before_problem = call(client.what_if, original)
            after, after_problem = call(client.what_if, edited)
            if before_problem is not None or after_problem is not None:
                st.warning(str(before_problem or after_problem))
            else:
                columns = st.columns(3)
                columns[0].metric(
                    "Chance of churning",
                    dc.percent(after["p_churn"]),
                    f"{(after['p_churn'] - before['p_churn']) * 100:+.1f} points",
                    delta_color="inverse",
                )
                columns[1].metric(
                    "Spend if they buy",
                    dc.money(after["spend_if_retained"]),
                    f"{after['spend_if_retained'] - before['spend_if_retained']:+,.0f}",
                )
                columns[2].metric(
                    "Margin at risk",
                    dc.money(after["margin_at_risk"]),
                    f"{after['margin_at_risk'] - before['margin_at_risk']:+,.0f}",
                    delta_color="inverse",
                )

# ---------------------------------------------------------------- assumptions

with assumptions_tab:
    st.write(
        "There is no data on the margin or the cost of a contact; both are assumptions. This shows "
        "how the list of customers to contact changes if they were different."
    )
    columns = st.columns(3)
    size = columns[0].number_input(
        "Customers selected", min_value=1, max_value=5000, value=500, step=50, key="size"
    )
    margins = columns[1].multiselect(
        "Margin", [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], default=[0.2, 0.3, 0.4], key="margins"
    )
    costs = columns[2].multiselect(
        "Cost per contact (£)",
        [1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0],
        default=[3.0, 5.0, 8.0],
        key="costs",
    )
    if margins and costs:
        rows, sensitivity_problem = call(client.sensitivity, int(size), margins, costs)
        if sensitivity_problem is not None:
            st.info(str(sensitivity_problem))
        else:
            tables = dc.sensitivity_tables(rows)
            if dc.assumptions_make_no_difference(rows):
                st.success(
                    "Under every combination shown, the same customers are selected. The fixed "
                    "number of customers decides the list; the margin and the contact cost only "
                    "matter once they are high enough to rule customers out."
                )
            else:
                st.info(
                    "Some combinations change who is selected. Margin scales everyone's margin at "
                    "risk equally, so it never reorders customers; it only moves the cutoff where "
                    "a customer stops being worth the contact cost."
                )
            st.write("Customers selected")
            st.dataframe(tables["selected"])
            st.write("Share of the baseline list still selected")
            st.dataframe(tables["overlap"].map(dc.percent))
    else:
        st.info("Choose at least one margin and one cost.")

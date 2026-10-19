# Problem statement

This is the business framing the project's models are built and judged against. The data
(UCI Online Retail II) is public and has no business owner, so these are assumptions I have
written down and checked against the data, not facts about a real company.

## Decision supported

A quarterly retention outreach to at-risk **active** customers, within a fixed contact budget
(a top-k list). Secondary: deciding which large wholesale accounts get a person's attention.
Stock planning is a different problem and out of scope.

## Population

Customers who bought in the **180 days before the snapshot date** (53% of customers at the
last test snapshot). Customers who have been silent for longer are lapsed: their churn is
77-90% in every snapshot and the model does no better than ranking by recency among them.
They belong on a separate win-back list. Wholesalers stay in scope, with a flag.

## What counts as lost

A customer is lost if they have **no net-positive spend (sales minus returns) in the next
90 days**. 90 days is about the 80th percentile of the gaps between repeat orders (median
32 days; 80.6% of gaps are 90 days or less). The other 19.4% of gaps are longer, so some
customers labelled lost do return. Sensitivity of the results to 60, 120 and 180 days is
tested during model evaluation.

## Returns

Returns are real losses. They are netted out of features and labels inside each window, and an
order cancelled the same day is not a purchase.

## Cost and value (assumptions; there is no data on either)

- Contact cost: **£5** per customer contacted.
- Margin: **30%** of spend.
- Customers are ranked by **margin at risk**: `P(churn) x spend if the customer buys x 30%`.
  "Spend if the customer buys" is the second stage of the value model, not the 90-day spend
  forecast: that forecast already includes the chance of not buying, which `P(churn)` accounts for,
  so using it would count that chance twice. (This refines the first draft of this document, which
  said "predicted 90-day spend".) Margin at risk is an upper bound on what contact could save.
- A customer is only worth contacting if their margin at risk is more than the £5 cost.
- **Outreach size:** the plan selects the top 500 customers per quarter (about 15% of those
  scored). The budget is not in the data, so this is a stated assumption and a parameter.
- **Holdout:** 10% of the selected customers, chosen at random with a fixed seed, are deliberately
  not contacted. The plan is stored, so the group is fixed and can be audited.
- The write-up shows how the top-k list changes as cost and margin vary.

## Success criteria

- **Churn model:** recall and precision in the top-k on the active population, against the
  recency baseline, plus calibration (the ranking multiplies the probability). "Good enough"
  means the paired-bootstrap interval of the gain over the baseline excludes 0 on most
  backtest origins.
- **Value model:** top-decile spend capture, plus a bound on total bias (see below), with RMSE kept as a
  guard against getting worse than the naive forecast.
- **Bias bound:** the value model's total predicted spend must be within **40%** of actual spend,
  in either direction. This is a sanity limit, not an accuracy target. It is wide enough for the
  seasonal level shift of predicting a holiday-peak quarter from one year of data (-32% at the
  last test snapshot; the naive last-quarter forecast is -27%), and narrow enough to catch a
  broken model (an early log-target version was -78%). The owner chose an absolute bound and confirmed
  the figure of 40% on 2026-10-19; it was proposed **after seeing those results**, so it is an
  assumption to revisit, not a requirement derived independently. A
  uniform bias does not change how customers rank by value; it moves which customers cross the
  contact-cost threshold, so it should be watched and shrunk as more seasons of data arrive.
- Accuracy is not used: always guessing "churn" is already right 57% of the time at the test
  snapshot.

## What this project does not claim

There is no campaign history in the data, so the project ranks **who is at risk**, never
**who can be saved**. It does not estimate uplift. The outreach optimizer includes a random
holdout group so that impact could be measured later.

## Evidence

| Evidence | Source |
|---|---|
| Gaps between repeat orders: 27,026 gaps, median 32 days, 80th percentile 88, 90th percentile 148, 80.6% within 90 days, 92.7% within 180 | read-only query on `staging.transactions` (distinct purchase days per identified customer) |
| Actionable share of customers per snapshot | `marts.customer_snapshots.is_actionable` (regression test `test_actionable_population_sizes_are_unchanged`) |

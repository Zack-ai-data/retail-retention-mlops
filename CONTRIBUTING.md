# Machine Learning Engineering workflow

How changes are made here, so that anyone (including the person who wrote the code) can say what a change does and does not guarantee.

## Definition of done

A change (new feature, bug fix, refactor, dependency bump) is done only when all of these hold:

1. **Tests exist for the new or changed behaviour** (see "Where tests go"). A bug fix starts with a test that fails for the bug.
2. **`docker compose run --rm quality` exits 0.** It runs, in order: `ruff check`, `ruff format --check`, `mypy --strict` (on `src/`), and `pytest --cov` with a 90% branch-coverage minimum.
3. **The real-data regression tests pass.** They copy the loaded raw table into a scratch database and run your *current* SQL and model code on it, so they need the real data loaded (`docker compose run --rm app` once). Run them with `--require-warehouse`, which turns a skip into a failure, so a green run really did check the data. Run the full pipeline too if data processing or models were touched.
4. **Intentional changes to results update the pinned values in the same change.** If a change deliberately alters the cleaning SQL, features, labels or models, update the expected numbers in `tests/test_warehouse_regression.py` and state why in the commit message. An unexplained change in those numbers is a regression until shown otherwise.
5. **Docs match the code**: README (commands, structure) and the review/decision notes where relevant.

## What "no regression" means, precisely

The gate being green proves:

- the code is lint-clean, consistently formatted and type-consistent;
- every test passes, and at least 90% of branches are exercised;
- the current cleaning and feature SQL, run on a copy of the real raw data, still produces the same row counts, line types, snapshot sizes and churn rates, and re-training on those tables still reproduces the recorded model results. (When the real data is not loaded these tests are skipped, so a green run without `--require-warehouse` does not prove this part.)

It does **not** prove:

- that behaviour no test covers is correct (coverage measures lines executed, not behaviour verified);
- that the business assumptions are right (for example the 90-day churn horizon);
- that the pinned numbers are themselves correct. They detect change; they are not an independent oracle.

So the accurate statement after a green gate is: "nothing the tests pin has changed". Say that, not "nothing broke".

## Coverage is a floor, not a goal

The gate requires 90% branch coverage so that large untested areas cannot creep in. It is not a target to maximise. Do not add tests whose only purpose is to raise the number: contrived tests for trivial or defensive code, or tests that run code without asserting on the result, add cost and false confidence without catching bugs.

- Coverage counts lines executed, not behaviour verified. A line can be covered by a test that checks nothing about it.
- Put the effort where a mistake would matter (data correctness, the promotion decision, metrics, error handling) and write precise assertions there.
- A few uncovered defensive branches are acceptable. A falling number is a prompt to ask whether new behaviour is tested, not to chase a percentage.
- The stronger measure of test quality is whether tests fail when the code is wrong. Mutation testing does this systematically. It was run once on 18 pure functions (91% of 605 mutants killed, using the faster tests only); the training, orchestration, database and SQL code were not mutated, so there is no mutation evidence for them either way. It is not part of the gate. The slow model-training tests make the tool hang, so leave them out.

## Where tests go

| What changed | Test file | Notes |
|---|---|---|
| Pure logic (metrics, feature building, naming) | `tests/test_metrics.py`, `tests/test_model.py`, `tests/test_extractor.py` | no database needed |
| The promotion gate, bootstrap, calibration, baselines, hurdle model | `tests/test_gate.py`, `tests/test_evaluation.py`, `tests/test_calibration.py`, `tests/test_baselines.py`, `tests/test_hurdle.py`, `tests/test_tracking.py` | no database needed; `test_tracking.py` uses a throwaway SQLite MLflow store |
| SQL (cleaning, features, labels) | `tests/test_processor.py` | runs the real SQL on a throwaway Postgres database with hand-built rows |
| Database helpers | `tests/test_database.py` | |
| A stage, the command line, exit codes | `tests/test_pipeline.py` | `-m e2e` runs the whole pipeline on a synthetic workbook |
| The scoring arithmetic, the optimizer, the API | `tests/test_scoring.py`, `tests/test_optimizer.py`, `tests/test_api.py` | the API tests use FastAPI's test client, a throwaway database and stand-in models |
| The dashboard | `tests/test_dashboard_client.py` (no Streamlit), `tests/test_dashboard_ui.py` (needs Streamlit) | the UI tests are skipped in the `quality` run; run them with `dashboard-tests` |
| A new way for things to fail | `tests/test_error_handling.py` | add an exception type in `src/errors.py` and a test for its message |
| Results on the real data | `tests/test_warehouse_regression.py` | `-m warehouse`; copies the loaded raw table into a scratch database and runs the current SQL and model code there; skipped when the real data is not loaded unless `--require-warehouse` is given |

Keep tests deterministic: seed randomness, no network, no reliance on test order.

## Conventions

- **Errors**: expected, fixable failures raise a `PipelineError` subclass with a message that says what happened and what to do. Catch broad exceptions only at the entry point (`main()`).
- **Types**: everything in `src/` is annotated; `mypy --strict` must pass. Avoid `# type: ignore`; if one is needed, add a reason.
- **Configuration** comes from environment variables read in `src/config.py`. Never hard-code or print secrets.
- **SQL** lives in `sql/`; Python runs it. Transform scripts are idempotent (full rebuild).
- **Raw data is never modified** after loading; cleaning happens in `staging`.
- **Evaluation** uses time-ordered splits; the test snapshot is not used to choose models or hyperparameters.

## Git workflow

- **`main` is always green.** It is protected: changes reach it only through a pull request whose CI run passed.
- **One change per branch**, named by kind and subject: `feat/airflow-dags`, `fix/champion-rescoring`, `docs/problem-statement`, `chore/update-lock-files`. Keep branches short-lived and pull requests small enough to review in one sitting.
- **Commit messages** say what changed and why, in the imperative, with a short prefix: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`. Example: `fix: score the champion on the same data as the candidate`. The body explains the reason when it is not obvious.
- **Open a pull request** with the template (`.github/pull_request_template.md`): what and why, how it was checked, and whether any pinned result changed and why. A change that moves a pinned value says so in the description, not only in the diff.
- **Merge with squash**, so `main` has one commit per change, then delete the branch. Milestones are tagged (`step-4`).
- **Never commit** `.env`, anything under `docs/review/` (private working material), or caches. `git status` before every commit.
- **History is not rewritten** on `main`. If a mistake gets in, fix it forward with a new change.

## Adding a service

Put its container setup in `services/<name>/` (Dockerfile and config), add it to `docker-compose.yml` with a healthcheck and a port bound to `127.0.0.1`, pin versions that must match between server and client, and document it in the README.

## Commands

```
docker compose up -d postgres mlflow          # start the services
docker compose build                          # rebuild the images (after changing code or a lock file)
docker compose run --rm app                   # run the pipeline (load, transform, train)
docker compose run --rm quality               # the gate, with the fast training tests (about 14 minutes)
docker compose run --rm -e FULL_TRAINING_TESTS=1 quality   # the same, with the real hyperparameter grids (about 25 minutes)
docker compose run --rm quality python -m pytest -m "not e2e and not warehouse"   # quick feedback
docker compose run --rm quality python -m pytest -m warehouse --require-warehouse  # real-data regression
docker compose up -d api dashboard            # the API (:8000) and the dashboard (:8501)
docker compose run --rm dashboard-tests       # the dashboard's own tests (they need Streamlit)
```

The `app` image is the runtime image: runtime libraries only, no tests or dev tools, and it runs as a non-root user. The tests and dev tools live in the `quality` image. The `app` image holds a copy of the code from when it was built, so rebuild it (`docker compose build app`) after changing code, or the pipeline runs the old version.

### Dependencies and reproducibility

- `pyproject.toml` lists only direct dependencies. The exact versions, with hashes, are in `requirements.lock` (runtime) and `requirements-dev.lock` (runtime plus pytest, ruff and mypy). The images and CI install from these with `--require-hashes`, and a test checks that the installed libraries match the lock file.
- After changing `pyproject.toml`, regenerate both lock files with the command at the top of each file, then rebuild the images. Add `-c` with the output of `pip freeze` to keep today's versions instead of taking the newest allowed ones; then run the gate, because a library upgrade can change model results.
- Training runs are tagged with where they came from: `git_sha`, the checksum of the Excel file that was loaded (`raw_file_sha256`, recorded at load time in `raw.load_log`), a checksum of the transform SQL, the lock file checksum and library versions. The image has no `.git` folder, so pass the commit when building: `GIT_SHA=$(git rev-parse HEAD) docker compose build`. Without it the tag says `unknown`.

Continuous integration (`.github/workflows/ci.yml`) has four jobs. `quality` (lint, format, strict types and the fast tests) and `dashboard` (the Streamlit tests) run on every push and pull request and are the checks a pull request must pass. `full-training` (the same tests with the real hyperparameter grids and 1,000 bootstrap resamples) and `regression` (loads the real dataset and checks the pinned results reproduce) run after a merge to `main` and by hand; they cannot be required checks because they do not run on pull requests, so a red run on `main` is fixed with a new pull request.

### Fast and full training tests

Tests that train models check plumbing (is the decision logged, is the model registered), not the numbers, so by default they train with one hyperparameter combination per algorithm and 100 bootstrap resamples (the `quick_training` fixture in `tests/conftest.py`). That more than halves the run time. `FULL_TRAINING_TESTS=1` switches to the real grids and 1,000 resamples. Tests marked `warehouse` always use the real settings, because they pin the real results. `tests/test_training_modes.py` checks which mode is active, so neither can silently turn into the other. Do not lower an assertion to make a quick-mode test pass: if a test only passes with the real settings, it belongs in the full run.

Not set up yet: pre-commit hooks and vulnerability scanning of the dependencies.

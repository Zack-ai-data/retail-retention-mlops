# Machine Learning Engineering workflow

How changes are made here, so that anyone (including the person who wrote the code) can say what a change does and does not guarantee.

## Definition of done

A change (new feature, bug fix, refactor, dependency bump) is done only when all of these hold:

1. **Tests exist for the new or changed behaviour.** A bug fix starts with a test that fails for the bug.
2. **`docker compose run --rm quality` exits 0.** It runs `ruff check`, `ruff format --check`, `mypy --strict` (on `src/`) and `pytest --cov` with a 90% branch-coverage minimum.
3. **The real-data regression tests pass.** They copy the loaded raw table into a scratch database and run the current SQL and model code on it, so the real data must be loaded (`docker compose run --rm app` once). Run them with `--require-warehouse`, which turns a skip into a failure, so a green run really did check the data.
4. **Intentional changes to results update the pinned values in the same change,** in `tests/test_warehouse_regression.py`, with the reason in the commit message. An unexplained change in those numbers is a regression until shown otherwise.
5. **Docs match the code:** the README (commands, structure) and the problem statement where behaviour or assumptions changed.

## What "no regression" means, precisely

A green gate proves that the code is lint-clean, formatted and type-consistent; that every test passes with at least 90% of branches exercised; and that the current cleaning and feature SQL, run on a copy of the real raw data, still gives the same pinned counts and model results (skipped when the real data is not loaded, so a green run without `--require-warehouse` does not prove this part).

It does **not** prove that behaviour no test covers is correct, that the business assumptions are right (for example the 90-day churn horizon), or that the pinned numbers are themselves correct: they detect change and are not an independent oracle.

So the accurate statement after a green gate is "nothing the tests pin has changed", not "nothing broke".

## Coverage is a floor, not a goal

The gate requires 90% branch coverage so that large untested areas cannot creep in. It is not a target to maximise. Do not add tests only to raise the number. Put the effort where a mistake would matter (data correctness, the promotion decision, metrics, error handling) and write precise assertions there. The stronger measure of test quality is whether tests fail when the code is wrong; mutation testing checks that, and was run once on a few pure functions. It is not part of the gate.

## Where tests go

Tests live in `tests/`, named after the module they test. Keep them deterministic: seed randomness, no network, no reliance on test order.

- SQL (cleaning, features, labels) is tested by running the real SQL on a throwaway Postgres database with hand-built rows.
- `-m e2e` runs the whole pipeline on a small synthetic workbook.
- Results on the real data are pinned in `tests/test_warehouse_regression.py` (`-m warehouse`).
- The dashboard's pages need Streamlit, so they are skipped in the `quality` run; run them with `docker compose run --rm dashboard-tests`.
- A new way for things to fail gets an exception type in `src/errors.py` and a test for its message in `tests/test_error_handling.py`.

## Conventions

- **Errors:** expected, fixable failures raise a `PipelineError` subclass with a message that says what happened and what to do. Catch broad exceptions only at the entry point (`main()`).
- **Types:** everything in `src/` is annotated and `mypy --strict` must pass. Avoid `# type: ignore`; if one is needed, add a reason.
- **Configuration** comes from environment variables read in `src/config.py`. Never hard-code or print secrets.
- **SQL** lives in `sql/`; Python runs it. Transform scripts are idempotent (full rebuild).
- **Raw data is never modified** after loading; cleaning happens in `staging`.
- **Evaluation** uses time-ordered splits; the test snapshot is not used to choose models or hyperparameters.

## Git workflow

- **`main` is always green.** It is protected: changes reach it only through a pull request whose CI run passed.
- **One change per branch**, named by kind and subject (`feat/airflow-dags`, `fix/champion-rescoring`, `docs/problem-statement`). Keep branches short-lived and pull requests small.
- **Commit messages** say what changed and why, in the imperative, with a prefix: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.
- **Open a pull request** with the template (`.github/pull_request_template.md`): what and why, how it was checked, and whether any pinned result changed and why.
- **Merge with squash**, then delete the branch.
- **Never commit** `.env`, anything under `docs/review/` or `notes/` (private working material), or caches. Run `git status` before every commit.
- **History is not rewritten** on `main`. If a mistake gets in, fix it forward with a new change.

## Releasing

Releases publish the API and dashboard images to the GitHub Container Registry (`.github/workflows/deploy.yml`). Tag a commit that is already on `main` and green:

```
git switch main && git pull
git tag -a v0.1.0 -m "What this release contains"
git push origin v0.1.0
```

The workflow builds both images, starts each to check it runs and does not run as root, and then publishes them as `ghcr.io/<owner>/<repo>-api` and `-dashboard`. To try the build without publishing, run the workflow by hand from the Actions tab and leave "Publish the images" unticked. Do not delete or move a published tag: publish a new version instead.

## Adding a service

Put its container setup in `services/<name>/` (Dockerfile and config), add it to `docker-compose.yml` with a healthcheck and a port bound to `127.0.0.1`, pin versions that must match between server and client, and document it in the README.

## Commands

```
docker compose up -d postgres mlflow          # start the services
docker compose build                          # rebuild the images (after changing code or a lock file)
docker compose run --rm app                   # run the pipeline (load, transform, train)
docker compose run --rm quality               # the gate, with the fast training tests
docker compose run --rm -e FULL_TRAINING_TESTS=1 quality   # the same, with the real hyperparameter grids
docker compose run --rm quality python -m pytest -m "not e2e and not warehouse"   # quick feedback
docker compose run --rm quality python -m pytest -m warehouse --require-warehouse  # real-data regression
docker compose up -d api dashboard            # the API (:8000) and the dashboard (:8501)
docker compose run --rm dashboard-tests       # the dashboard's own tests (they need Streamlit)
docker compose --profile airflow up -d airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer   # Airflow, http://localhost:8080
docker compose --profile airflow exec airflow-scheduler airflow dags trigger retail_etl
docker compose --profile airflow exec airflow-scheduler airflow dags trigger retail_retrain_score
docker compose --profile airflow stop         # stop Airflow (its data is kept)
docker compose --profile monitoring up -d     # Prometheus (:9090) and Grafana (:3000); start the API too
docker compose stop grafana prometheus        # stop only monitoring
docker compose restart grafana                # after editing a dashboard or provisioning file
docker compose --profile monitoring up grafana-db-init   # re-run the read-only login's grants
docker compose run --rm app python -m src.main --stage monitor   # refresh the tables the dashboards read
docker compose run --rm app python -m src.demo_traffic --api-url http://api:8000 --minutes 5   # synthetic demo traffic
```

The `app` image is the runtime image: runtime libraries only, no tests or dev tools, and it runs as a non-root user. The tests and dev tools live in the `quality` image. The `app` image holds a copy of the code from when it was built, so rebuild it (`docker compose build app`) after changing code, or the pipeline runs the old version.

### Dependencies and reproducibility

- `pyproject.toml` lists only direct dependencies. The exact versions, with hashes, are in `requirements.lock` (runtime) and `requirements-dev.lock` (runtime plus pytest, ruff and mypy). The images and CI install from these with `--require-hashes`, and a test checks that the installed libraries match.
- After changing `pyproject.toml`, regenerate both lock files with the command at the top of each file, then rebuild the images. Add `-c` with the output of `pip freeze` to keep today's versions instead of taking the newest allowed ones; then run the gate, because a library upgrade can change model results.
- Dependabot (`.github/dependabot.yml`) proposes updates to GitHub Actions, Docker base images and the images in `docker-compose.yml`. It never touches the Python libraries (regenerate those as above), a major version of the stateful services, or the Python of the images. Review its pull requests like any other: CI must pass first. When you add an image, a test fails until the image has a decision in the Dependabot file.
- Training runs are tagged with where they came from: the git commit, a checksum of the Excel file, of the transform SQL and of the lock file, and the library versions. The image has no `.git` folder, so pass the commit when building: `GIT_SHA=$(git rev-parse HEAD) docker compose build`. Without it the tag says `unknown`.

### Continuous integration

`.github/workflows/ci.yml` has six jobs. `quality`, `dashboard` and `airflow-dags` run on every push and pull request and are the required checks. `full-training`, `regression` and `fresh-machine` run after a merge to `main` and by hand; they cannot be required checks, so a red run on `main` is fixed with a new pull request. All six also run every Monday on a schedule, as a reproducibility check of what is not pinned (Docker images, the runner image, the actions). To try the jobs without waiting, use Actions > CI > Run workflow.

### Fast and full training tests

Tests that train models check plumbing (is the decision logged, is the model registered), not the numbers, so by default they train with one hyperparameter combination per algorithm and 100 bootstrap resamples (the `quick_training` fixture in `tests/conftest.py`). `FULL_TRAINING_TESTS=1` switches to the real grids and 1,000 resamples. Tests marked `warehouse` always use the real settings, because they pin the real results. Do not lower an assertion to make a quick-mode test pass: if a test only passes with the real settings, it belongs in the full run.

Not set up yet: pre-commit hooks.

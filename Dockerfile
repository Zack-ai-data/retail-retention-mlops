# Two images from one file:
#   runtime  what runs the pipeline (and, later, the API): runtime dependencies only, no tests or
#            dev tools, as a non-root user.         docker compose build app
#   dev      runtime + pytest, ruff, mypy and the tests, for the quality gate.
#                                                    docker compose build quality
# Both install the exact versions in the lock files (requirements.lock, requirements-dev.lock),
# with hash checking, so the same code gets the same libraries on every machine and in CI.
# Pass the commit so training runs can record it (the image has no .git folder):
#   GIT_SHA=$(git rev-parse HEAD) docker compose build

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# The dependency layer only changes when a lock file does (editing the README or the code does not
# reinstall anything).
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock


FROM base AS runtime

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    HOME=/home/app

RUN useradd --create-home --uid 1000 app

COPY src ./src
COPY sql ./sql

USER app
CMD ["python", "-m", "src.main"]


FROM base AS dev

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

COPY requirements-dev.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-dev.lock

COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests
COPY sql ./sql

CMD ["python", "-m", "src.main"]

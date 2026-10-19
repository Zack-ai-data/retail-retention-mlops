# The Streamlit dashboard. It talks to the API over HTTP and imports only src/dashboard_client.py,
# so it needs none of the training stack (scikit-learn, MLflow, the database driver).
#   runtime  what runs the dashboard, as a non-root user.   docker compose up -d dashboard
#   test     runtime + pytest, for the dashboard's tests.   docker compose run --rm dashboard-tests
# Build from the project folder (the compose file does): the context must contain src/.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY services/dashboard/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

RUN useradd --create-home --uid 1000 app

COPY src/__init__.py src/dashboard_client.py ./src/
COPY services/dashboard/app.py ./app.py

USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]


FROM runtime AS test

USER root
COPY services/dashboard/requirements-test.lock ./requirements-test.lock
RUN pip install --no-cache-dir --require-hashes -r requirements-test.lock

CMD ["python", "-m", "pytest", "-q"]

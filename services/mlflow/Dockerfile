FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep the server and the client (pyproject.toml) on the same MLflow version
RUN pip install --no-cache-dir "mlflow==3.17.1" psycopg2-binary

COPY services/mlflow/entrypoint.py /entrypoint.py

EXPOSE 5000
ENTRYPOINT ["python", "/entrypoint.py"]

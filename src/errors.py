"""Exceptions for failures the person running the pipeline can fix.

Each carries a message that says what went wrong and what to do about it. `src.main` prints these
without a traceback and exits with a non-zero status; anything else is a bug and keeps its
traceback.
"""


class PipelineError(Exception):
    """Base class for expected, explainable pipeline failures."""


class ConfigurationError(PipelineError):
    """A required setting (for example a database credential) is missing."""


class DataFileError(PipelineError):
    """The source data file is missing, unreadable or does not have the expected columns."""


class DatabaseUnavailableError(PipelineError):
    """PostgreSQL could not be reached."""


class TrackingServerError(PipelineError):
    """The MLflow tracking server could not be reached or rejected a request."""


class EmptyTableError(PipelineError):
    """A table a stage depends on is missing or empty (an earlier stage probably has not run)."""


class RegistryError(PipelineError):
    """The model registry is in a state the pipeline will not act on (for example a champion whose
    metric cannot be read)."""


class ScoringError(PipelineError):
    """The models returned scores that cannot be used (missing or impossible values)."""


class DataQualityError(PipelineError):
    """One or more data quality checks failed."""

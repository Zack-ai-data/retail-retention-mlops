"""Packaging: declared dependencies, the lock files, and the Dockerfile.

These catch the mistakes that only show up in a fresh image or on another machine: importing a
library nobody declared, a lock file that has drifted from pyproject.toml, or dev tools leaking
into the runtime image.
"""

import ast
import importlib.metadata
import re
import sys
import tomllib

import pytest

from src import config

ROOT = config.PROJECT_ROOT
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement):
    return _normalise(re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0])


# The dashboard tests import Streamlit, which only the dashboard image has
DASHBOARD_TEST_TOOLS = {"streamlit"}
RUNTIME = {_requirement_name(r) for r in PYPROJECT["project"]["dependencies"]}
DEV = RUNTIME | {_requirement_name(r) for r in PYPROJECT["project"]["optional-dependencies"]["dev"]}


def _lock_pins(filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    return {
        _normalise(m.group(1)): m.group(2)
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", text, flags=re.M)
    }


def _third_party_imports(folder):
    """Top-level names imported by the Python files in a folder, excluding the standard library
    and the project's own packages."""
    own = {"src", "tests"}
    found = {}
    for path in sorted((ROOT / folder).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in sys.stdlib_module_names and name not in own:
                    found.setdefault(name, path.name)
    return found


def _distributions_providing(module):
    return {_normalise(d) for d in importlib.metadata.packages_distributions().get(module, [])}


def _dashboard_package(module):
    """The package that provides a module the dashboard imports. The dashboard's libraries are not
    installed in this image, so when the lookup finds nothing the import name is used (true of
    streamlit, requests and pandas)."""
    return _distributions_providing(module) or {_normalise(module)}


# ------------------------------------------------------------ declared dependencies


def test_every_library_the_application_imports_is_a_declared_dependency():
    undeclared = {
        f"{module} (imported in {where})": sorted(_distributions_providing(module))
        for module, where in _third_party_imports("src").items()
        if not (_distributions_providing(module) & RUNTIME)
    }

    assert not undeclared, f"imported but not in [project].dependencies: {undeclared}"


def test_every_library_the_tests_import_is_declared_somewhere():
    undeclared = {
        f"{module} (imported in {where})": sorted(_distributions_providing(module))
        for module, where in _third_party_imports("tests").items()
        if not (
            (_distributions_providing(module) or _dashboard_package(module))
            & (DEV | DASHBOARD_TEST_TOOLS)
        )
    }

    assert not undeclared, f"imported by tests but not declared: {undeclared}"


def test_libraries_the_application_does_not_use_are_not_declared():
    imported = set()
    for module in _third_party_imports("src"):
        imported |= _distributions_providing(module)
    # openpyxl is used by pandas to read the Excel file; uvicorn is the server process that runs
    # the API (`uvicorn src.api:app`), so nothing imports it
    used_indirectly = {"openpyxl", "uvicorn"}

    unused = RUNTIME - imported - used_indirectly

    assert not unused, f"declared but never imported in src/: {sorted(unused)}"


# ---------------------------------------------------------------- lock files


@pytest.mark.parametrize(
    "lock, declared", [("requirements.lock", RUNTIME), ("requirements-dev.lock", DEV)]
)
def test_the_lock_files_pin_every_declared_dependency(lock, declared):
    pins = _lock_pins(lock)

    assert declared <= set(pins), f"{lock} is missing {sorted(declared - set(pins))}"


def test_the_runtime_lock_has_no_development_tools():
    runtime = set(_lock_pins("requirements.lock"))

    leaked = runtime & {"pytest", "pytest-cov", "ruff", "mypy", "sqlalchemy", "alembic"}

    assert not leaked, f"dev tools in the runtime image: {sorted(leaked)}"


def test_the_dev_lock_is_the_runtime_lock_plus_tools_with_the_same_versions():
    runtime, dev = _lock_pins("requirements.lock"), _lock_pins("requirements-dev.lock")

    assert set(runtime) <= set(dev)
    assert {p: v for p, v in runtime.items() if dev[p] != v} == {}
    assert {"pytest", "ruff", "mypy"} <= set(dev) - set(runtime)


@pytest.mark.parametrize("lock", ["requirements.lock", "requirements-dev.lock"])
def test_every_pinned_package_has_a_hash(lock):
    text = (ROOT / lock).read_text(encoding="utf-8")
    entries = re.split(r"^(?=[A-Za-z0-9_.\-]+==)", text, flags=re.M)[1:]

    unhashed = [e.split("==")[0] for e in entries if "--hash=sha256:" not in e]

    assert entries and not unhashed, f"no hash for {unhashed}"


def test_the_installed_libraries_are_the_ones_in_the_lock_file():
    """The environment these tests run in was built from requirements-dev.lock; if a library has
    drifted, results can change for reasons that have nothing to do with the code."""
    pins = _lock_pins("requirements-dev.lock")
    wrong = {}
    for package, pinned in pins.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            wrong[package] = (pinned, "not installed")
            continue
        if installed != pinned:
            wrong[package] = (pinned, installed)

    assert not wrong, f"installed differs from requirements-dev.lock (pinned, installed): {wrong}"


# ---------------------------------------------------------------- Dockerfile


def _stages():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    parts = re.split(r"^FROM .* AS (\w+)$", text, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def test_the_dockerfile_has_a_runtime_and_a_dev_stage_built_from_the_same_base():
    assert set(_stages()) == {"base", "runtime", "dev"}


def test_the_runtime_image_has_no_tests_no_dev_tools_and_a_non_root_user():
    runtime = _stages()["runtime"]

    assert "requirements-dev.lock" not in runtime
    assert "COPY tests" not in runtime
    assert re.search(r"^USER app$", runtime, flags=re.M)
    assert "useradd" in runtime


def test_dependencies_are_installed_from_the_lock_files_with_hash_checking():
    stages = _stages()

    assert "--require-hashes -r requirements.lock" in stages["base"]
    assert "--require-hashes -r requirements-dev.lock" in stages["dev"]
    assert "pip install ." not in "".join(stages.values())  # nothing resolved at build time


def test_changing_the_readme_or_code_does_not_reinstall_the_dependencies():
    base = _stages()["base"]

    copied = re.findall(r"^COPY (.+) \./?$", base, flags=re.M)

    assert copied == ["requirements.lock"]  # the only file the dependency layer depends on


def test_the_commit_can_be_passed_in_because_the_image_has_no_git_folder():
    stages = _stages()

    assert all("ARG GIT_SHA" in stages[s] and "GIT_SHA=" in stages[s] for s in ("runtime", "dev"))
    assert ".git" in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------- the dashboard image

DASHBOARD = "services/dashboard"
TRAINING_STACK = {"scikit-learn", "mlflow-skinny", "mlflow", "psycopg", "psycopg-binary", "scipy"}


def _requirement_names(filename):
    """The packages named in a requirements .in file."""
    return {
        _requirement_name(line)
        for line in (ROOT / DASHBOARD / filename).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-r"))
    }


def test_the_dashboard_lock_pins_what_it_declares_and_nothing_from_the_training_stack():
    pins = _lock_pins(f"{DASHBOARD}/requirements.lock")

    assert _requirement_names("requirements.in") <= set(pins)
    assert not (set(pins) & TRAINING_STACK), "the dashboard image must not carry the training stack"


def test_the_dashboard_test_lock_adds_only_test_tools():
    runtime = _lock_pins(f"{DASHBOARD}/requirements.lock")
    tests = _lock_pins(f"{DASHBOARD}/requirements-test.lock")

    assert set(runtime) <= set(tests)
    assert "pytest" in set(tests) - set(runtime)
    assert {p: v for p, v in runtime.items() if tests[p] != v} == {}


def test_the_dashboard_uses_the_same_pandas_numpy_and_requests_as_the_api():
    api = _lock_pins("requirements.lock")
    dashboard = _lock_pins(f"{DASHBOARD}/requirements.lock")

    for package in ("pandas", "numpy", "requests"):
        assert dashboard[package] == api[package], package


def test_everything_the_dashboard_imports_is_in_its_image():
    sources = [ROOT / DASHBOARD / "app.py", ROOT / "src" / "dashboard_client.py"]
    imported = set()
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = {m for m in imported if m not in sys.stdlib_module_names and m != "src"}
    in_image = set(_lock_pins(f"{DASHBOARD}/requirements.lock"))

    missing = {
        m: sorted(_dashboard_package(m))
        for m in third_party
        if not (_dashboard_package(m) & in_image)
    }

    assert third_party, "the scan should have found the dashboard's imports"
    assert not missing, f"imported by the dashboard but not in its image: {missing}"


# ------------------------------------------------------------ the release workflow

RELEASE = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")


def _release_images():
    return re.findall(r"- name: (\w+)\n\s+dockerfile: (\S+)\n\s+target: (\w+)", RELEASE)


def test_the_release_builds_the_api_and_the_dashboard():
    assert {name for name, _, _ in _release_images()} == {"api", "dashboard"}


def test_each_release_image_points_at_a_real_dockerfile_and_stage():
    for name, dockerfile, target in _release_images():
        path = ROOT / dockerfile
        assert path.exists(), f"{name}: {dockerfile} does not exist"
        stages = re.findall(r"^FROM .* AS (\w+)$", path.read_text(encoding="utf-8"), flags=re.M)
        assert target in stages, f"{name}: {dockerfile} has no stage {target!r} (has {stages})"


def test_the_release_publishes_only_on_a_version_tag_or_when_asked():
    assert re.search(r'tags: \["v\*"\]', RELEASE)
    assert "branches:" not in RELEASE  # a push to a branch must never publish
    assert "startsWith(github.ref, 'refs/tags/v') || inputs.publish" in RELEASE


def test_the_image_is_checked_before_it_is_published():
    assert RELEASE.index("Check the image") < RELEASE.index("- name: Publish")


def test_only_the_release_workflow_may_write_packages():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "packages: write" in RELEASE
    assert "packages" not in ci  # CI runs on pull requests and needs read access only

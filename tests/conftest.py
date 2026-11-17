"""
Pytest configuration and session-wide fixtures.
"""
import os
import glob
import pytest


@pytest.fixture(scope="session", autouse=True)
def guard_models_and_reports():
    """Record modification times of models/*.pkl and reports/* and fail if any change."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")

    def snapshot():
        mtimes = {}
        for p in glob.glob(os.path.join(models_dir, "*.pkl")):
            if os.path.isfile(p):
                mtimes[os.path.abspath(p)] = os.path.getmtime(p)
        for p in glob.glob(os.path.join(reports_dir, "*")):
            if os.path.isfile(p):
                mtimes[os.path.abspath(p)] = os.path.getmtime(p)
        return mtimes

    before = snapshot()

    yield

    after = snapshot()
    changed = []
    for path, mtime in before.items():
        if path not in after:
            changed.append(f"Deleted: {os.path.relpath(path, project_root)}")
        elif after[path] != mtime:
            changed.append(f"Modified: {os.path.relpath(path, project_root)}")

    for path in after:
        if path not in before:
            changed.append(f"Created: {os.path.relpath(path, project_root)}")

    assert not changed, f"Tests modified models/*.pkl or reports/*: {changed}"

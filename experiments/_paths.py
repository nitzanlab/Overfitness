"""Shared path helpers for experiment scripts."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def data_path(name):
    """Return absolute path under data/ and ensure parent dir exists."""
    p = os.path.join(DATA_DIR, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

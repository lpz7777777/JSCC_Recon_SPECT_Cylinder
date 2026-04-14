import os
import sys


def setup_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(this_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root

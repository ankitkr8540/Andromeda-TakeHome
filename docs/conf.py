"""Sphinx configuration for guides and API documentation from real imports."""

import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
project = "Andromeda function approximation"
author = metadata["authors"][0]["name"]
release = metadata["version"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", ".DS_Store"]
root_doc = "index"

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
nitpicky = True

html_theme = "alabaster"
html_title = project
html_theme_options = {"description": "MLP and learnable Fourier regression"}

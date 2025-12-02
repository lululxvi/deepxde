# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
from unittest.mock import MagicMock

# -- Project information -----------------------------------------------------

project = "DeepXDE"
copyright = "2019, Lu Lu"
author = "Lu Lu"

# The short X.Y version and full version
try:
    release = get_version("deepxde")
except PackageNotFoundError:
    # Fallback when building docs without the package installed
    release = "dev"

version = ".".join(release.split(".")[:2])

# -- Mock heavy backends so autodoc doesn't load real TF/Torch/Paddle/JAX ----

AUTODOC_MOCK_MODULES = [
    "tensorflow",
    "tensorflow.compat",
    "tensorflow.compat.v1",
    "tensorflow_probability",
    "tensorflow_probability.substrates",
    "tensorflow_probability.substrates.numpy",
    "tensorflow_probability.substrates.jax",
    "tensorflow_probability.substrates.tensorflow",
    "jax",
    "jax.numpy",
    "flax",
    "optax",
    "torch",
    "paddle",
    "paddlepaddle",
]

class _Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

for _mod in AUTODOC_MOCK_MODULES:
    sys.modules.setdefault(_mod, _Mock())

autodoc_mock_imports = AUTODOC_MOCK_MODULES

# Ensure DeepXDE picks some backend, but since everything is mocked, it's safe.
os.environ.setdefault("DDEBACKEND", "tensorflow")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

# Only .rst sources
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Sphinx 8 warns if language is None; use explicit "en".
language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "DeepXDEdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, "DeepXDE.tex", "DeepXDE Documentation", "Lu Lu", "manual")
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "deepxde", "DeepXDE Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "DeepXDE",
        "DeepXDE Documentation",
        author,
        "DeepXDE",
        "One line description of project.",
        "Miscellaneous",
    )
]

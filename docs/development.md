# Build and maintain the documentation

## Install documentation tools

Sphinx and MyST are in the optional `docs` dependency group, separate from
runtime dependencies:

```bash
uv sync --locked --inexact --group docs
```

## Build HTML

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html
```

Alternatively, after installation:

```bash
make -C docs html
```

The build treats warnings as failures, checks references in nitpicky mode,
and imports the actual package for API pages. It does not mock runtime
dependencies, start MLflow, train benchmark models, or execute notebooks.
Generated HTML and doctrees are ignored by Git.

## Check examples and implementation

```bash
uv run --group docs sphinx-build -W --keep-going -b doctest docs docs/_build/doctest
MPLBACKEND=Agg uv run python -m pytest -q
uv run ruff check function_approximation scripts tests docs/conf.py
```

The quickstart contains an executable shape example. The test suite covers full
training, checkpoint ownership, train-only preprocessing, inference without
target calls, invalid settings, and MLflow artifacts against temporary local
SQLite stores.

## Writing conventions

- Put user-facing guides in Markdown; MyST integrates them into Sphinx.
- Add new pages to `docs/index.rst` or the relevant API table of contents.
- Use Google-style Python docstrings with a short summary and relevant `Args`,
  `Returns`, `Raises`, and `Notes` sections. Sphinx Napoleon renders them.
- Document input/output shapes, coordinate units, fitted state, side effects,
  and data-split boundaries where they matter.
- Explain why numerical or lifecycle choices exist. Avoid comments that only
  repeat the next line of code.
- Keep published config examples synchronized through `literalinclude` when
  referring to a specific checked-in YAML.

API pages use [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
and [Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).
MyST provides [Markdown support](https://www.sphinx-doc.org/en/master/usage/markdown.html).

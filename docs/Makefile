SPHINXBUILD ?= ../.venv/bin/sphinx-build
BUILDDIR = _build

.PHONY: html doctest clean

html:
	$(SPHINXBUILD) -W --keep-going -b html . $(BUILDDIR)/html

doctest:
	$(SPHINXBUILD) -W --keep-going -b doctest . $(BUILDDIR)/doctest

clean:
	$(SPHINXBUILD) -M clean . $(BUILDDIR)

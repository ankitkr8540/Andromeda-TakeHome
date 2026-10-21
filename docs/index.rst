Andromeda function approximation
================================

Train an MLP baseline or a learnable Fourier model from sampled scalar
observations. One YAML-driven pipeline handles preprocessing, Adam training,
validation-based checkpoint selection, metrics, plots, and optional MLflow tracking.

The Fourier model estimates starting frequencies from training observations with
Lomb–Scargle and fits coefficients with least squares before Adam refinement.
The synthetic target generates labels and visual references; saved-model
inference uses learned parameters and the fitted preprocessor only.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   workflow
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference and development

   api/index
   development

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

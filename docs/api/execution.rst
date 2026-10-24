Experiment and study scripts
============================

These functions train experiments or evaluate existing artifacts when explicitly
called. Importing the modules for this reference does not launch training.

.. automodule:: scripts.run_experiments
   :members: set_seed, evaluate_splits, run_experiment

.. automodule:: scripts.run_ablation
   :members: prepare_jobs, write_json, summarize, run_study

.. automodule:: scripts.run_multiseed
   :members: aggregate

.. automodule:: scripts.evaluate_extrapolation_distance
   :members: distance_bands, evaluate_runs

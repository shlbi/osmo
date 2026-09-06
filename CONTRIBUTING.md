# Contributing

Install `pip install -e ".[test]"` and run `python -m pytest -q`. Keep changes small and document numerical assumptions, supported units and dataset boundaries.

For model comparisons, hold cohorts, splits, preprocessing and selection rules constant. Report observed endpoints accurately; never substitute scalar-property scores for organ concentration accuracy. Do not optimize against a sealed test set.

Include a meaningful regression test for numerical or data-integrity fixes. Use generated toy fixtures where possible. Do not commit credentials, patient data, model checkpoints or datasets without an explicit redistribution basis. Cite external methods and do not copy code under an incompatible license into MIT files.

Please discuss larger features in an issue before implementation. Contributions to original project code are provided under this repository's MIT license.

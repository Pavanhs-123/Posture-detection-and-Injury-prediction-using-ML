# Local model artifacts

Serialized models and `model_registry.json` are intentionally not included in the repository branch. They are generated, local-only artifacts and are ignored by `.gitignore`.

The active movement classifier supports `LUNGE`, `SMASH_LANDING`, and `CHANGE_OF_DIRECTION`. Its absence is handled by the explicitly identified heuristic baseline. Automated clinical `LOW`/`MODERATE`/`HIGH` injury-risk prediction is disabled. Do not load pickle files from untrusted sources.

Create models only from independently reviewed data with the grouped evaluation tools. The current dataset is not included in the public branch, and the available sample is too small to support generalization claims.

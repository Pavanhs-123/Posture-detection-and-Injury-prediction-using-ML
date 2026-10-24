# Migration audit and branch preparation

Date: 2026-10-24

## Starting point

This staging branch was created locally from GitHub `main` at `a60dafc9859f95d0df96a4fd44051d89b464575b`. The remote branch named `modified` did not exist when inspected. The original Windows project folder has no `.git` metadata.

A separate local backup was created and checked before migration. Source, tests, docs, data, and model files were copied; the project virtual environment and top-level caches were excluded. Dataset and model hashes were verified against the source. The backup is not part of the repository.

The project folder did not contain the three audit reports named in the task (`AUDIT_REPORT.md`, `GITHUB_AUDIT_BASELINE.md`, `GITHUB_AUDIT_REPORT.md`), so their contents could not be inspected or copied. This report records checks performed directly against the current source instead.

## Baseline checks

- Windows `.venv`: Python 3.11.0.
- Baseline test command: `.venv\\Scripts\\python.exe -m pytest -q` — **85 passed**.
- `.venv\\Scripts\\python.exe -m pip check` — no broken requirements.
- Current dataset: 34 labeled rows across six source videos. The three active movement classes are `LUNGE` (18), `SMASH_LANDING` (10), and `CHANGE_OF_DIRECTION` (6).
- The two local serialized model files had the same SHA-256 before migration. Their values are kept in the local backup/final audit record and neither model file is included in this branch.
- Linux testing was not performed.

## Verified project safeguards

The current code retains the three active movement classes, deferred categories disabled by default, uncertain/no-event outcomes, human review, event-manifest provenance, grouped model evaluation, and missing-measurement handling. Automated clinical injury-risk categories remain disabled. The standalone trainer now refuses splits that are not independent by video and class-complete.

## Public branch comparison

The public `main` commit above is an older, compact implementation. The local version adds modular candidate detection, review/manifest support, grouped evaluation, uncertainty handling, and regression tests. Public-only documentation and the empty root `pose_data.csv` were retained. The public root pose task asset matched the local asset by Git blob hash; the branch moves it under `assets/` without changing its contents.

## Branch inclusion and exclusions

Included: current application and Python source, tests, project documentation, pinned direct runtime dependencies, a development test dependency, `.gitignore`, and the MediaPipe pose task asset.

Excluded: all `data/` content (labeled dataset, human-review package, history database, raw videos, generated clips, keyframes, manifests and processed outputs); `models/*.pkl`; `models/model_registry.json`; virtual environments; caches; and `docs/legacy_tasks.txt`. These include user data, derived footage, generated artifacts, or private task notes and are not necessary to reproduce the source/test code.

`requirements-lock.txt` was not migrated: its pins differ from the verified environment (including NumPy, pandas, OpenCV and Streamlit). The direct runtime requirements are pinned to versions observed in the verified environment; `requirements-dev.txt` pins pytest. A clean installation from these files was not run, and no dependencies were installed during this migration.

## Remaining verification limits

The test suite and dependency check were run on Windows with Python 3.11.0. Linux compatibility, fresh environment installation, browser-based Streamlit interactions, and clinical validity were not verified. Model pickle files remain executable serialization formats and should only be loaded from trusted sources. No dataset or classifier artifact was published.

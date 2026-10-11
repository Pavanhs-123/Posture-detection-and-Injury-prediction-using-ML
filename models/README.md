# Models Directory

This directory stores trained model pipelines, encoders, and evaluation metadata.

## Artifacts Specification
- `model_registry.json`: Registry tracking the winning candidate, selection metric (Macro F1), validation scores, feature schema, and timestamp.
- `injury_risk_model.pkl`: Best validated injury-risk classifier pipeline (Random Forest, Extra Trees, or SVM).
- `movement_classifier.pkl`: Movement category classification pipeline.
- `randomforest_*.pkl`: Serialized Random Forest candidate pipeline.
- `extratrees_*.pkl`: Serialized Extra Trees candidate pipeline.
- `svm_*.pkl`: Serialized Support Vector Machine candidate pipeline.

## Reproducibility
All models are trained with explicit random seeds (`seed=42`) using video-grouped splitting.
Pickled artifacts are excluded from Git commits via `.gitignore`.

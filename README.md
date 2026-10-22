# Uncertainty-Aware PSO-SVM

## Research Objective
Investigate whether uncertainty-aware selection of PSO-tuned Support Vector Machines (SVMs) can reduce overtuning on small medical datasets.

## Research Idea
Particle Swarm Optimization (PSO) tunes the SVM hyperparameters C and gamma. We will compare standard best-validation-score selection against an uncertainty-aware selection strategy that considers both mean cross-validation loss and fold-to-fold variability.

## Datasets
- Breast Cancer Wisconsin Diagnostic
- UCI Heart Disease (Cleveland)
- Diabetes dataset (source and contents to be verified)

## Planned Workflow
1. Dataset verification and quality analysis
2. Data preprocessing
3. Baseline SVM evaluation
4. PSO-based hyperparameter optimization
5. Uncertainty-aware candidate selection
6. Nested cross-validation
7. Performance, stability, and overtuning analysis

## Important
Raw datasets are excluded from Git tracking. All preprocessing and model selection must be performed without using the outer test fold.

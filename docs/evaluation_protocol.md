
# Evaluation Protocol

## 1. Research Objective

Compare standard model selection with uncertainty-aware model selection for RBF-SVM hyperparameters tuned using Particle Swarm Optimization (PSO).

The goal is to investigate whether considering validation-score variability can reduce overtuning on small medical datasets.

## 2. Datasets

Evaluate the following datasets:

- Wisconsin Diagnostic Breast Cancer
- Cleveland Heart Disease
- Diabetes

Use the documented preprocessing protocol for all experiments.

## 3. Evaluation Design

Use nested stratified 5-fold cross-validation.

### Outer Cross-Validation

- Split the dataset into five stratified folds.
- Use four folds for model development.
- Reserve the remaining fold for final evaluation.
- Repeat until every fold has been used for evaluation.

### Inner Cross-Validation

Within each outer training partition:

- Use stratified 5-fold cross-validation to evaluate candidate SVM hyperparameters.
- Search over the RBF-SVM hyperparameters C and gamma using PSO.
- Do not use the outer evaluation fold for hyperparameter tuning or model selection.

## 4. Model-Selection Methods

### Method A: Standard Selection

Select the candidate with the lowest mean inner-CV loss.

### Method B: Uncertainty-Aware Selection

For each candidate, calculate:

S = mean inner-CV loss + lambda * standard deviation of inner-CV loss

Select the candidate with the lowest uncertainty-aware score.

The standard deviation measures variability across folds. It is a stability measure, not automatically a standard error.

## 5. Fair Comparison

- Use identical outer folds for both methods.
- Use the same preprocessing rules.
- Use the same SVM hyperparameter search bounds and computational budget.
- Record random seeds and PSO settings.
- Define lambda before examining outer-fold results, or select it using training data only.
- Do not use outer evaluation results to choose lambda or adjust the experiment.

## 6. Evaluation Metrics

Record:

- ROC-AUC
- Balanced accuracy
- Sensitivity (recall)
- Specificity
- F1-score
- Inner-to-outer performance gap
- Variability across outer folds
- Computational cost

For the overtuning gap, compare the selected model's inner-CV performance with its performance on the corresponding outer evaluation fold, using a consistent loss definition.

## 7. Reporting

Report the mean and standard deviation of outer-fold metrics for each method.

Report results separately for each dataset.

Do not claim that uncertainty-aware selection is better unless the experimental results support that conclusion.

## 8. Current Status

The existing ordinary 5-fold SVM baseline is preliminary.

The nested cross-validation comparison has not yet been implemented.

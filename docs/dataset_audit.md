# Dataset Audit and Preprocessing Decisions

## 1. Breast Cancer Wisconsin Diagnostic

- Samples: 569
- Columns: 32 (ID, diagnosis, 30 numeric predictors)
- Target: B = benign, M = malignant
- Missing values detected: None
- Duplicate rows detected: 0
- Class counts: B = 357, M = 212
- Decision: Exclude the ID column from model features.
- Decision: Standardize predictors within each training fold.

## 2. UCI Heart Disease - Cleveland

- Samples: 303
- Columns: 14 (13 predictors and target)
- Original target values: 0, 1, 2, 3, 4
- Missing entries: ca = 4, thal = 2
- Duplicate rows detected: 0
- Original class counts: 0 = 164, 1 = 55, 2 = 36, 3 = 35, 4 = 13
- Proposed binary mapping: original 0 -> 0; original 1-4 -> 1.
- Proposed missing-value treatment: impute within each training fold.
- Decision: Confirm and document feature types before finalizing preprocessing.

## 3. Diabetes

- Samples: 768
- Columns: 9 (8 predictors and target)
- Target values: 0 and 1
- Missing values detected by pandas: None
- Duplicate rows detected: 0
- Class counts: 0 = 500, 1 = 268
- Zero counts: glucose = 5, blood pressure = 35,
  skin thickness = 227, insulin = 374, BMI = 11.
- Decision pending: Verify dataset provenance and determine whether
  selected zero measurements should be treated as missing.
- Do not treat pregnancies = 0 as missing automatically.

## 4. Research safeguards

- Raw datasets are excluded from Git tracking by .gitignore.
- Do not use the outer test fold during preprocessing, PSO tuning,
  candidate selection, or other model-selection decisions.
- Fit preprocessing transformations using training data only.
- Evaluate the complete selection procedure using nested cross-validation.
- Compare standard selection and uncertainty-aware selection fairly,
  using identical outer splits and evaluation metrics.
- Report mean performance, variability, overtuning gap, and computation cost.

## 5. Open decisions

1. Verify and record the source and license of each dataset.
2. Finalize the Diabetes zero-value policy.
3. Define the inner-CV loss and uncertainty penalty.
4. Define the PSO search bounds and computational budget.
5. Fix the evaluation protocol before comparing methods.

## Diabetes Dataset Provenance

- The file schema matches the commonly used Pima Indians Diabetes dataset.
- The file contains 768 rows, eight predictors, and one binary target.
- The column descriptions match published descriptions of this dataset.
- Exact download URL and license for the local copy remain unverified.
- Original research reference: Smith et al. (1988), "Using the ADAP
  learning algorithm to forecast the onset of diabetes mellitus."
- Zero-valued measurements require a documented preprocessing policy.

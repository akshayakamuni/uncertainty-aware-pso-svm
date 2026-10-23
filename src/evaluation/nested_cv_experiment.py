
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_FOLDS = 5
INNER_FOLDS = 5

N_PARTICLES = 6
N_ITERATIONS = 4

LOWER_BOUND = -3.0
UPPER_BOUND = 3.0

# Fixed for this initial experiment.
LAMBDA_VALUE = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CREATE AN SVM MODEL
# ============================================================

def create_model(C, gamma, numeric_features, categorical_features):
    """
    Create a complete machine learning pipeline.

    Numerical data:
        Fill missing values and standardize.

    Categorical data:
        Fill missing values and encode categories.

    Finally, train an RBF-kernel SVM.
    """

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(C),
                    gamma=float(gamma),
                ),
            ),
        ]
    )

    return model


# ============================================================
# 3. EVALUATE ONE SVM CONFIGURATION
# ============================================================

def evaluate_particle(
    position,
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    inner_splits,
):
    """
    A particle represents a possible pair of C and gamma.

    Evaluate that pair using inner cross-validation.
    Return the loss from every inner fold.
    """

    log_C, log_gamma = position

    C = 10 ** log_C
    gamma = 10 ** log_gamma

    model = create_model(
        C,
        gamma,
        numeric_features,
        categorical_features,
    )

    auc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=inner_splits,
        scoring="roc_auc",
        n_jobs=1,
    )

    fold_losses = 1.0 - auc_scores

    return C, gamma, fold_losses


# ============================================================
# 4. RUN PSO USING ONLY OUTER TRAINING DATA
# ============================================================

def run_inner_pso(
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    outer_fold,
):
    """
    Search for SVM configurations using PSO.

    The outer test data is never used here.
    PSO minimizes mean inner cross-validation loss.
    """

    rng = np.random.default_rng(
        RANDOM_STATE + outer_fold
    )

    # Every particle contains log10(C) and log10(gamma).
    positions = rng.uniform(
        LOWER_BOUND,
        UPPER_BOUND,
        size=(N_PARTICLES, 2),
    )

    velocities = rng.uniform(
        -1.0,
        1.0,
        size=(N_PARTICLES, 2),
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    inner_splits = list(
        inner_cv.split(X_train, y_train)
    )

    personal_best_positions = positions.copy()
    personal_best_losses = np.full(
        N_PARTICLES,
        np.inf,
    )

    global_best_position = None
    global_best_loss = np.inf

    history = []

    inertia = 0.7
    cognitive = 1.5
    social = 1.5

    evaluation_number = 0

    for iteration in range(N_ITERATIONS):

        print(
            f"\nOuter fold {outer_fold}/{OUTER_FOLDS} "
            f"- PSO iteration {iteration + 1}/{N_ITERATIONS}"
        )

        for particle in range(N_PARTICLES):

            position = positions[particle].copy()

            C, gamma, fold_losses = evaluate_particle(
                position,
                X_train,
                y_train,
                numeric_features,
                categorical_features,
                inner_splits,
            )

            mean_loss = float(np.mean(fold_losses))
            std_loss = float(np.std(fold_losses, ddof=1))

            evaluation_number += 1

            history.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration + 1,
                    "particle": particle,
                    "evaluation": evaluation_number,
                    "C": C,
                    "gamma": gamma,
                    "mean_inner_loss": mean_loss,
                    "std_inner_loss": std_loss,
                    "fold_losses": fold_losses.tolist(),
                }
            )

            print(
                f"  Particle {particle}: "
                f"C={C:.6g}, gamma={gamma:.6g}, "
                f"mean loss={mean_loss:.4f}"
            )

            # Update this particle's best position.
            if mean_loss < personal_best_losses[particle]:
                personal_best_losses[particle] = mean_loss
                personal_best_positions[particle] = position.copy()

            # Update the best position found by the whole swarm.
            if mean_loss < global_best_loss:
                global_best_loss = mean_loss
                global_best_position = position.copy()

        # Move particles to explore new configurations.
        for particle in range(N_PARTICLES):

            r1 = rng.random(2)
            r2 = rng.random(2)

            velocities[particle] = (
                inertia * velocities[particle]
                + cognitive
                * r1
                * (
                    personal_best_positions[particle]
                    - positions[particle]
                )
                + social
                * r2
                * (
                    global_best_position
                    - positions[particle]
                )
            )

            positions[particle] += velocities[particle]

            positions[particle] = np.clip(
                positions[particle],
                LOWER_BOUND,
                UPPER_BOUND,
            )

    # Remove repeated C/gamma combinations.
    unique_candidates = {}

    for item in history:

        key = (
            round(item["C"], 10),
            round(item["gamma"], 10),
        )

        if key not in unique_candidates:
            unique_candidates[key] = item

    candidates = list(unique_candidates.values())

    return candidates, history, len(history)


# ============================================================
# 5. SELECT ONE CONFIGURATION USING EACH METHOD
# ============================================================

def select_candidates(candidates):
    """
    Standard method:
        Select the smallest mean loss.

    Uncertainty-aware method:
        Select the smallest mean loss + lambda * std deviation.
    """

    standard_candidate = min(
        candidates,
        key=lambda item: item["mean_inner_loss"],
    )

    uncertainty_candidate = min(
        candidates,
        key=lambda item: (
            item["mean_inner_loss"]
            + LAMBDA_VALUE * item["std_inner_loss"]
        ),
    )

    return {
        "Standard": standard_candidate,
        "Uncertainty-aware": uncertainty_candidate,
    }


# ============================================================
# 6. TEST A SELECTED MODEL ON THE OUTER TEST FOLD
# ============================================================

def evaluate_outer_test(
    candidate,
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    method_name,
):
    """
    Train on the outer training data and evaluate on the
    untouched outer test data.
    """

    C = candidate["C"]
    gamma = candidate["gamma"]

    model = create_model(
        C,
        gamma,
        numeric_features,
        categorical_features,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)

    outer_auc = roc_auc_score(
        y_test,
        decision_scores,
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    outer_loss = 1.0 - outer_auc

    selection_score = (
        candidate["mean_inner_loss"]
        + LAMBDA_VALUE * candidate["std_inner_loss"]
    )

    return {
        "method": method_name,
        "C": C,
        "gamma": gamma,
        "inner_mean_loss": candidate["mean_inner_loss"],
        "inner_std_loss": candidate["std_inner_loss"],
        "selection_score": selection_score,
        "outer_roc_auc": outer_auc,
        "outer_loss": outer_loss,
        "overtuning_gap": (
            outer_loss - candidate["mean_inner_loss"]
        ),
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
    }


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 60)
    print("NESTED CROSS-VALIDATION EXPERIMENT")
    print("=" * 60)

    dataset_name = "diabetes"

    # IMPORTANT:
    # Your existing data loader returns exactly three values.
    X, y, categorical_features = get_dataset(dataset_name)

    # Derive numerical features from the columns.
    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    print(f"Dataset: {dataset_name}")
    print(f"Records: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Outer folds: {OUTER_FOLDS}")
    print(f"Inner folds: {INNER_FOLDS}")
    print(f"Particles: {N_PARTICLES}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Lambda: {LAMBDA_VALUE}")

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_outer_results = []
    all_search_history = []

    for outer_fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):

        print("\n" + "=" * 60)
        print(
            f"OUTER FOLD {outer_fold}/{OUTER_FOLDS}"
        )
        print("=" * 60)

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        # Step A: Search only within the outer training data.
        candidates, history, evaluation_count = run_inner_pso(
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            outer_fold,
        )

        all_search_history.extend(history)

        # Step B: Choose one candidate for each method.
        selected_candidates = select_candidates(candidates)

        # Step C: Test both choices on the same outer test fold.
        for method_name, candidate in selected_candidates.items():

            print(f"\nMethod: {method_name}")
            print(f"C: {candidate['C']:.6g}")
            print(f"gamma: {candidate['gamma']:.6g}")
            print(
                f"Mean inner loss: "
                f"{candidate['mean_inner_loss']:.4f}"
            )
            print(
                f"Inner loss standard deviation: "
                f"{candidate['std_inner_loss']:.4f}"
            )

            metrics = evaluate_outer_test(
                candidate,
                X_train,
                y_train,
                X_test,
                y_test,
                numeric_features,
                categorical_features,
                method_name,
            )

            result = {
                "dataset": dataset_name,
                "outer_fold": outer_fold,
                "lambda": (
                    LAMBDA_VALUE
                    if method_name == "Uncertainty-aware"
                    else 0.0
                ),
                "unique_candidates": len(candidates),
                "particle_evaluations": evaluation_count,
                **metrics,
            }

            all_outer_results.append(result)

            print(
                f"Outer ROC-AUC: "
                f"{metrics['outer_roc_auc']:.4f}"
            )
            print(
                f"Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(
                f"Sensitivity: {metrics['sensitivity']:.4f}"
            )
            print(
                f"Specificity: {metrics['specificity']:.4f}"
            )
            print(
                f"F1-score: {metrics['f1_score']:.4f}"
            )

    # ========================================================
    # 8. SAVE RESULTS
    # ========================================================

    results_path = (
        RESULTS_DIR / "nested_cv_diabetes_outer_results.csv"
    )

    results_df = pd.DataFrame(all_outer_results)
    results_df.to_csv(results_path, index=False)

    history_path = (
        RESULTS_DIR / "nested_cv_diabetes_search_history.csv"
    )

    history_df = pd.DataFrame(all_search_history)
    history_df.to_csv(history_path, index=False)

    print("\n" + "=" * 60)
    print("NESTED CROSS-VALIDATION COMPLETE")
    print("=" * 60)

    print("\nResults for every outer fold:")
    print(
        results_df[
            [
                "outer_fold",
                "method",
                "outer_roc_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1_score",
            ]
        ].to_string(index=False)
    )

    metric_columns = [
        "outer_roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "overtuning_gap",
    ]

    print("\nMean metrics across outer folds:")
    print(
        results_df.groupby("method")[metric_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nOuter results saved to:\n{results_path}")
    print(f"\nSearch history saved to:\n{history_path}")


if __name__ == "__main__":
    main()

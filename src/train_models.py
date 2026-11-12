from pathlib import Path
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

CUTOFF = 28
RANDOM_STATE = 42
TARGET = "final_result"
POSITIVE_CLASSES = ["Fail", "Withdrawn"]
GROUP = "id_student"
N_SPLITS = 5

COLUMNS_NOT_IN_X = ["id_student", "final_result"]

NUMERIC_COLUMNS = [
    "total_clicks",
    "active_days",
    "max_daily_clicks",
    "clicks_days_0_6",
    "clicks_days_7_13",
    "clicks_days_14_20",
    "clicks_days_21_28",
    "days_since_last_activity",
    "mean_clicks_per_active_day",
    "activity_rate",
    "click_rate_change",
    "assessments_due",
    "assessments_completed",
    "banked_assessments",
    "late_submissions",
    "missed_assessments",
    "early_mean_score",
    "early_score_count",
    "assessment_completion_rate",
    "studied_credits",
    "num_of_prev_attempts",
    "registration_date_missing",
    "no_activity",
    "no_assessments_due",
    "early_score_missing",
]

CATEGORICAL_COLUMNS = [
    "code_module",
    "code_presentation",
    "region",
    "imd_band",
    "age_band",
    "highest_education",
    "disability",
    "gender",
]

MODELS = {
    "logistic_regression": Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            NUMERIC_COLUMNS,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "encode",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=0.01,
                                        ),
                                    ),
                                ]
                            ),
                            CATEGORICAL_COLUMNS,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(strategy="median"),
                            NUMERIC_COLUMNS,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "encode",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=0.01,
                                        ),
                                    ),
                                ]
                            ),
                            CATEGORICAL_COLUMNS,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}


def prepare_data():
    features = pd.read_csv(REPORTS / f"features_day_{CUTOFF}.csv")

    if features[TARGET].isna().any():
        raise ValueError("Missing target values found.")

    missing = [
        column
        for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    X = features.drop(columns=COLUMNS_NOT_IN_X)
    y = features[TARGET].isin(POSITIVE_CLASSES).astype(int)
    groups = features[GROUP]

    return features, X, y, groups


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    features, X, y, groups = prepare_data()

    print(f"Enrolments: {len(features):,}")
    print(f"Unique students: {groups.nunique():,}")
    print("Positive class: Fail or Withdrawn")
    print(f"Positive enrolments: {int(y.sum()):,}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    splitter = GroupKFold(n_splits=N_SPLITS)

    all_results = {}
    all_predictions = []

    for model_name, model in MODELS.items():
        print(f"\nTraining and evaluating: {model_name}")

        probabilities = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=splitter,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        result = evaluate(y, probabilities, threshold=0.5)
        all_results[model_name] = result

        print(json.dumps(result, indent=2))

        predictions = features[
            [
                "code_module",
                "code_presentation",
                "id_student",
                "final_result",
            ]
        ].copy()
        predictions["y_true"] = y
        predictions["y_prob"] = probabilities
        predictions["y_pred"] = (probabilities >= 0.5).astype(int)
        predictions["model"] = model_name
        all_predictions.append(predictions)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(
        REPORTS / f"model_predictions_day_{CUTOFF}.csv",
        index=False,
    )

    with open(
        REPORTS / f"model_results_day_{CUTOFF}.json",
        "w",
    ) as handle:
        json.dump(all_results, handle, indent=2)

    print("\nMODEL COMPARISON")
    comparison = pd.DataFrame(all_results).T
    print(
        comparison[
            [
                "roc_auc",
                "average_precision",
                "brier_score",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .round(3)
        .to_string()
    )

    print("\nSaved predictions and results.")


if __name__ == "__main__":
    main()
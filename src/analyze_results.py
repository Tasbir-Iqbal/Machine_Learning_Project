from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
RISK_TIERS = [0.20, 0.40, 0.60]

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

    X = features.drop(columns=COLUMNS_NOT_IN_X)
    y = features[TARGET].isin(POSITIVE_CLASSES).astype(int)
    groups = features[GROUP]

    return features, X, y, groups


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "flagged_enrolments": int(y_pred.sum()),
        "flagged_percentage": round(y_pred.mean() * 100, 2),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0),
            3,
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0),
            3,
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0),
            3,
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "model": None,
    }


def overall_metrics(y_true, y_prob):
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 3),
        "average_precision": round(
            average_precision_score(y_true, y_prob),
            3,
        ),
        "brier_score": round(
            brier_score_loss(y_true, y_prob),
            3,
        ),
    }


def risk_stratification(features, y_true, y_prob, model_name):
    result = features[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "final_result",
        ]
    ].copy()

    result["y_true"] = y_true
    result["risk_probability"] = y_prob

    result["risk_tier"] = pd.cut(
        result["risk_probability"],
        bins=[
            -np.inf,
            RISK_TIERS[0],
            RISK_TIERS[1],
            RISK_TIERS[2],
            np.inf,
        ],
        labels=["Very low", "Low", "Medium", "High"],
        include_lowest=True,
    )

    result["model"] = model_name

    summary = (
        result.groupby("risk_tier", observed=False)
        .agg(
            enrolments=("y_true", "size"),
            actual_at_risk=("y_true", "sum"),
            average_risk_probability=("risk_probability", "mean"),
        )
        .reset_index()
    )

    summary["at_risk_rate"] = (
        summary["actual_at_risk"] / summary["enrolments"]
    ).round(3)
    summary["model"] = model_name

    return result, summary


def permutation_importance_analysis(
    model_name,
    model,
    X,
    y,
    groups,
):
    print(f"\nCalculating permutation importance: {model_name}")

    splitter = GroupKFold(n_splits=N_SPLITS)

    probabilities = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    fitted_model = model
    fitted_model.fit(X, y)

    importances = permutation_importance(
        fitted_model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importances.importances_mean,
            "importance_std": importances.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    importance["rank"] = range(1, len(importance) + 1)
    importance["model"] = model_name

    return probabilities, importance


def main():
    features, X, y, groups = prepare_data()

    print(f"Enrolments: {len(features):,}")
    print(f"Unique students: {groups.nunique():,}")
    print("Positive class: Fail or Withdrawn")
    print(f"Positive enrolments: {int(y.sum()):,}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    threshold_rows = []
    risk_rows = []
    importance_rows = []
    summary_rows = []

    for model_name, model in MODELS.items():
        print(f"\nAnalyzing: {model_name}")

        probabilities, importance = permutation_importance_analysis(
            model_name,
            model,
            X,
            y,
            groups,
        )

        overall = overall_metrics(y, probabilities)
        overall["model"] = model_name
        summary_rows.append(overall)

        for threshold in THRESHOLDS:
            row = evaluate_at_threshold(y, probabilities, threshold)
            row["model"] = model_name
            threshold_rows.append(row)

        predictions, tier_summary = risk_stratification(
            features,
            y,
            probabilities,
            model_name,
        )

        predictions.to_csv(
            REPORTS / f"predictions_{model_name}_day_{CUTOFF}.csv",
            index=False,
        )

        tier_summary.to_csv(
            REPORTS / f"risk_tiers_{model_name}_day_{CUTOFF}.csv",
            index=False,
        )

        risk_rows.append(tier_summary)
        importance_rows.append(importance)

        print(json.dumps(overall, indent=2))

    threshold_table = pd.DataFrame(threshold_rows)
    risk_table = pd.concat(risk_rows, ignore_index=True)
    importance_table = pd.concat(importance_rows, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)

    threshold_table.to_csv(
        REPORTS / f"threshold_analysis_day_{CUTOFF}.csv",
        index=False,
    )

    risk_table.to_csv(
        REPORTS / f"risk_stratification_day_{CUTOFF}.csv",
        index=False,
    )

    importance_table.to_csv(
        REPORTS / f"permutation_importance_day_{CUTOFF}.csv",
        index=False,
    )

    summary_table.to_csv(
        REPORTS / f"model_summary_day_{CUTOFF}.csv",
        index=False,
    )

    print("\nOVERALL MODEL PERFORMANCE")
    print(
        summary_table[
            [
                "model",
                "roc_auc",
                "average_precision",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    print("\nTHRESHOLD ANALYSIS")
    print(threshold_table.to_string(index=False))

    print("\nRISK STRATIFICATION")
    print(risk_table.to_string(index=False))

    print("\nTOP 15 PERMUTATION IMPORTANCES")

    for model_name in MODELS:
        print(f"\n{model_name}")
        model_importance = importance_table.loc[
            importance_table["model"].eq(model_name)
        ].head(15)

        print(
            model_importance[
                [
                    "rank",
                    "feature",
                    "importance_mean",
                    "importance_std",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    print("\nSaved analysis files:")
    print(f"- threshold_analysis_day_{CUTOFF}.csv")
    print(f"- risk_stratification_day_{CUTOFF}.csv")
    print(f"- permutation_importance_day_{CUTOFF}.csv")
    print(f"- model_summary_day_{CUTOFF}.csv")
    print(f"- predictions_logistic_regression_day_{CUTOFF}.csv")
    print(f"- predictions_random_forest_day_{CUTOFF}.csv")
    print(f"- risk_tiers_logistic_regression_day_{CUTOFF}.csv")
    print(f"- risk_tiers_random_forest_day_{CUTOFF}.csv")


if __name__ == "__main__":
    main()
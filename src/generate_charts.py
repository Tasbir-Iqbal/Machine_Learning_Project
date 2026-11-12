from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"

CUTOFF = 28
SELECTED_THRESHOLD = 0.5
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

KEYS = ["code_module", "code_presentation", "id_student"]

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "k_nearest_neighbors",
    "naive_bayes",
    "gradient_boosting",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "gradient_boosting": "Gradient Boosting",
}

MODEL_COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "k_nearest_neighbors": "#ff7f0e",
    "naive_bayes": "#9467bd",
    "gradient_boosting": "#d62728",
}

METHODS = [
    "K-Means",
    "Agglomerative Hierarchical Clustering",
    "DBSCAN",
    "HDBSCAN",
]

METHOD_LABELS = {
    "K-Means": "K-Means",
    "Agglomerative Hierarchical Clustering": "Agglomerative",
    "DBSCAN": "DBSCAN",
    "HDBSCAN": "HDBSCAN",
}

TIER_LABELS = ["Very low", "Low", "Medium", "High"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 10,
})


def save_chart(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}", flush=True)


def read_csv(name):
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_predictions():
    files = [
        f"predictions_logistic_regression_day_{CUTOFF}.csv",
        f"predictions_random_forest_day_{CUTOFF}.csv",
        f"additional_model_predictions_day_{CUTOFF}.csv",
    ]

    frames = []

    for name in files:
        frame = read_csv(name)

        if "y_prob" not in frame.columns:
            if "risk_probability" not in frame.columns:
                raise ValueError(f"{name}: probability column is missing.")
            frame = frame.rename(
                columns={"risk_probability": "y_prob"}
            )

        required = KEYS + ["model", "y_true", "y_prob"]
        missing = [c for c in required if c not in frame.columns]

        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

        frame = frame[required].copy()

        if frame[required].isna().any().any():
            raise ValueError(
                f"{name}: genuinely missing identifiers, labels, "
                "or probabilities. No rows have been dropped."
            )

        for column in ["y_true", "y_prob"]:
            frame[column] = pd.to_numeric(
                frame[column], errors="raise"
            )

        if not frame["y_true"].isin([0, 1]).all():
            raise ValueError(f"{name}: y_true must contain only 0 and 1.")

        if not frame["y_prob"].between(0, 1).all():
            raise ValueError(f"{name}: probabilities must be in [0, 1].")

        frame["y_true"] = frame["y_true"].astype(int)
        frame["y_pred"] = (
            frame["y_prob"] >= SELECTED_THRESHOLD
        ).astype(int)

        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)

    if predictions.duplicated(["model"] + KEYS).any():
        raise ValueError("Duplicate model-enrolment predictions found.")

    available = set(predictions["model"])
    if available != set(MODEL_ORDER):
        raise ValueError(f"Unexpected model coverage: {available}")

    reference = (
        predictions.loc[
            predictions["model"].eq(MODEL_ORDER[0])
        ]
        .set_index(KEYS)["y_true"]
        .sort_index()
    )

    for model in MODEL_ORDER:
        current = (
            predictions.loc[predictions["model"].eq(model)]
            .set_index(KEYS)["y_true"]
            .sort_index()
        )
        if not current.equals(reference):
            raise ValueError(
                f"{model}: enrolments or labels differ from the "
                "reference model."
            )

    print("\nValidated prediction rows per model:")
    print(predictions.groupby("model").size().to_string())
    print("No prediction rows were dropped.\n")

    return predictions


def build_tables(predictions):
    summary_rows = []
    threshold_rows = []
    tier_rows = []

    for model in MODEL_ORDER:
        frame = predictions.loc[
            predictions["model"].eq(model)
        ].copy()

        y = frame["y_true"].to_numpy()
        probability = frame["y_prob"].to_numpy()
        predicted = (
            probability >= SELECTED_THRESHOLD
        ).astype(int)

        summary_rows.append({
            "model": model,
            "enrolments": len(frame),
            "threshold": SELECTED_THRESHOLD,
            "roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability),
            "brier_score": brier_score_loss(y, probability),
            "precision": precision_score(
                y, predicted, zero_division=0
            ),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1": f1_score(y, predicted, zero_division=0),
        })

        for threshold in THRESHOLDS:
            labels = (probability >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y, labels, labels=[0, 1]
            ).ravel()

            threshold_rows.append({
                "model": model,
                "threshold": threshold,
                "precision": precision_score(
                    y, labels, zero_division=0
                ),
                "recall": recall_score(y, labels, zero_division=0),
                "f1": f1_score(y, labels, zero_division=0),
                "flagged_enrolments": int(labels.sum()),
                "flagged_percentage": labels.mean() * 100,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            })

        frame["risk_tier"] = pd.cut(
            probability,
            bins=[-np.inf, 0.2, 0.4, 0.6, np.inf],
            labels=TIER_LABELS,
            right=True,
        )

        tiers = (
            frame.groupby("risk_tier", observed=False)
            .agg(
                enrolments=("y_true", "size"),
                actual_at_risk=("y_true", "sum"),
                average_model_score=("y_prob", "mean"),
            )
            .reset_index()
        )
        tiers["at_risk_rate"] = (
            tiers["actual_at_risk"]
            / tiers["enrolments"].replace(0, np.nan)
        )
        tiers["model"] = model
        tier_rows.append(tiers)

    summary = pd.DataFrame(summary_rows)
    thresholds = pd.DataFrame(threshold_rows)
    tiers = pd.concat(tier_rows, ignore_index=True)

    summary.to_csv(
        REPORTS / f"all_model_summary_day_{CUTOFF}.csv",
        index=False,
    )
    thresholds.to_csv(
        REPORTS / f"all_model_thresholds_day_{CUTOFF}.csv",
        index=False,
    )
    tiers.to_csv(
        REPORTS / f"all_model_risk_tiers_day_{CUTOFF}.csv",
        index=False,
    )

    return summary, thresholds, tiers


def plot_model_comparison(summary):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    metrics = [
        ("roc_auc", "ROC AUC — higher is better"),
        ("average_precision", "Average precision — higher is better"),
        ("brier_score", "Brier score — lower is better"),
        ("precision", "Precision at threshold 0.5"),
        ("recall", "Recall at threshold 0.5"),
        ("f1", "F1 at threshold 0.5"),
    ]

    ordered = summary.set_index("model").loc[MODEL_ORDER]
    positions = np.arange(len(MODEL_ORDER))
    names = [MODEL_LABELS[m] for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    for axis, (metric, title) in zip(axes.flat, metrics):
        bars = axis.bar(
            positions, ordered[metric], color=colors
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_xticks(positions, names, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Five Classifiers: Out-of-Fold Performance", fontsize=16)
    save_chart(fig, "01_model_performance_comparison")


def plot_thresholds(thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, metric in zip(axes, ["precision", "recall", "f1"]):
        for model in MODEL_ORDER:
            frame = thresholds.loc[
                thresholds["model"].eq(model)
            ]
            axis.plot(
                frame["threshold"],
                frame[metric],
                marker="o",
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )

        axis.axvline(SELECTED_THRESHOLD, color="gray", linestyle="--")
        axis.set_title(metric.upper() if metric == "f1" else metric.title())
        axis.set_xlabel("Score threshold")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)

    axes[-1].legend(fontsize=8)
    fig.suptitle("Threshold Trade-offs: All Five Classifiers", fontsize=16)
    save_chart(fig, "02_threshold_analysis")


def plot_confusions(predictions):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for axis, model in zip(axes.flat, MODEL_ORDER):
        frame = predictions.loc[predictions["model"].eq(model)]
        matrix = confusion_matrix(
            frame["y_true"], frame["y_pred"], labels=[0, 1]
        )

        axis.imshow(matrix, cmap="Blues")
        midpoint = matrix.max() / 2

        for row in range(2):
            for column in range(2):
                axis.text(
                    column, row, f"{matrix[row, column]:,}",
                    ha="center", va="center",
                    color="white" if matrix[row, column] > midpoint else "black",
                    fontsize=13,
                )

        axis.set_xticks([0, 1], ["Pass/Distinction", "Fail/Withdrawn"])
        axis.set_yticks([0, 1], ["Pass/Distinction", "Fail/Withdrawn"])
        axis.set_xlabel("Predicted outcome group")
        axis.set_ylabel("Actual outcome group")
        axis.set_title(MODEL_LABELS[model])

    axes.flat[-1].axis("off")
    fig.suptitle(
        f"Out-of-Fold Confusion Matrices: Threshold {SELECTED_THRESHOLD}",
        fontsize=16,
    )
    save_chart(fig, "03_confusion_matrices")


def plot_risk_tiers(tiers):
    fig, axis = plt.subplots(figsize=(13, 6))
    positions = np.arange(4)
    width = 0.15

    for index, model in enumerate(MODEL_ORDER):
        frame = (
            tiers.loc[tiers["model"].eq(model)]
            .set_index("risk_tier")
            .reindex(TIER_LABELS)
        )
        axis.bar(
            positions + (index - 2) * width,
            frame["at_risk_rate"],
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )

    axis.set_xticks(positions, TIER_LABELS)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel("Observed Fail/Withdrawn rate")
    axis.set_xlabel("Exploratory score tier: ≤0.2, ≤0.4, ≤0.6, >0.6")
    axis.set_title("Observed Outcomes by Model-Score Tier")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    save_chart(fig, "04_risk_stratification")


def plot_importances():
    path = REPORTS / f"permutation_importance_day_{CUTOFF}.csv"

    if not path.exists():
        print("Skipping feature importance: CSV not found.")
        return

    importance = pd.read_csv(path)

    for model in MODEL_ORDER:
        frame = importance.loc[importance["model"].eq(model)]

        if frame.empty:
            print(f"Skipping importance for {model}: not calculated.")
            continue

        frame = (
            frame.sort_values("importance_mean", ascending=False)
            .head(15)
            .sort_values("importance_mean")
        )

        fig, axis = plt.subplots(figsize=(11, 7))
        axis.barh(
            frame["feature"],
            frame["importance_mean"],
            xerr=frame["importance_std"],
            color=MODEL_COLORS[model],
            capsize=2,
        )
        axis.axvline(0, color="gray", linewidth=1)
        axis.set_xlabel("Decrease in average precision after shuffling")
        axis.set_title(
            f"{MODEL_LABELS[model]}: Training-Set Permutation Importance"
        )
        axis.grid(axis="x", alpha=0.2)
        save_chart(fig, f"05_feature_importance_{model}")


def plot_curves(predictions):
    fig_roc, axis_roc = plt.subplots(figsize=(9, 7))
    fig_pr, axis_pr = plt.subplots(figsize=(9, 7))
    fig_cal, axis_cal = plt.subplots(figsize=(9, 7))

    baseline = None

    for model in MODEL_ORDER:
        frame = predictions.loc[predictions["model"].eq(model)]
        y = frame["y_true"]
        probability = frame["y_prob"]
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]

        fpr, tpr, _ = roc_curve(y, probability)
        axis_roc.plot(
            fpr, tpr, color=color,
            label=f"{label} (AUC {roc_auc_score(y, probability):.3f})",
        )

        precision, recall, _ = precision_recall_curve(y, probability)
        axis_pr.plot(
            recall, precision, color=color,
            label=(
                f"{label} "
                f"(AP {average_precision_score(y, probability):.3f})"
            ),
        )

        observed, predicted = calibration_curve(
            y, probability, n_bins=10, strategy="quantile"
        )
        axis_cal.plot(
            predicted, observed, marker="o",
            color=color, label=label,
        )
        baseline = y.mean()

    axis_roc.plot([0, 1], [0, 1], "--", color="gray")
    axis_roc.set(
        title="Out-of-Fold ROC Curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )

    axis_pr.axhline(
        baseline, linestyle="--", color="gray",
        label=f"Prevalence baseline: {baseline:.1%}",
    )
    axis_pr.set(
        title="Out-of-Fold Precision–Recall Curves",
        xlabel="Recall",
        ylabel="Precision",
    )

    axis_cal.plot(
        [0, 1], [0, 1], "--", color="gray",
        label="Perfect calibration",
    )
    axis_cal.set(
        title="Calibration Curves: Ten Quantile Bins",
        xlabel="Mean predicted probability",
        ylabel="Observed Fail/Withdrawn rate",
    )

    for axis in [axis_roc, axis_pr, axis_cal]:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)

    save_chart(fig_roc, "06_roc_curves")
    save_chart(fig_pr, "07_precision_recall_curves")
    save_chart(fig_cal, "12_calibration_curves")


def plot_cluster_metrics(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    columns = [
        ("number_of_clusters", "Number of clusters"),
        ("silhouette_score", "Existing silhouette score*"),
        ("noise_or_unassigned", "Noise / unassigned enrolments"),
    ]
    names = metrics["method"].map(METHOD_LABELS).fillna(metrics["method"])

    for axis, (column, title) in zip(axes, columns):
        bars = axis.bar(names, metrics[column], color="#3783a5")
        axis.bar_label(
            bars,
            fmt="%.3f" if column == "silhouette_score" else "%.0f",
            padding=3,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Clustering Metrics\n"
        "*Density-method scores currently include noise as a label.",
        fontsize=13,
    )
    save_chart(fig, "08_clustering_metrics")


def plot_cluster_outcomes(outcomes, baseline):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    for axis, method in zip(axes.flat, METHODS):
        frame = outcomes.loc[
            outcomes["clustering_method"].eq(method)
        ].sort_values("at_risk_rate", ascending=False)

        labels = [
            "Noise" if value == -1 else str(int(value))
            for value in frame["cluster"]
        ]
        colors = [
            "#999999" if cluster == -1 else "#3783a5"
            for cluster in frame["cluster"]
        ]

        bars = axis.bar(labels, frame["at_risk_rate"], color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
        axis.axhline(
            baseline, color="red", linestyle="--",
            label=f"Cohort baseline: {baseline:.1%}",
        )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Cluster ID (arbitrary within each method)")
        axis.set_ylabel("Observed Fail/Withdrawn rate")
        axis.set_ylim(0, 1.12)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.tick_params(axis="x", rotation=60)
        axis.legend(fontsize=8)

    fig.suptitle("Descriptive Outcomes of Discovered Clusters", fontsize=16)
    save_chart(fig, "09_cluster_outcomes")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    for axis, method in zip(axes.flat, METHODS):
        frame = outcomes.loc[
            outcomes["clustering_method"].eq(method)
        ].sort_values("cluster")
        labels = [
            "Noise" if value == -1 else str(int(value))
            for value in frame["cluster"]
        ]
        axis.bar(labels, frame["enrolments"], color="#3783a5")
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Cluster")
        axis.set_ylabel("Enrolments")
        axis.tick_params(axis="x", rotation=60)

    fig.suptitle("Cluster Sizes and Noise Coverage", fontsize=16)
    save_chart(fig, "11_cluster_sizes")


def plot_kmeans_profiles(profiles):
    columns = [
        "total_clicks",
        "active_days",
        "days_since_last_activity",
        "assessment_completion_rate",
        "early_mean_score",
        "missed_assessments",
        "early_score_missing",
        "no_assessments_due",
    ]

    frame = (
        profiles.loc[
            profiles["clustering_method"].eq("K-Means"),
            ["cluster"] + columns,
        ]
        .set_index("cluster")
        .sort_index()
    )

    ranges = frame.max() - frame.min()
    normalized = (frame - frame.min()) / ranges.replace(0, 1)

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#dddddd")

    fig, axis = plt.subplots(figsize=(14, 6))
    image = axis.imshow(
        np.ma.masked_invalid(normalized.to_numpy()),
        cmap=cmap, vmin=0, vmax=1, aspect="auto",
    )

    for row in range(len(frame)):
        for column in range(len(columns)):
            value = frame.iloc[row, column]
            axis.text(
                column, row,
                "N/A" if pd.isna(value) else f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color=(
                    "white"
                    if pd.notna(normalized.iloc[row, column])
                    and normalized.iloc[row, column] > 0.6
                    else "black"
                ),
            )

    axis.set_xticks(
        np.arange(len(columns)), columns, rotation=35, ha="right"
    )
    axis.set_yticks(
        np.arange(len(frame)),
        [f"Cluster {int(value)}" for value in frame.index],
    )
    axis.set_title(
        "K-Means Profiles: Cells Show Original Feature Means\n"
        "Colors Scale Each Feature Across Clusters; Gray = Unavailable"
    )
    fig.colorbar(image, ax=axis, label="Within-feature relative value")
    save_chart(fig, "10_kmeans_cluster_profiles")


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    summary, thresholds, tiers = build_tables(predictions)

    metrics = read_csv(f"cluster_metrics_day_{CUTOFF}.csv")
    outcomes = read_csv(f"cluster_outcomes_day_{CUTOFF}.csv")
    profiles = read_csv(f"cluster_profiles_day_{CUTOFF}.csv")

    baseline = predictions.loc[
        predictions["model"].eq(MODEL_ORDER[0]), "y_true"
    ].mean()

    plot_model_comparison(summary)
    plot_thresholds(thresholds)
    plot_confusions(predictions)
    plot_risk_tiers(tiers)
    plot_importances()
    plot_curves(predictions)
    plot_cluster_metrics(metrics)
    plot_cluster_outcomes(outcomes, baseline)
    plot_kmeans_profiles(profiles)

    notes = (
        "Charts use saved out-of-fold classifier predictions.\n"
        "No predictions were dropped or probabilities imputed.\n"
        "Threshold and score-tier analyses are exploratory, not "
        "independent test-set estimates after threshold selection.\n"
        "Score tiers are not guarantees of calibrated risk.\n"
        "Permutation importance was calculated on training data "
        "by analyze_results.py; it is not held-out importance.\n"
        "Importance is currently available only for Logistic "
        "Regression and Random Forest.\n"
        "Existing DBSCAN/HDBSCAN silhouette scores include noise "
        "as a label; interpret method comparisons cautiously.\n"
        "Cluster outcomes are descriptive, not validated predictions.\n"
        "Blank profile cells mean unavailable measurements, not zero.\n"
    )
    (CHARTS / "chart_notes.txt").write_text(notes, encoding="utf-8")

    print("\nALL FIVE CLASSIFIERS")
    print(
        summary[
            [
                "model", "roc_auc", "average_precision",
                "brier_score", "precision", "recall", "f1",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\nCharts saved to: {CHARTS}")
    print("Combined summary, thresholds, and risk tiers saved to reports/.")
    print("All charts generated successfully.")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

CUTOFF = 28
RANDOM_STATE = 42
TARGET = "final_result"
POSITIVE_CLASSES = ["Fail", "Withdrawn"]

CLUSTER_FEATURES = [
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

KMEANS_CLUSTERS = 4
AGGLOMERATIVE_CLUSTERS = 4
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 20

HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 20


def prepare_data():
    features = pd.read_csv(REPORTS / f"features_day_{CUTOFF}.csv")

    missing = [
        column
        for column in CLUSTER_FEATURES
        if column not in features.columns
    ]

    if missing:
        raise ValueError(f"Missing expected clustering columns: {missing}")

    X = features[CLUSTER_FEATURES].copy()
    y = features[TARGET].isin(POSITIVE_CLASSES).astype(int)

    return features, X, y


def scale_features(X):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler


def calculate_silhouette(X_scaled, labels):
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) == 2 and -1 in unique_labels:
        return np.nan

    return silhouette_score(
        X_scaled,
        labels,
        metric="euclidean",
        sample_size=min(5000, len(X_scaled)),
        random_state=RANDOM_STATE,
    )


def summarize_clusters(features, y, labels, method_name):
    result = features[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "final_result",
        ]
    ].copy()

    result["y_true"] = y
    result["cluster"] = labels
    result["clustering_method"] = method_name

    summary = (
        result.groupby("cluster")
        .agg(
            enrolments=("y_true", "size"),
            actual_at_risk=("y_true", "sum"),
        )
        .reset_index()
    )

    summary["at_risk_rate"] = (
        summary["actual_at_risk"] / summary["enrolments"]
    ).round(3)

    summary["cluster_percentage"] = (
        summary["enrolments"] / len(result) * 100
    ).round(2)

    summary["clustering_method"] = method_name

    cluster_profiles = (
        features[CLUSTER_FEATURES]
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
        .round(2)
        .reset_index()
    )

    cluster_profiles["clustering_method"] = method_name

    return result, summary, cluster_profiles


def run_kmeans(X_scaled, features, y):
    print("\nRunning K-Means...")

    model = KMeans(
        n_clusters=KMEANS_CLUSTERS,
        n_init=10,
        random_state=RANDOM_STATE,
    )

    labels = model.fit_predict(X_scaled)
    silhouette = calculate_silhouette(X_scaled, labels)

    predictions, summary, profiles = summarize_clusters(
        features,
        y,
        labels,
        "K-Means",
    )

    metrics = {
        "method": "K-Means",
        "number_of_clusters": int(len(np.unique(labels))),
        "silhouette_score": (
            round(silhouette, 3)
            if not np.isnan(silhouette)
            else None
        ),
        "noise_or_unassigned": int((labels == -1).sum()),
    }

    return predictions, summary, profiles, metrics


def run_agglomerative(X_scaled, features, y):
    print("\nRunning Agglomerative Hierarchical Clustering...")

    model = AgglomerativeClustering(
        n_clusters=AGGLOMERATIVE_CLUSTERS,
        linkage="ward",
    )

    labels = model.fit_predict(X_scaled)
    silhouette = calculate_silhouette(X_scaled, labels)

    predictions, summary, profiles = summarize_clusters(
        features,
        y,
        labels,
        "Agglomerative Hierarchical Clustering",
    )

    metrics = {
        "method": "Agglomerative Hierarchical Clustering",
        "number_of_clusters": int(len(np.unique(labels))),
        "silhouette_score": (
            round(silhouette, 3)
            if not np.isnan(silhouette)
            else None
        ),
        "noise_or_unassigned": int((labels == -1).sum()),
    }

    return predictions, summary, profiles, metrics


def run_dbscan(X_scaled, features, y):
    print("\nRunning DBSCAN...")

    model = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="euclidean",
        n_jobs=-1,
    )

    labels = model.fit_predict(X_scaled)
    silhouette = calculate_silhouette(X_scaled, labels)

    predictions, summary, profiles = summarize_clusters(
        features,
        y,
        labels,
        "DBSCAN",
    )

    metrics = {
        "method": "DBSCAN",
        "number_of_clusters": int(
            len(np.unique(labels[labels != -1]))
        ),
        "silhouette_score": (
            round(silhouette, 3)
            if not np.isnan(silhouette)
            else None
        ),
        "noise_or_unassigned": int((labels == -1).sum()),
    }

    return predictions, summary, profiles, metrics


def run_hdbscan(X_scaled, features, y):
    if not HDBSCAN_AVAILABLE:
        print("\nHDBSCAN is not installed. Skipping HDBSCAN.")
        return None, None, None, None

    print("\nRunning HDBSCAN...")

    model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = model.fit_predict(X_scaled)
    silhouette = calculate_silhouette(X_scaled, labels)

    predictions, summary, profiles = summarize_clusters(
        features,
        y,
        labels,
        "HDBSCAN",
    )

    metrics = {
        "method": "HDBSCAN",
        "number_of_clusters": int(
            len(np.unique(labels[labels != -1]))
        ),
        "silhouette_score": (
            round(silhouette, 3)
            if not np.isnan(silhouette)
            else None
        ),
        "noise_or_unassigned": int((labels == -1).sum()),
    }

    return predictions, summary, profiles, metrics


def main():
    features, X, y = prepare_data()

    print(f"Enrolments: {len(features):,}")
    print(f"Unique students: {features['id_student'].nunique():,}")
    print(f"Positive enrolments: {int(y.sum()):,}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    X_scaled, scaler = scale_features(X)

    all_predictions = []
    all_summaries = []
    all_profiles = []
    all_metrics = []

    kmeans_predictions, kmeans_summary, kmeans_profiles, kmeans_metrics = (
        run_kmeans(X_scaled, features, y)
    )

    if kmeans_predictions is not None:
        all_predictions.append(kmeans_predictions)
        all_summaries.append(kmeans_summary)
        all_profiles.append(kmeans_profiles)
        all_metrics.append(kmeans_metrics)

    agglomerative_predictions, agglomerative_summary, agglomerative_profiles, agglomerative_metrics = (
        run_agglomerative(X_scaled, features, y)
    )

    if agglomerative_predictions is not None:
        all_predictions.append(agglomerative_predictions)
        all_summaries.append(agglomerative_summary)
        all_profiles.append(agglomerative_profiles)
        all_metrics.append(agglomerative_metrics)

    dbscan_predictions, dbscan_summary, dbscan_profiles, dbscan_metrics = (
        run_dbscan(X_scaled, features, y)
    )

    if dbscan_predictions is not None:
        all_predictions.append(dbscan_predictions)
        all_summaries.append(dbscan_summary)
        all_profiles.append(dbscan_profiles)
        all_metrics.append(dbscan_metrics)

    hdbscan_predictions, hdbscan_summary, hdbscan_profiles, hdbscan_metrics = (
        run_hdbscan(X_scaled, features, y)
    )

    if hdbscan_predictions is not None:
        all_predictions.append(hdbscan_predictions)
        all_summaries.append(hdbscan_summary)
        all_profiles.append(hdbscan_profiles)
        all_metrics.append(hdbscan_metrics)

    predictions = pd.concat(all_predictions, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    profiles = pd.concat(all_profiles, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)

    predictions.to_csv(
        REPORTS / f"cluster_assignments_day_{CUTOFF}.csv",
        index=False,
    )

    summaries.to_csv(
        REPORTS / f"cluster_outcomes_day_{CUTOFF}.csv",
        index=False,
    )

    profiles.to_csv(
        REPORTS / f"cluster_profiles_day_{CUTOFF}.csv",
        index=False,
    )

    metrics.to_csv(
        REPORTS / f"cluster_metrics_day_{CUTOFF}.csv",
        index=False,
    )

    print("\nCLUSTERING METRICS")
    print(metrics.to_string(index=False))

    print("\nCLUSTER OUTCOMES")
    print(summaries.to_string(index=False))

    print("\nCLUSTER PROFILES")
    print(profiles.to_string(index=False))

    print("\nSaved clustering outputs:")
    print(f"- cluster_assignments_day_{CUTOFF}.csv")
    print(f"- cluster_outcomes_day_{CUTOFF}.csv")
    print(f"- cluster_profiles_day_{CUTOFF}.csv")
    print(f"- cluster_metrics_day_{CUTOFF}.csv")


if __name__ == "__main__":
    main()
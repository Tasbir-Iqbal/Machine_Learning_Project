from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFF = 28


def main():
    cohort_path = REPORTS / f"cohort_day_{CUTOFF}.csv"

    if not cohort_path.exists():
        raise FileNotFoundError("Run feature_engineering.py first.")

    cohort = pd.read_csv(cohort_path)
    parts = []

    print("Reading early activity in chunks...", flush=True)

    for number, chunk in enumerate(
        pd.read_csv(DATA / "studentVle.csv", chunksize=500_000),
        start=1,
    ):
        early = chunk.loc[
            chunk["date"].between(0, CUTOFF)
        ]

        early = early.merge(
            cohort[KEYS],
            on=KEYS,
            how="inner",
            validate="many_to_one",
        )

        if not early.empty:
            daily = (
                early.groupby(KEYS + ["date"], as_index=False)["sum_click"]
                .sum()
            )
            parts.append(daily)

        print(f"Processed chunk {number}", flush=True)

    if not parts:
        raise ValueError("No eligible activity found in the selected window.")

    daily = (
        pd.concat(parts, ignore_index=True)
        .groupby(KEYS + ["date"], as_index=False)["sum_click"]
        .sum()
    )

    if daily["sum_click"].lt(0).any():
        raise ValueError("Negative click counts found.")

    daily = daily.loc[daily["sum_click"].gt(0)].copy()

    activity = (
        daily.groupby(KEYS)
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            last_activity_day=("date", "max"),
            max_daily_clicks=("sum_click", "max"),
        )
        .reset_index()
    )

    daily["window"] = np.select(
        [
            daily["date"].le(6),
            daily["date"].le(13),
            daily["date"].le(20),
        ],
        [1, 2, 3],
        default=4,
    )

    weekly = (
        daily.groupby(KEYS + ["window"])["sum_click"]
        .sum()
        .unstack("window", fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
        .rename(columns={
            1: "clicks_days_0_6",
            2: "clicks_days_7_13",
            3: "clicks_days_14_20",
            4: "clicks_days_21_28",
        })
        .reset_index()
    )

    features = cohort.merge(
        activity, on=KEYS, how="left", validate="one_to_one"
    ).merge(
        weekly, on=KEYS, how="left", validate="one_to_one"
    )

    zero_columns = [
        "total_clicks",
        "active_days",
        "max_daily_clicks",
        "clicks_days_0_6",
        "clicks_days_7_13",
        "clicks_days_14_20",
        "clicks_days_21_28",
    ]

    features[zero_columns] = features[zero_columns].fillna(0)

    features["no_activity"] = features["active_days"].eq(0).astype(int)

    features["days_since_last_activity"] = (
        CUTOFF - features["last_activity_day"]
    ).fillna(CUTOFF + 1)

    features["mean_clicks_per_active_day"] = (
        features["total_clicks"]
        / features["active_days"].replace(0, np.nan)
    ).fillna(0)

    features["activity_rate"] = features["active_days"] / (CUTOFF + 1)

    features["click_rate_change"] = (
        features["clicks_days_21_28"] / 8
        - features["clicks_days_0_6"] / 7
    )

    features = features.drop(columns="last_activity_day")

    if len(features) != len(cohort):
        raise ValueError("Enrolment count changed during merging.")

    if features.duplicated(KEYS).any():
        raise ValueError("Duplicate enrolment keys found.")

    output = REPORTS / f"activity_features_day_{CUTOFF}.csv"
    features.to_csv(output, index=False)

    print("\nACTIVITY FEATURE SUMMARY")
    print(f"Enrolments: {len(features):,}")
    print(f"Columns: {len(features.columns)}")
    print(f"No recorded early activity: {features['no_activity'].sum():,}")

    print("\nClick statistics:")
    print(features["total_clicks"].describe().round(2).to_string())

    print("\nMissing activity-feature values:")
    activity_columns = zero_columns + [
        "no_activity",
        "days_since_last_activity",
        "mean_clicks_per_active_day",
        "activity_rate",
        "click_rate_change",
    ]
    print(features[activity_columns].isna().sum().to_string())

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
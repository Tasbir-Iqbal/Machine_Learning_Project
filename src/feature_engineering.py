from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFF = 28


def build_cohort(cutoff=CUTOFF):
    students = pd.read_csv(DATA / "studentInfo.csv")
    registrations = pd.read_csv(DATA / "studentRegistration.csv")
    courses = pd.read_csv(DATA / "courses.csv")

    if students.duplicated(KEYS).any():
        raise ValueError("Duplicate student-course keys in studentInfo.csv")

    if registrations.duplicated(KEYS).any():
        raise ValueError("Duplicate student-course keys in studentRegistration.csv")

    cohort = students.merge(
        registrations,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not cohort["_merge"].eq("both").all():
        raise ValueError("Some enrolments have no matching registration record")

    cohort = cohort.drop(columns="_merge")

    withdrawn = (
        cohort["date_unregistration"].notna()
        & cohort["date_unregistration"].le(cutoff)
    )

    registered_later = cohort["date_registration"].gt(cutoff)

    excluded = withdrawn | registered_later

    print(f"Original enrolments: {len(cohort):,}")
    print(f"Withdrawn on/before day {cutoff}: {withdrawn.sum():,}")
    print(f"Registered after day {cutoff}: {registered_later.sum():,}")
    print(f"Overlap between exclusion groups: {(withdrawn & registered_later).sum():,}")
    print(f"Total excluded enrolments: {excluded.sum():,}")

    cohort = cohort.loc[~excluded].copy()

    cohort["registration_date_missing"] = (
        cohort["date_registration"].isna().astype(int)
    )

    cohort = cohort.drop(columns="date_unregistration")

    cohort = cohort.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )

    if cohort["module_presentation_length"].isna().any():
        raise ValueError("Some enrolments have no matching course record")

    cohort["cutoff_day"] = cutoff

    return cohort


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort()

    print(f"\nEligible enrolments: {len(cohort):,}")
    print(f"Unique students: {cohort['id_student'].nunique():,}")

    print("\nELIGIBLE COHORT OUTCOMES")
    counts = cohort["final_result"].value_counts()
    distribution = counts.rename("count").to_frame()
    distribution["percentage"] = (
        distribution["count"] / len(cohort) * 100
    ).round(2)
    print(distribution.to_string())

    output = REPORTS / f"cohort_day_{CUTOFF}.csv"
    cohort.to_csv(output, index=False)

    distribution.to_csv(
        REPORTS / f"cohort_day_{CUTOFF}_distribution.csv"
    )

    print(f"\nSaved cohort: {output}")
    print("Withdrawal date removed from the saved cohort.")


if __name__ == "__main__":
    main()
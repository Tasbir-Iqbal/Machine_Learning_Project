from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

KEYS = ["code_module", "code_presentation", "id_student"]
COURSE_KEYS = ["code_module", "code_presentation"]
CUTOFF = 28


def main():
    features = pd.read_csv(
        REPORTS / f"activity_features_day_{CUTOFF}.csv"
    )
    assessments = pd.read_csv(DATA / "assessments.csv")
    submissions = pd.read_csv(DATA / "studentAssessment.csv")

    if assessments["id_assessment"].duplicated().any():
        raise ValueError("Duplicate assessment IDs found.")

    if submissions.duplicated(["id_student", "id_assessment"]).any():
        raise ValueError("Duplicate student-assessment records found.")

    due = assessments.loc[
        assessments["date"].between(0, CUTOFF)
        & assessments["assessment_type"].ne("Exam"),
        COURSE_KEYS + ["id_assessment", "date"],
    ].copy()

    expected = features[KEYS].merge(
        due,
        on=COURSE_KEYS,
        how="inner",
        validate="many_to_many",
    )

    expected = expected.merge(
        submissions[
            ["id_student", "id_assessment", "date_submitted", "is_banked", "score"]
        ],
        on=["id_student", "id_assessment"],
        how="left",
        validate="many_to_one",
    )

    expected["banked"] = expected["is_banked"].eq(1)

    expected["submitted_by_cutoff"] = (
        expected["date_submitted"].notna()
        & expected["date_submitted"].le(CUTOFF)
        & ~expected["banked"]
    )

    expected["completed_by_cutoff"] = (
        expected["submitted_by_cutoff"] | expected["banked"]
    )

    expected["late_submission"] = (
        expected["submitted_by_cutoff"]
        & expected["date_submitted"].gt(expected["date"])
    )

    expected["missing_due_assessment"] = (
        ~expected["completed_by_cutoff"]
    )

    expected["usable_score"] = expected["score"].where(
        expected["completed_by_cutoff"]
    )

    aggregated = (
        expected.groupby(KEYS)
        .agg(
            assessments_due=("id_assessment", "size"),
            assessments_completed=("completed_by_cutoff", "sum"),
            banked_assessments=("banked", "sum"),
            late_submissions=("late_submission", "sum"),
            missed_assessments=("missing_due_assessment", "sum"),
            early_mean_score=("usable_score", "mean"),
            early_score_count=("usable_score", "count"),
        )
        .reset_index()
    )

    result = features.merge(
        aggregated,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "assessments_due",
        "assessments_completed",
        "banked_assessments",
        "late_submissions",
        "missed_assessments",
        "early_score_count",
    ]

    result[count_columns] = result[count_columns].fillna(0).astype(int)

    result["no_assessments_due"] = (
        result["assessments_due"].eq(0).astype(int)
    )

    result["early_score_missing"] = (
        result["early_mean_score"].isna().astype(int)
    )

    result["assessment_completion_rate"] = (
        result["assessments_completed"]
        / result["assessments_due"].where(result["assessments_due"].gt(0))
    )

    if len(result) != len(features):
        raise ValueError("Enrolment count changed.")

    if result.duplicated(KEYS).any():
        raise ValueError("Duplicate enrolment keys found.")

    if not (
        result["assessments_completed"] + result["missed_assessments"]
    ).eq(result["assessments_due"]).all():
        raise ValueError("Assessment counts are inconsistent.")

    if result["date_unregistration"].notna().any() if "date_unregistration" in result else False:
        raise ValueError("Withdrawal date should not be present.")

    output = REPORTS / f"features_day_{CUTOFF}.csv"
    result.to_csv(output, index=False)

    print("\nASSESSMENT FEATURE SUMMARY")
    print(f"Enrolments: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print(f"No assessments due: {result['no_assessments_due'].sum():,}")
    print(f"No usable early score: {result['early_score_missing'].sum():,}")
    print(f"Enrolments with missed work: {result['missed_assessments'].gt(0).sum():,}")
    print(f"Banked assessment records counted: {result['banked_assessments'].sum():,}")

    print("\nAssessment statistics:")
    print(
        result[
            [
                "assessments_due",
                "assessments_completed",
                "missed_assessments",
                "early_mean_score",
                "assessment_completion_rate",
            ]
        ].describe().round(2).to_string()
    )

    print("\nMissing assessment-feature values:")
    print(
        result[
            count_columns
            + [
                "early_mean_score",
                "assessment_completion_rate",
                "no_assessments_due",
                "early_score_missing",
            ]
        ].isna().sum().to_string()
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
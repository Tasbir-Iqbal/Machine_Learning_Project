from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

filenames = [
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
]

missing_files = [name for name in filenames if not (DATA / name).is_file()]

if missing_files:
    raise FileNotFoundError(f"Missing CSV files: {missing_files}")

summary = []

for filename in filenames:
    path = DATA / filename
    rows = 0
    missing = None

    columns = pd.read_csv(path, nrows=0).columns.tolist()

    print(f"\nReading {filename}...", flush=True)

    for chunk in pd.read_csv(path, chunksize=100_000):
        rows += len(chunk)
        counts = chunk.isna().sum()
        missing = counts if missing is None else missing.add(counts, fill_value=0)

    if missing is None:
        missing = pd.Series(0, index=columns, dtype="int64")

    missing = missing.astype("int64")

    print(f"Rows: {rows:,} | Columns: {len(columns)}")
    print("Columns:", ", ".join(columns))
    print("Missing values:")
    print(missing.to_string())

    summary.append({
        "file": filename,
        "rows": rows,
        "columns": len(columns),
        "size_mb": round(path.stat().st_size / (1024 ** 2), 2),
    })

    missing.rename("missing_count").to_csv(
        REPORTS / f"{path.stem}_missing_values.csv"
    )

students = pd.read_csv(DATA / "studentInfo.csv")

distribution = students["final_result"].value_counts().rename("count")
distribution = distribution.to_frame()
distribution["percentage"] = (
    distribution["count"] / len(students) * 100
).round(2)

keys = ["code_module", "code_presentation", "id_student"]

print("\n" + "=" * 60)
print("FINAL RESULT DISTRIBUTION")
print(distribution.to_string())

print("\nDuplicate student-course keys:")
print(students.duplicated(subset=keys).sum())

print("\nUnique student IDs:")
print(students["id_student"].nunique())

print("\nEnrolment rows:")
print(len(students))

pd.DataFrame(summary).to_csv(REPORTS / "dataset_summary.csv", index=False)
distribution.to_csv(REPORTS / "target_distribution.csv")

print("\nFinished. Summary files saved in:", REPORTS)

assessments = pd.read_csv(DATA / "assessments.csv")
registrations = pd.read_csv(DATA / "studentRegistration.csv")

cutoff = 28

early_assessments = assessments[
    assessments["date"].notna()
    & assessments["date"].between(0, cutoff)
    & assessments["assessment_type"].ne("Exam")
]

early_counts = (
    early_assessments
    .groupby(["code_module", "code_presentation"])
    .size()
    .rename("assessments_due_by_day_28")
    .reset_index()
)

courses = pd.read_csv(DATA / "courses.csv")

early_counts = courses[
    ["code_module", "code_presentation"]
].merge(
    early_counts,
    on=["code_module", "code_presentation"],
    how="left",
)

early_counts["assessments_due_by_day_28"] = (
    early_counts["assessments_due_by_day_28"]
    .fillna(0)
    .astype(int)
)

withdrawn_by_cutoff = (
    registrations["date_unregistration"].notna()
    & registrations["date_unregistration"].le(cutoff)
)

print("\nASSESSMENTS DUE BY DAY 28")
print(early_counts.to_string(index=False))

print("\nEnrolments withdrawn on or before day 28:")
print(withdrawn_by_cutoff.sum())

print("\nEnrolments registered after day 28:")
print(registrations["date_registration"].gt(cutoff).sum())

early_counts.to_csv(
    REPORTS / "early_assessment_availability.csv",
    index=False,
)
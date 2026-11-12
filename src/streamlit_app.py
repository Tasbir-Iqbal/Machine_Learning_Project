from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SAVED_MODELS = REPORTS / "saved_models"

N_FOLDS = 5

MODEL_LABELS = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
}

RISK_TIERS = [
    (0.00, 0.30, "Low"),
    (0.30, 0.50, "Moderate"),
    (0.50, 0.70, "High"),
    (0.70, 1.01, "Very High"),
]

st.set_page_config(
    page_title="Student Risk Early-Warning System",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Risk Early-Warning System")
st.write(
    "This tool estimates the probability that a student enrolment will "
    "result in Fail or Withdrawn, using information available by day 28."
)

st.warning(
    "This is a decision-support tool. Risk scores are not certain outcomes "
    "and should support, not replace, academic judgement."
)


@st.cache_resource
def load_available_models():
    available = {}

    for model_name in MODEL_LABELS:
        directory = SAVED_MODELS / model_name
        pipelines = []

        for fold in range(1, N_FOLDS + 1):
            path = directory / f"fold_{fold}_pipeline.joblib"

            if path.exists():
                pipelines.append(joblib.load(path))

        if pipelines:
            available[model_name] = pipelines

    return available


@st.cache_data
def load_metadata():
    path = SAVED_MODELS / "gradient_boosting" / "fold_1_metadata.json"

    if not path.exists():
        raise FileNotFoundError(
            "Gradient Boosting metadata not found. Run save_models.py first."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def assign_risk(probability):
    for lower, upper, label in RISK_TIERS:
        if lower <= probability < upper:
            return label

    return "Very High"


def predict_with_models(models, input_row):
    results = {}

    for model_name, pipelines in models.items():
        probabilities = []

        for pipeline in pipelines:
            probability = pipeline.predict_proba(input_row)[:, 1]
            probabilities.append(float(probability[0]))

        results[model_name] = float(np.mean(probabilities))

    return results


def create_input_frame(values, feature_columns):
    row = {column: np.nan for column in feature_columns}
    row.update(values)

    return pd.DataFrame([row])


def explain_risk(values):
    concerns = []

    if values["missed_assessments"] > 0:
        concerns.append(
            f"{int(values['missed_assessments'])} assessment(s) were missed."
        )

    if values["late_submissions"] > 0:
        concerns.append(
            f"{int(values['late_submissions'])} submission(s) were late."
        )

    if values["early_score_missing"] == 1:
        concerns.append("Early assessment score information is missing.")

    elif values["early_mean_score"] < 50:
        concerns.append(
            f"Early mean score is low: {values['early_mean_score']:.1f}."
        )

    if values["no_activity"] == 1:
        concerns.append("No VLE activity was recorded.")

    elif values["days_since_last_activity"] > 7:
        concerns.append(
            "No recorded activity for "
            f"{int(values['days_since_last_activity'])} days."
        )

    if values["activity_rate"] < 0.5:
        concerns.append(
            f"Activity rate is low: {values['activity_rate']:.2f}."
        )

    if not concerns:
        concerns.append(
            "No major individual risk indicators were identified from the "
            "entered values."
        )

    return concerns


available_models = load_available_models()
metadata = load_metadata()
feature_columns = metadata["feature_columns"]

if not available_models:
    st.error(
        "No saved model pipelines were found. Run save_models.py first."
    )
    st.stop()

saved_names = list(available_models)
saved_labels = [MODEL_LABELS[name] for name in saved_names]

st.sidebar.success(
    f"{len(saved_names)} of 5 model types are available:\n\n"
    + "\n".join(f"- {label}" for label in saved_labels)
)

page = st.sidebar.radio(
    "Choose mode",
    [
        "Single student prediction",
        "Compare all algorithms",
        "Batch CSV prediction",
        "Model evaluation",
        "Student clustering",
    ],
)

st.sidebar.header("Risk tiers")
st.sidebar.markdown(
    """
- **Low:** probability below 0.30  
- **Moderate:** probability 0.30 to below 0.50  
- **High:** probability 0.50 to below 0.70  
- **Very High:** probability 0.70 or above  
"""
)

if page == "Single student prediction":
    st.subheader("Enter student information")

    with st.form("single_student_form"):
        st.markdown("### Course and student background")

        col1, col2, col3 = st.columns(3)

        code_module = col1.selectbox(
            "Course module",
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        )

        code_presentation = col2.selectbox(
            "Course presentation",
            ["2013J", "2014J", "2013B", "2014B"],
        )

        region = col3.selectbox(
            "Region",
            [
                "East Anglian Region",
                "North Western Region",
                "South Eastern Region",
                "West Midlands Region",
                "Yorkshire Region",
                "North Region",
                "South Region",
                "South West Region",
                "East Midlands Region",
                "Wales",
                "Scotland",
                "Ireland",
                "London Region",
            ],
        )

        col1, col2, col3 = st.columns(3)

        imd_band = col1.selectbox(
            "Index of Multiple Deprivation band",
            ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%"],
        )

        age_band = col2.selectbox(
            "Age band",
            ["0-35", "35-55", "55<="],
        )

        highest_education = col3.selectbox(
            "Highest education",
            [
                "A Level or Equivalent",
                "Lower Than A Level",
                "HE Qualification",
                "No Formal Education",
                "Post Graduate Qualification",
            ],
        )

        col1, col2, col3 = st.columns(3)

        disability = col1.selectbox(
            "Disability status",
            ["N", "Y"],
        )

        gender = col2.selectbox(
            "Gender",
            ["M", "F"],
        )

        num_of_prev_attempts = col3.number_input(
            "Previous attempts",
            min_value=0,
            max_value=10,
            value=0,
        )

        col1, col2 = st.columns(2)

        studied_credits = col1.number_input(
            "Studied credits",
            min_value=0,
            max_value=500,
            value=60,
        )

        registration_date_missing = col2.selectbox(
            "Registration date missing",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
        )

        st.markdown("### VLE activity by day 28")

        col1, col2, col3 = st.columns(3)

        total_clicks = col1.number_input(
            "Total clicks",
            min_value=0,
            value=100,
        )

        active_days = col2.number_input(
            "Active days",
            min_value=0,
            max_value=28,
            value=10,
        )

        max_daily_clicks = col3.number_input(
            "Maximum clicks in one day",
            min_value=0,
            value=20,
        )

        col1, col2, col3, col4 = st.columns(4)

        clicks_days_0_6 = col1.number_input(
            "Clicks: days 0-6",
            min_value=0,
            value=20,
        )

        clicks_days_7_13 = col2.number_input(
            "Clicks: days 7-13",
            min_value=0,
            value=20,
        )

        clicks_days_14_20 = col3.number_input(
            "Clicks: days 14-20",
            min_value=0,
            value=20,
        )

        clicks_days_21_28 = col4.number_input(
            "Clicks: days 21-28",
            min_value=0,
            value=20,
        )

        col1, col2, col3 = st.columns(3)

        days_since_last_activity = col1.number_input(
            "Days since last activity",
            min_value=0,
            max_value=28,
            value=2,
        )

        mean_clicks_per_active_day = col2.number_input(
            "Mean clicks per active day",
            min_value=0.0,
            value=10.0,
        )

        activity_rate = col3.number_input(
            "Activity rate",
            min_value=0.0,
            max_value=1.0,
            value=0.36,
        )

        col1, col2 = st.columns(2)

        click_rate_change = col2.number_input(
            "Click-rate change",
            value=0.0,
        )

        no_activity = col1.selectbox(
            "No activity",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
        )

        st.markdown("### Assessment status")

        col1, col2, col3 = st.columns(3)

        assessments_due = col1.number_input(
            "Assessments due",
            min_value=0,
            value=2,
        )

        assessments_completed = col2.number_input(
            "Assessments completed",
            min_value=0,
            value=1,
        )

        banked_assessments = col3.number_input(
            "Banked assessments",
            min_value=0,
            value=0,
        )

        col1, col2, col3 = st.columns(3)

        late_submissions = col1.number_input(
            "Late submissions",
            min_value=0,
            value=0,
        )

        missed_assessments = col2.number_input(
            "Missed assessments",
            min_value=0,
            value=0,
        )

        early_mean_score = col3.number_input(
            "Early mean score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
        )

        col1, col2, col3 = st.columns(3)

        early_score_count = col1.number_input(
            "Early score count",
            min_value=0,
            value=1,
        )

        assessment_completion_rate = col2.number_input(
            "Assessment completion rate",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
        )

        early_score_missing = col3.selectbox(
            "Early score missing",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
        )

        selected_model = st.selectbox(
            "Prediction algorithm",
            saved_labels,
        )

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        selected_name = [
            name
            for name, label in MODEL_LABELS.items()
            if label == selected_model and name in available_models
        ][0]

        values = {
            "total_clicks": total_clicks,
            "active_days": active_days,
            "max_daily_clicks": max_daily_clicks,
            "clicks_days_0_6": clicks_days_0_6,
            "clicks_days_7_13": clicks_days_7_13,
            "clicks_days_14_20": clicks_days_14_20,
            "clicks_days_21_28": clicks_days_21_28,
            "days_since_last_activity": days_since_last_activity,
            "mean_clicks_per_active_day": mean_clicks_per_active_day,
            "activity_rate": activity_rate,
            "click_rate_change": click_rate_change,
            "assessments_due": assessments_due,
            "assessments_completed": assessments_completed,
            "banked_assessments": banked_assessments,
            "late_submissions": late_submissions,
            "missed_assessments": missed_assessments,
            "early_mean_score": early_mean_score,
            "early_score_count": early_score_count,
            "assessment_completion_rate": assessment_completion_rate,
            "studied_credits": studied_credits,
            "num_of_prev_attempts": num_of_prev_attempts,
            "registration_date_missing": registration_date_missing,
            "no_activity": no_activity,
            "no_assessments_due": 1 if assessments_due == 0 else 0,
            "early_score_missing": early_score_missing,
            "code_module": code_module,
            "code_presentation": code_presentation,
            "region": region,
            "imd_band": imd_band,
            "age_band": age_band,
            "highest_education": highest_education,
            "disability": disability,
            "gender": gender,
        }

        input_frame = create_input_frame(values, feature_columns)
        results = predict_with_models(
            {selected_name: available_models[selected_name]},
            input_frame,
        )

        probability = results[selected_name]
        risk_tier = assign_risk(probability)
        prediction = 1 if probability >= 0.5 else 0

        st.subheader("Prediction result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "At-risk probability",
            f"{probability:.3f}",
        )

        col2.metric(
            "Risk tier",
            risk_tier,
        )

        col3.metric(
            "Prediction at 0.5 threshold",
            "At risk" if prediction == 1 else "Not at risk",
        )

        st.progress(min(probability, 1.0))

        st.subheader("Key risk indicators")
        for concern in explain_risk(values):
            st.write(f"- {concern}")

        st.caption(
            "This result is a model-based risk estimate, not a certain "
            "prediction of the student's final outcome."
        )

elif page == "Compare all algorithms":
    st.subheader("Compare algorithms for one student")

    st.info(
        "Enter the student's information once. The app will generate a "
        "prediction using every available saved model."
    )

    with st.form("comparison_form"):
        col1, col2, col3 = st.columns(3)

        code_module = col1.selectbox(
            "Course module",
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
            key="compare_module",
        )

        code_presentation = col2.selectbox(
            "Course presentation",
            ["2013J", "2014J", "2013B", "2014B"],
            key="compare_presentation",
        )

        region = col3.selectbox(
            "Region",
            [
                "East Anglian Region",
                "North Western Region",
                "South Eastern Region",
                "West Midlands Region",
                "Yorkshire Region",
                "North Region",
                "South Region",
                "South West Region",
                "East Midlands Region",
                "Wales",
                "Scotland",
                "Ireland",
                "London Region",
            ],
            key="compare_region",
        )

        col1, col2, col3 = st.columns(3)

        imd_band = col1.selectbox(
            "Index of Multiple Deprivation band",
            ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%"],
            key="compare_imd",
        )

        age_band = col2.selectbox(
            "Age band",
            ["0-35", "35-55", "55<="],
            key="compare_age",
        )

        highest_education = col3.selectbox(
            "Highest education",
            [
                "A Level or Equivalent",
                "Lower Than A Level",
                "HE Qualification",
                "No Formal Education",
                "Post Graduate Qualification",
            ],
            key="compare_education",
        )

        col1, col2, col3 = st.columns(3)

        disability = col1.selectbox(
            "Disability status",
            ["N", "Y"],
            key="compare_disability",
        )

        gender = col2.selectbox(
            "Gender",
            ["M", "F"],
            key="compare_gender",
        )

        num_of_prev_attempts = col3.number_input(
            "Previous attempts",
            min_value=0,
            max_value=10,
            value=0,
            key="compare_attempts",
        )

        col1, col2 = st.columns(2)

        studied_credits = col1.number_input(
            "Studied credits",
            min_value=0,
            max_value=500,
            value=60,
            key="compare_credits",
        )

        registration_date_missing = col2.selectbox(
            "Registration date missing",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            key="compare_registration",
        )

        col1, col2, col3 = st.columns(3)

        total_clicks = col1.number_input(
            "Total clicks",
            min_value=0,
            value=100,
            key="compare_total_clicks",
        )

        active_days = col2.number_input(
            "Active days",
            min_value=0,
            max_value=28,
            value=10,
            key="compare_active_days",
        )

        max_daily_clicks = col3.number_input(
            "Maximum clicks in one day",
            min_value=0,
            value=20,
            key="compare_max_clicks",
        )

        col1, col2, col3, col4 = st.columns(4)

        clicks_days_0_6 = col1.number_input(
            "Clicks: days 0-6",
            min_value=0,
            value=20,
            key="compare_clicks_0_6",
        )

        clicks_days_7_13 = col2.number_input(
            "Clicks: days 7-13",
            min_value=0,
            value=20,
            key="compare_clicks_7_13",
        )

        clicks_days_14_20 = col3.number_input(
            "Clicks: days 14-20",
            min_value=0,
            value=20,
            key="compare_clicks_14_20",
        )

        clicks_days_21_28 = col4.number_input(
            "Clicks: days 21-28",
            min_value=0,
            value=20,
            key="compare_clicks_21_28",
        )

        col1, col2, col3 = st.columns(3)

        days_since_last_activity = col1.number_input(
            "Days since last activity",
            min_value=0,
            max_value=28,
            value=2,
            key="compare_last_activity",
        )

        mean_clicks_per_active_day = col2.number_input(
            "Mean clicks per active day",
            min_value=0.0,
            value=10.0,
            key="compare_mean_clicks",
        )

        activity_rate = col3.number_input(
            "Activity rate",
            min_value=0.0,
            max_value=1.0,
            value=0.36,
            key="compare_activity_rate",
        )

        col1, col2 = st.columns(2)

        click_rate_change = col2.number_input(
            "Click-rate change",
            value=0.0,
            key="compare_click_change",
        )

        no_activity = col1.selectbox(
            "No activity",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            key="compare_no_activity",
        )

        col1, col2, col3 = st.columns(3)

        assessments_due = col1.number_input(
            "Assessments due",
            min_value=0,
            value=2,
            key="compare_assessments_due",
        )

        assessments_completed = col2.number_input(
            "Assessments completed",
            min_value=0,
            value=1,
            key="compare_assessments_completed",
        )

        banked_assessments = col3.number_input(
            "Banked assessments",
            min_value=0,
            value=0,
            key="compare_banked",
        )

        col1, col2, col3 = st.columns(3)

        late_submissions = col1.number_input(
            "Late submissions",
            min_value=0,
            value=0,
            key="compare_late",
        )

        missed_assessments = col2.number_input(
            "Missed assessments",
            min_value=0,
            value=0,
            key="compare_missed",
        )

        early_mean_score = col3.number_input(
            "Early mean score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            key="compare_early_score",
        )

        col1, col2, col3 = st.columns(3)

        early_score_count = col1.number_input(
            "Early score count",
            min_value=0,
            value=1,
            key="compare_score_count",
        )

        assessment_completion_rate = col2.number_input(
            "Assessment completion rate",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key="compare_completion_rate",
        )

        early_score_missing = col3.selectbox(
            "Early score missing",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            key="compare_score_missing",
        )

        submitted = st.form_submit_button("Compare all algorithms")

    if submitted:
        values = {
            "total_clicks": total_clicks,
            "active_days": active_days,
            "max_daily_clicks": max_daily_clicks,
            "clicks_days_0_6": clicks_days_0_6,
            "clicks_days_7_13": clicks_days_7_13,
            "clicks_days_14_20": clicks_days_14_20,
            "clicks_days_21_28": clicks_days_21_28,
            "days_since_last_activity": days_since_last_activity,
            "mean_clicks_per_active_day": mean_clicks_per_active_day,
            "activity_rate": activity_rate,
            "click_rate_change": click_rate_change,
            "assessments_due": assessments_due,
            "assessments_completed": assessments_completed,
            "banked_assessments": banked_assessments,
            "late_submissions": late_submissions,
            "missed_assessments": missed_assessments,
            "early_mean_score": early_mean_score,
            "early_score_count": early_score_count,
            "assessment_completion_rate": assessment_completion_rate,
            "studied_credits": studied_credits,
            "num_of_prev_attempts": num_of_prev_attempts,
            "registration_date_missing": registration_date_missing,
            "no_activity": no_activity,
            "no_assessments_due": 1 if assessments_due == 0 else 0,
            "early_score_missing": early_score_missing,
            "code_module": code_module,
            "code_presentation": code_presentation,
            "region": region,
            "imd_band": imd_band,
            "age_band": age_band,
            "highest_education": highest_education,
            "disability": disability,
            "gender": gender,
        }

        input_frame = create_input_frame(values, feature_columns)
        results = predict_with_models(available_models, input_frame)

        comparison = pd.DataFrame(
            [
                {
                    "Algorithm": MODEL_LABELS[model_name],
                    "At-risk probability": probability,
                    "Risk tier": assign_risk(probability),
                    "Prediction at 0.5": (
                        "At risk" if probability >= 0.5 else "Not at risk"
                    ),
                }
                for model_name, probability in results.items()
            ]
        ).sort_values(
            "At-risk probability",
            ascending=False,
        )

        st.subheader("Algorithm comparison")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            comparison.set_index("Algorithm")["At-risk probability"]
        )

        st.subheader("Key risk indicators")
        for concern in explain_risk(values):
            st.write(f"- {concern}")

        st.caption(
            "Different algorithms may produce different risk estimates for "
            "the same student. The validated out-of-fold results should "
            "guide interpretation of overall model performance."
        )

elif page == "Batch CSV prediction":
    st.subheader("Batch CSV prediction")

    uploaded_file = st.file_uploader(
        "Upload a day-28 feature CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info(
            "Upload a CSV containing the required day-28 feature columns."
        )
        st.stop()

    data = pd.read_csv(uploaded_file)
    missing = [
        column for column in feature_columns if column not in data.columns
    ]

    if missing:
        st.error(
            "The uploaded file is missing required columns:\n\n"
            + "\n".join(f"- {column}" for column in missing)
        )
        st.stop()

    X = data[feature_columns].copy()

    if st.button("Generate risk predictions"):
        with st.spinner(
            "Generating predictions using available model pipelines..."
        ):
            probabilities = []

            for pipeline in available_models["gradient_boosting"]:
                probability = pipeline.predict_proba(X)[:, 1]
                probabilities.append(probability)

            probability = np.mean(probabilities, axis=0)
            prediction = (probability >= 0.5).astype(int)
            risk_tier = [assign_risk(value) for value in probability]

        output = data.copy()
        output["at_risk_probability"] = probability
        output["at_risk_prediction"] = prediction
        output["risk_tier"] = risk_tier

        st.success("Predictions generated successfully.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Enrolments assessed", f"{len(output):,}")
        col2.metric("Mean risk probability", f"{probability.mean():.3f}")
        col3.metric(
            "High or Very High",
            f"{(output['risk_tier'].isin(['High', 'Very High'])).sum():,}",
        )
        col4.metric("Predicted at risk", f"{prediction.sum():,}")

        st.subheader("Risk-tier distribution")
        st.bar_chart(output["risk_tier"].value_counts())

        st.subheader("Prediction results")
        st.dataframe(
            output[
                [
                    "code_module",
                    "code_presentation",
                    "id_student",
                    "at_risk_probability",
                    "at_risk_prediction",
                    "risk_tier",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="student_risk_predictions.csv",
            mime="text/csv",
        )

elif page == "Model evaluation":
    st.subheader("Model evaluation")

    st.info(
        "These results come from five-fold cross-validation grouped by "
        "student. Each model was evaluated only on validation enrolments "
        "from students excluded from its training folds."
    )

    charts_dir = REPORTS / "charts"
    importance_dir = REPORTS / "validation_importance" / "charts"
    comparison_dir = REPORTS / "feature_group_comparison" / "charts"

    summary_path = REPORTS / "model_performance_summary.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)

        st.subheader("Classifier performance")
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Model performance summary CSV was not found. "
            "Expected: reports/model_performance_summary.csv"
        )

    st.subheader("ROC curves")
    roc_path = charts_dir / "06_roc_curves.png"

    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.warning("ROC curve chart was not found.")

    st.subheader("Precision-recall curves")
    pr_path = charts_dir / "07_precision_recall_curves.png"

    if pr_path.exists():
        st.image(str(pr_path), use_container_width=True)
    else:
        st.warning("Precision-recall curve chart was not found.")

    st.subheader("Calibration")
    calibration_path = charts_dir / "12_calibration_curves.png"

    if calibration_path.exists():
        st.image(str(calibration_path), use_container_width=True)
    else:
        st.warning("Calibration curve chart was not found.")

    st.subheader("Confusion matrices")
    confusion_path = charts_dir / "03_confusion_matrices.png"

    if confusion_path.exists():
        st.image(str(confusion_path), use_container_width=True)
    else:
        st.warning("Confusion matrix chart was not found.")

    st.subheader("Validation feature importance")

    importance_files = {
        "Logistic Regression": "validation_importance_logistic_regression.png",
        "Random Forest": "validation_importance_random_forest.png",
        "K-Nearest Neighbors": "validation_importance_k_nearest_neighbors.png",
        "Gaussian Naive Bayes": "validation_importance_naive_bayes.png",
        "Gradient Boosting": "validation_importance_gradient_boosting.png",
    }

    selected_model = st.selectbox(
        "Choose a model",
        list(importance_files),
    )

    importance_path = importance_dir / importance_files[selected_model]

    if importance_path.exists():
        st.image(str(importance_path), use_container_width=True)
    else:
        st.warning(
            "Validation importance chart was not found for the selected model."
        )

    st.subheader("Feature-group comparison")
    group_chart = comparison_dir / "feature_group_roc_auc.png"

    if group_chart.exists():
        st.image(str(group_chart), use_container_width=True)
    else:
        st.warning("Feature-group comparison chart was not found.")

    st.caption(
        "Feature importance describes changes in validation performance "
        "after shuffling a feature. It does not prove that a feature causes "
        "the outcome."
    )
elif page == "Student clustering":
    st.subheader("Student clustering")

    st.info(
        "Clustering groups students with similar early behaviour. "
        "It is an unsupervised analysis, so it does not use final results "
        "to create the groups."
    )

    charts_dir = REPORTS / "charts"

    chart_files = {
        "Cluster sizes": "11_cluster_sizes.png",
        "K-Means cluster profiles": "10_kmeans_cluster_profiles.png",
        "Cluster outcomes": "09_cluster_outcomes.png",
        "Clustering metrics": "08_clustering_metrics.png",
    }

    selected_chart = st.selectbox(
        "Choose a clustering result",
        list(chart_files),
    )

    chart_path = charts_dir / chart_files[selected_chart]

    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
    else:
        st.warning(
            f"Chart not found: {chart_path}. "
            "Run the clustering analysis first."
        )

    st.caption(
        "Clusters describe broad behavioural patterns. They do not prove "
        "that a group causes failure or withdrawal."
    )    
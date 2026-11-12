# Student Risk Early-Warning System

First project update: a work-in-progress OULAD machine-learning project shared so teammates can run and develop it on their own computers. Further improvements will follow.

## What it does

- Estimates a binary outcome: Fail/Withdrawn versus Pass/Distinction, using early-course information up to day 28.
- Compares Logistic Regression, Random Forest, KNN, Gaussian Naive Bayes, and Gradient Boosting.
- Evaluates classifiers with five-fold cross-validation grouped by student ID, with preprocessing inside each training fold.
- Explores enrolment groups using K-Means, Agglomerative Clustering, DBSCAN, and HDBSCAN.
- Provides a Streamlit frontend for single predictions, model comparison, batch predictions, evaluation, and clustering charts.

This is a research prototype, not a finished or independently validated deployment. Risk estimates are not certain outcomes.

## What to share and what stays local

Share source code, this README, and dependency instructions. Each teammate creates their own virtual environment and generates their own feature tables, trained models, and reports. Do not upload venv, secrets, raw dataset CSVs, or generated student-level outputs. Charts may be deliberately shared later if needed.

## 1. Get the project and place the dataset

Clone this repository or download and extract its ZIP. Open a terminal in the project root: the folder containing src and requirements.txt.

Place the seven OULAD files directly inside data/:

```text
data/
  assessments.csv
  courses.csv
  studentAssessment.csv
  studentInfo.csv
  studentRegistration.csv
  studentVle.csv
  vle.csv
```

The original dataset is available from The Open University: https://figshare.com/articles/dataset/OULAD_Open_University_Learning_Analytics_Dataset/5081998

## 2. Create your own environment

Windows PowerShell:

```powershell
python --version
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is not required when using the explicit Python path above. On macOS/Linux, create the environment with python3 and replace .\venv\Scripts\python.exe in subsequent commands with ./venv/bin/python.

The original working Python and package versions have not yet been recorded. The dependencies are not locked; a clean installation on every platform has not been verified. Record the working versions when setup succeeds. Saved models should be regenerated locally rather than copied between incompatible environments. Load only trusted joblib files.

## 3. Build the day-28 features

Run these in order:

```powershell
.\venv\Scripts\python.exe src/feature_engineering.py
.\venv\Scripts\python.exe src/build_activity_features.py
.\venv\Scripts\python.exe src/build_assessment_features.py
```

This produces reports/features_day_28.csv. The cohort excludes withdrawals on/before day 28 and registrations after day 28. Activity covers days 0-28 inclusive. Assessment features use non-exam assessments due by the cutoff. Exact mark availability by the cutoff is a research limitation because score-release dates are not provided.

## 4. Save models and open the frontend

For the prediction frontend, run:

```powershell
.\venv\Scripts\python.exe src/save_models.py
.\venv\Scripts\python.exe -m streamlit run src/streamlit_app.py --server.headless true
```

save_models.py trains and saves five fold pipelines per classifier under reports/saved_models/. It can take time. The app needs these pipelines and Gradient Boosting metadata. It averages saved fold probabilities for each classifier; batch mode currently uses Gradient Boosting.

Open the Local URL printed in the terminal, usually http://localhost:8501. Keep the terminal running. After Ctrl+C, rerun only the Streamlit command if the saved models are still available.

The current frontend is src/streamlit_app.py. The older app/ FastAPI scaffold is not the frontend used here.

## 5. Generate evaluation and clustering charts

These steps are needed for the corresponding report pages, not for building the initial prediction models:

```powershell
.\venv\Scripts\python.exe src/train_models.py
.\venv\Scripts\python.exe src/train_additional_models.py
.\venv\Scripts\python.exe src/analyze_results.py
.\venv\Scripts\python.exe src/cluster_students.py
.\venv\Scripts\python.exe src/generate_charts.py
```

Run in that order. generate_charts.py expects the prediction files produced by analyze_results.py and train_additional_models.py, plus the clustering tables. Main charts are saved under reports/charts/. Training and analysis may take considerable time; full-cohort agglomerative clustering and DBSCAN can require substantial memory.

Additional experiments:

```powershell
.\venv\Scripts\python.exe src/validate_importance.py
.\venv\Scripts\python.exe src/compare_feature_groups.py
```

These create reports/validation_importance/ and reports/feature_group_comparison/. Prefer held-out importance from validate_importance.py over the legacy training-set importance from analyze_results.py. The comparison configuration named full is a selected feature subset, not all baseline inputs.

## Project folders

```text
src/          Source scripts and current Streamlit app
data/         Local original OULAD CSVs
reports/      Locally generated feature tables, models, and reports
models/       Earlier placeholder, not the active saved-model directory
app/          Earlier FastAPI scaffold
notebooks/    Notebook workspace
```

## Current first-update notes

- Generate models and reports locally; they are intentionally not required in GitHub.
- Use saved out-of-fold predictions for evaluation. Running original training records through the app's averaged pipelines is not an independent test.
- Clustering charts describe groups, not individual risk predictions. Cluster IDs have no automatic high/low-risk meaning. Current density-method silhouette scores include noise as a label.
- The manual forms still need stronger consistency checks and complete category choices. Derived inputs must match the feature-builder formulas. Days 0-28 contain 29 day indices, although current form limits stop at 28 for active days and inactivity.
- The app's risk-tier boundaries differ from the legacy analysis chart boundaries. The app's key risk indicators are handwritten checks, not model-specific explanations.
- The batch results display currently expects id_student, code_module, and code_presentation as well as the required feature columns.
- If a report page shows a missing-file message, generate the relevant reports and verify the filename against that page's code.
- End-to-end clean-machine testing, dependency locking, input validation, and further research improvements remain future updates.

---

## Original README (preserved)

The original README is reproduced unchanged below for reference. For the current runnable workflow, use the teammate instructions above.

# Early Prediction of Student Academic Performance Using Machine Learning and Learning Behaviour Data

This project uses the Open University Learning Analytics Dataset (OULAD) to predict student academic outcomes from early-course learning behaviour.

## Research area

Educational Data Mining and Predictive Machine Learning

## Project objective

Build machine-learning models that predict whether a student is likely to pass, fail, withdraw, or achieve distinction, using information available during the early weeks of a course.

## Dataset

Open University Learning Analytics Dataset (OULAD), publicly available from The Open University.

Place the downloaded OULAD CSV files inside the `data/` folder.

## Algorithms

### Classification

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Random Forest Classifier
- Gradient Boosting Classifier

### Clustering

- K-Means
- Agglomerative Hierarchical Clustering
- DBSCAN
- HDBSCAN

### Supporting analysis

- Principal Component Analysis (PCA)
- Feature-importance analysis
- Model evaluation and comparison

## Project structure

```text
.
├── data/                  # OULAD CSV files
├── notebooks/             # Exploratory analysis and experiments
├── src/                   # Reusable Python modules
├── models/                # Saved trained models
├── reports/figures/       # Generated charts
└── app/                   # FastAPI backend and simple web interface
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web application

```bash
uvicorn app.main:app --reload
```

Then open: http://127.0.0.1:8000

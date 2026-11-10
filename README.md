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

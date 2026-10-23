# Telecom Customer Churn Prediction

A supervised machine-learning project for identifying telecom customers who may leave a service. The workflow explores churn patterns, prepares customer-level features, compares classification models, and supports risk-based retention analysis.

## Business problem

Customer churn can reduce recurring revenue and increase acquisition costs. A churn model can help teams prioritize retention outreach, provided its scores are evaluated carefully and used alongside business context.

## Project objectives

- Explore the distribution of churn and customer behavior.
- Prepare the target variable `CHURN_FLAG` (1 = churn, 0 = retained).
- Engineer aggregate usage features.
- Compare binary classification models.
- Produce churn probabilities or risk segments for analysis.

## Technology stack

Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn are suitable for the data preparation, visualization, and modeling workflow documented by this project. Check the notebook and dependency files for the exact libraries used in the current implementation.

## Dataset overview

The existing project notes describe approximately 4,617 customer records and a churn rate of approximately 14.2%. Features include account tenure, international and voicemail plans, calling activity, charges, customer-service calls, and an original churn indicator.

Validate these figures against the version of the dataset in the repository before using them in a report or interview. Do not publish personal identifiers from customer records.

## Workflow

1. **Data quality:** inspect schema, missing values, duplicates, and class distribution.
2. **Target preparation:** derive a binary target from the original churn column.
3. **Feature engineering:** create aggregate usage variables and encode categorical/binary fields.
4. **Exploratory analysis:** study churn rates across customer behavior and service-plan groups.
5. **Modeling:** establish a baseline and compare candidate classifiers.
6. **Evaluation:** use precision, recall, F1-score, ROC-AUC, and a confusion matrix.
7. **Risk segmentation:** if calibrated probabilities are available, define and validate risk tiers.
8. **Business interpretation:** connect model findings to testable retention actions.

## Repository contents

The repository includes a main Jupyter Notebook, a source CSV, and customer-level prediction/risk-score CSV files according to the existing project documentation. Check the repository tree for the current filenames before running the notebook.

## Run the notebook

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the dependencies from the repository's requirements file if present, then launch Jupyter:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

If the repository does not contain a file named `requirements.txt`, create one from the packages actually imported by the notebook before running the install command.

## Evaluation and business impact

Do not treat a model's accuracy alone as evidence of business value, especially when churn classes are imbalanced. Record the held-out test metrics, threshold-selection method, calibration results (if used), and a baseline comparison. No new performance or revenue-impact claims are made here without verified evaluation evidence.

## Important modeling considerations

- Split data before fitting transformations to avoid data leakage.
- Check for identifiers or post-outcome variables that could leak the target.
- Use stratified splitting when appropriate.
- Select thresholds based on the cost of false positives and false negatives.
- Keep sensitive customer data out of public examples and screenshots.

## Future improvements

- [ ] Add a concise data dictionary and verified dataset source.
- [ ] Document the exact models and hyperparameters compared.
- [ ] Add a reproducible metrics table and confusion matrix.
- [ ] Validate probability calibration before assigning risk tiers.
- [ ] Add explainability for retention stakeholders.
- [ ] Add tests and a reproducible dependency file.
- [ ] Document the output CSV schema and safe handling of customer-level files.

## License

Add or confirm the dataset and code licenses before redistribution.

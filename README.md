# Hillstrom Email Uplift Modeling

This project analyzes whether promotional emails increase purchases and estimates which customer profiles benefit most from receiving an email. It combines randomized A/B testing data with uplift modeling to support campaign targeting.

## Dataset

The project uses the Hillstrom email marketing dataset, containing 64,000 customer records from a randomized experiment. Customers were assigned to three groups:

- **No E-Mail:** the control group, which received no promotional email.
- **Mens E-Mail:** a treatment group that received a men's merchandise email.
- **Womens E-Mail:** a treatment group that received a women's merchandise email.

Customer attributes include purchase recency, purchase history, previous men's and women's purchases, zip-code category, new-customer status, and shopping channel. The recorded outcomes are website visits (`visit`), purchases (`conversion`), and spending (`spend`). This project predicts uplift in **conversion**, meaning whether a customer made a purchase.

## Why this dataset?

Randomized treatment assignment and a control group allow purchase outcomes after an email to be compared with outcomes without an email. Customer attributes recorded before treatment also allow the models to estimate how the email's effect varies across customer profiles.

The project evaluates Mens E-Mail versus No E-Mail and Womens E-Mail versus No E-Mail separately. Each comparison uses the same control group, so one email campaign is not mistakenly treated as the untreated baseline for the other.

## Why A/B testing is done

Some customers would purchase even without an email. Counting purchases among email recipients alone cannot establish how many additional purchases the campaign caused. A/B testing randomly assigns customers to treatment and control groups and compares their conversion rates to estimate the campaign's average effect.

```text
Conversion uplift = email-group conversion rate - control-group conversion rate
```

For example, conversion rates of 3% in an email group and 2% in the control group represent an increase of **1 percentage point**. These values are illustrative, not project results.

A/B testing measures the average campaign effect. Uplift modeling extends this analysis by estimating which customer profiles benefit more or less from receiving an email. This helps prioritize incremental purchases rather than simply target customers who are already likely to buy. The experiment is already recorded in the dataset; this pipeline analyzes it and does not send emails or run a new experiment.

## Project structure

```text
uplift_modeling/
    data_collection.py     # Download Hillstrom data and save/reuse the raw CSV
    data_preparation.py    # Validate data and identify pretreatment features
    model_building.py      # Preprocessing, S-/T-learner construction and prediction
    training.py            # Stratified splits, saved datasets, and model fitting
    evaluation.py          # Validation selection, test metrics, curves, and audit
    pipeline.py            # Command-line workflow and model export
data/
    raw/hillstrom.csv      # Extracted source dataset
    processed/             # Per-campaign train, validation, and test CSVs
models/                    # Selected models saved as a joblib bundle
reports/
    figures/               # Qini and uplift plots
    *.csv                  # Data checks, validation/test metrics, uplift audit
app.py                     # Streamlit prediction application
requirements.txt
```

## Setup and execution

Run these commands from this project directory using Python 3.10 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Collect and store data separately, without fitting models:

```powershell
python -m uplift_modeling.pipeline --collect-only
```

Run the full collection, preparation, training, evaluation, and export workflow:

```powershell
python -m uplift_modeling.pipeline
```

Existing `data/raw/hillstrom.csv` is reused. Add `--refresh-data` to fetch the source again. The first collection requires internet access unless scikit-uplift has already cached the dataset. Rerunning training overwrites generated splits, reports, and models. Paths are relative to your current working directory, so run commands from the `Uplift Modelling` project folder.

Launch the prediction app after training:

```powershell
python -m streamlit run app.py
```

## Modeling behavior

The outcome is `conversion`. Each email treatment is compared separately with No E-Mail. `visit`, `spend`, `conversion`, and `segment` are excluded from baseline predictors to prevent outcome leakage. Matching customer attributes are not treated as proof of duplicate customers.

Each campaign uses a reproducible, treatment-and-outcome-stratified 60/20/20 split. Imputation and encoding are fitted on training data. S-learner and T-learner random forests are compared using validation Qini AUC; only the selected approach is evaluated on the held-out test set and exported. Exported models remain fitted on the training split.

`reports/validation_metrics.csv`, `reports/test_metrics.csv`, and `reports/uplift_audit.csv` contain measured results. Estimated uplift is an incremental conversion estimate, not a guaranteed individual outcome. Load only trusted joblib files.

Generated data and artifacts are excluded from Git. Dependencies are listed in `requirements.txt`; use the same environment for training and prediction because serialized sklearn models depend on library versions.

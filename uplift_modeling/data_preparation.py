"""Validate outcomes and identify pretreatment features without leakage."""
import numpy as np
from pathlib import Path

REPORTS_DIR = Path('reports')

def prepare_data(df):
    # Keep data-quality reports available even when this function is run on its own.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Fail early if core experiment labels or outcomes are absent.
    assert {'segment','conversion','visit','spend'}.issubset(df.columns)
    # Identify the untreated baseline from the dataset's treatment labels.
    control_label=next((x for x in df.segment.unique() if 'no e-mail' in str(x).lower() or 'no email' in str(x).lower()),None)
    assert control_label is not None, 'Inspect experiment labels and set control_label manually.'
    treatment_labels=[s for s in df.segment.unique() if s!=control_label]
    assert len(treatment_labels)==2
    # Exclude experiment assignment and post-treatment outcomes to prevent leakage.
    predictors=df.drop(columns=['segment','conversion','visit','spend']).columns.tolist()
    assert set(predictors).isdisjoint({'conversion','visit','spend','segment'})
    print('Control:',control_label,'Treatments:',treatment_labels)
    print('Pretreatment features:',predictors)
    for label in treatment_labels:
        sub=df[df.segment.isin([control_label,label])]
        print(label,sub.shape,sub.groupby('segment').conversion.mean().to_dict())
    # Save schema and missingness summaries so data issues can be inspected later.
    df.isna().sum().to_csv(REPORTS_DIR / 'missing_values.csv')
    df.dtypes.astype(str).to_csv(REPORTS_DIR / 'data_types.csv')
    print('Identical rows (not deleted):',df.duplicated().sum())
    # Confirm conversion is a complete binary outcome before fitting classifiers.
    assert df.conversion.dropna().isin([0,1]).all()
    assert not df['conversion'].isna().any()
    # Return feature types so model preprocessing can handle each consistently.
    num_cols=df[predictors].select_dtypes(include=np.number).columns.tolist()
    cat_cols=[col for col in predictors if col not in num_cols]
    print('Numeric:',num_cols,'Categorical:',cat_cols)
    return predictors, num_cols, treatment_labels, control_label

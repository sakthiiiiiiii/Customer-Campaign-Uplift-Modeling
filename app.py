import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path('models/hillstrom_uplift_models.joblib')

st.set_page_config(page_title="Email Uplift Predictor")
st.title("Email Campaign Uplift Predictor")
st.caption("Estimated incremental conversion probability; no email is the baseline. Not a guarantee.")
# Stop before loading a missing artifact so users get a clear training instruction.
if not MODEL_PATH.exists():
    st.error("Train the models first: python -m uplift_modeling.pipeline")
    st.stop()
bundle=joblib.load(MODEL_PATH)
features=bundle["predictors"]
# Build inputs from the saved feature list so the app stays aligned with training.
values={}
for column in features:
    if column == "recency": values[column]=st.number_input(column,min_value=0,value=5,step=1)
    elif column == "history": values[column]=st.number_input(column,min_value=0.0,value=100.0)
    elif column == "mens": values[column]=st.selectbox(column,[0,1])
    elif column == "womens": values[column]=st.selectbox(column,[0,1])
    elif column == "zip_code": values[column]=st.selectbox(column,["Urban","Suburban","Rural"])
    elif column == "newbie": values[column]=st.selectbox(column,[0,1])
    elif column == "channel": values[column]=st.selectbox(column,["Phone","Web","Multichannel"])
    else: values[column]=st.text_input(column,"Unknown")

if st.button("Estimate campaign uplift"):
    # Preserve the model's training column order when creating one customer row.
    X=pd.DataFrame([values],columns=features)
    scores=[]
    for label,details in bundle['campaigns'].items():
        if details['strategy']=='S-learner':
            # Score the same customer as treated and untreated to estimate uplift.
            a=X.assign(email_received=1);b=X.assign(email_received=0)
            uplift=details['model'].predict_proba(a)[:,1][0]-details['model'].predict_proba(b)[:,1][0]
        else:
            # The T-learner has separate outcome models for treated and control customers.
            tr,ctrl=details['model']
            uplift=tr.predict_proba(X)[:,1][0]-ctrl.predict_proba(X)[:,1][0]
        # Convert probability difference to percentage points for easier reading.
        scores.append({'Campaign':label,'Predicted uplift (percentage points)':round(float(uplift)*100,2)})
    st.dataframe(pd.DataFrame(scores),hide_index=True)
    st.info("Negative uplift suggests sending that campaign may not help. Validate decisions via randomized trials and account for email costs.")

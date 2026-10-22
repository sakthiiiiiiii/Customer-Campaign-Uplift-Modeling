"""Persist stratified 60/20/20 splits and train each campaign independently."""
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from .model_building import fit_s_learner, fit_t_learner

PROCESSED_DIR = Path('data/processed')

def train_models(df, predictors, treatment_labels, control_label):
    experiments={}
    # Compare each email campaign to the same untreated group in an independent experiment.
    for label in treatment_labels:
        exp=df.loc[df.segment.isin([control_label,label])].copy().reset_index(drop=True)
        exp['email_received']=(exp.segment==label).astype(int)
        # Stratification preserves treatment/outcome proportions in the initial split.
        strat=exp.email_received.astype(str)+'_'+exp.conversion.astype(str)
        train_idx,remaining_idx=train_test_split(np.arange(len(exp)),test_size=.4,random_state=42,stratify=strat)
        remaining=exp.iloc[remaining_idx]
        remaining_strat=remaining.email_received.astype(str)+'_'+remaining.conversion.astype(str)
        # Split the remaining data evenly to create separate validation and test sets.
        valid_idx,test_idx=train_test_split(remaining_idx,test_size=.5,random_state=42,stratify=remaining_strat)
        train=exp.iloc[train_idx];valid=exp.iloc[valid_idx];test=exp.iloc[test_idx]
        campaign_dir = PROCESSED_DIR / str(label).lower().replace(' ', '_').replace('-', '')
        campaign_dir.mkdir(parents=True, exist_ok=True)
        for split_name, frame in [('train', train), ('validation', valid), ('test', test)]:
            frame.to_csv(campaign_dir / (split_name + '.csv'), index=False)
        # Fit both learner strategies using training data only; validation selects between them.
        Xtr=train[predictors];ytr=train.conversion.astype(int);ttr=train.email_received.astype(int)
        s=fit_s_learner(Xtr,ytr,ttr)
        t=fit_t_learner(Xtr,ytr,ttr)
        experiments[str(label)]={'train':train,'valid':valid,'test':test,'S-learner':s,'T-learner':t}
        print(label,'trained:',len(train),'validation:',len(valid),'test:',len(test))
    return experiments

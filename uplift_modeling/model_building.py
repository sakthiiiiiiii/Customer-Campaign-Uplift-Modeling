"""Build and score S-learner and T-learner pipelines."""
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
base_classifier = RandomForestClassifier(n_estimators=140, min_samples_leaf=20, random_state=42, n_jobs=-1)

def make_pipeline(features, num_cols):
    # Keep numeric and categorical preprocessing distinct to match each feature's data type.
    nums=[c for c in features if c in num_cols or c=='email_received']
    cats=[c for c in features if c not in nums]
    transformer=ColumnTransformer([
      # Impute numeric gaps and scale values before fitting the classifier.
      ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      # Impute categories and encode them while tolerating categories seen only at prediction.
      ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)
    ])
    # Clone the shared classifier so each learner gets an independent fitted estimator.
    return Pipeline([('preprocess',transformer),('model',clone(base_classifier))])

def fit_s_learner(X,y,t):
    predictors = X.columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    # Include treatment assignment as an input so one model can learn treatment effects.
    model=make_pipeline(predictors+['email_received'], num_cols)
    X1=X.copy();X1['email_received']=np.asarray(t)
    model.fit(X1,y)
    return model

def predict_s_uplift(model,X):
    # Estimate individual uplift as the treated prediction minus the control prediction.
    treated=X.copy();treated['email_received']=1
    control=X.copy();control['email_received']=0
    return model.predict_proba(treated)[:,1]-model.predict_proba(control)[:,1]

def fit_t_learner(X,y,t):
    predictors = X.columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    t=np.asarray(t).astype(bool)
    # Fit separate response models so each estimates outcomes within its own group.
    treated_model=make_pipeline(predictors, num_cols)
    control_model=make_pipeline(predictors, num_cols)
    treated_model.fit(X.loc[t],y.loc[t]);control_model.fit(X.loc[~t],y.loc[~t])
    return treated_model,control_model

def predict_t_uplift(models,X):
    # Difference the two group-specific conversion probabilities to estimate uplift.
    return models[0].predict_proba(X)[:,1]-models[1].predict_proba(X)[:,1]

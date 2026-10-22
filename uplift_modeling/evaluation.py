"""Select on validation data, then evaluate the selected models on test data."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from sklift.metrics import qini_curve, uplift_curve
from pathlib import Path
from .model_building import predict_s_uplift, predict_t_uplift

REPORTS_DIR = Path('reports')

def plot_qini_curve(y, scores, treatment):
    _plot_curve(qini_curve, y, scores, treatment)

def plot_uplift_curve(y, scores, treatment):
    _plot_curve(uplift_curve, y, scores, treatment)

def _plot_curve(curve, y, scores, treatment):
    x, values = curve(y, scores, treatment)
    plt.figure()
    plt.plot(x, values, label='Selected model')
    # Compare the model's ranking against an unranked random-targeting baseline.
    plt.plot([0, x[-1]], [0, values[-1]], '--', label='Random targeting')
    plt.xlabel('Customers targeted')
    plt.ylabel('Incremental conversions')
    plt.legend()

def evaluate_models(experiments, predictors):
    (REPORTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    validation_rows=[]
    # Use validation outcomes to compare learners without using the held-out test set.
    for label,run in experiments.items():
        val=run['valid'];Xv=val[predictors];yv=val.conversion.astype(int).to_numpy();tv=val.email_received.to_numpy()
        for name in ['S-learner','T-learner']:
            preds=predict_s_uplift(run[name],Xv) if name=='S-learner' else predict_t_uplift(run[name],Xv)
            validation_rows.append({'campaign':label,'model':name,'val_qini_auc':qini_auc_score(yv,preds,tv),'val_uplift_auc':uplift_auc_score(yv,preds,tv)})
    validation_results=pd.DataFrame(validation_rows).sort_values(['campaign','val_qini_auc'],ascending=[True,False])
    validation_results.to_csv(REPORTS_DIR / 'validation_metrics.csv', index=False)
    # Lock one learner per campaign before measuring final test performance.
    for label,run in experiments.items():
        selected=validation_results.loc[validation_results.campaign==label].iloc[0]['model']
        run['selected_model_name']=selected
    print('Model selections locked prior to evaluating the test set')
    test_rows=[]
    # Evaluate only the selected model on untouched test customers.
    for label,run in experiments.items():
        test=run['test'];Xt=test[predictors];yt=test.conversion.astype(int).to_numpy();tt=test.email_received.to_numpy()
        chosen=run['selected_model_name']
        preds=predict_s_uplift(run[chosen],Xt) if chosen=='S-learner' else predict_t_uplift(run[chosen],Xt)
        run['test_scores']=preds
        test_rows.append({'campaign':label,'chosen_model':chosen,'test_qini_auc':qini_auc_score(yt,preds,tt),
                          'test_uplift_auc':uplift_auc_score(yt,preds,tt),
                          'uplift_at_top_30pct':uplift_at_k(yt,preds,tt,strategy='overall',k=.3)})
        # Save ranking curves to inspect gains over random targeting visually.
        plot_qini_curve(yt,preds,tt);plt.title(f'Qini — {label} ({chosen})');plt.savefig(REPORTS_DIR / 'figures' / (label.replace(' ', '_') + '_' + str(len(plt.get_fignums())) + '_' + plt.gca().get_title().split(' ')[0] + '.png'), bbox_inches='tight');plt.close('all')
        plot_uplift_curve(yt,preds,tt);plt.title(f'Uplift curve — {label}');plt.savefig(REPORTS_DIR / 'figures' / (label.replace(' ', '_') + '_' + str(len(plt.get_fignums())) + '_' + plt.gca().get_title().split(' ')[0] + '.png'), bbox_inches='tight');plt.close('all')
    test_results=pd.DataFrame(test_rows)
    test_results.to_csv(REPORTS_DIR / 'test_metrics.csv', index=False)
    print('Top 30% uplift is an estimate from randomized test groups, not guaranteed profit.')
    audit=[]
    # Compare observed randomized outcomes in the full test set and the model-ranked segment.
    for label,run in experiments.items():
        test=run['test'].copy();test['uplift_score']=run['test_scores']
        top=test.nlargest(max(1,int(round(len(test)*.30))), 'uplift_score')
        for title,frame in [('All test customers',test),('Top 30% ranked',top)]:
            rates=frame.groupby('email_received').conversion.agg(['mean','size'])
            if len(rates)==2:
                audit.append({'campaign':label,'population':title,'treated_count':int(rates.loc[1,'size']),
                  'control_count':int(rates.loc[0,'size']),
                  'treated_rate':float(rates.loc[1,'mean']),'control_rate':float(rates.loc[0,'mean']),
                  'observed_uplift_pp':100*float(rates.loc[1,'mean']-rates.loc[0,'mean'])})
    pd.DataFrame(audit).to_csv(REPORTS_DIR / 'uplift_audit.csv', index=False)
    print('Model results require prospective randomized campaign validation before production use.')
    return validation_results, test_results

"""Orchestrate collection, preparation, training, evaluation, and export."""
import argparse
import joblib
from pathlib import Path
from .data_collection import collect_data
from .data_preparation import prepare_data

MODEL_PATH = Path('models/hillstrom_uplift_models.joblib')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true', help='Save raw data without training')
    parser.add_argument('--refresh-data', action='store_true', help='Fetch data again instead of using the saved CSV')
    args = parser.parse_args()
    # Collect first so both collect-only and full runs use the same saved source data.
    df = collect_data(refresh=args.refresh_data)
    print(f'Loaded {len(df)} rows from data/raw/hillstrom.csv')
    if args.collect_only:
        return
    # Import training/evaluation only for full runs to keep collection-only lightweight.
    from .training import train_models
    from .evaluation import evaluate_models
    # Prepare features, train each comparison, then select models using validation data.
    predictors, _, treatments, control = prepare_data(df)
    experiments = train_models(df, predictors, treatments, control)
    evaluate_models(experiments, predictors)
    # Export only selected models plus feature metadata needed by the prediction app.
    export = {'predictors': predictors, 'campaigns': {}}
    for label, run in experiments.items():
        chosen = run['selected_model_name']
        export['campaigns'][label] = {'strategy': chosen, 'model': run[chosen]}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(export, MODEL_PATH)
    print(f'Saved selected models: {MODEL_PATH}')

if __name__ == '__main__':
    main()

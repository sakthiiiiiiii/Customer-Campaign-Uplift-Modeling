"""Download the randomized experiment and persist the extracted data."""
import pandas as pd
from sklift.datasets import fetch_hillstrom
from pathlib import Path

DATA_PATH = Path('data/raw/hillstrom.csv')

def collect_data(refresh=False):
    # Reuse the extracted dataset unless a refresh is explicitly requested.
    if DATA_PATH.exists() and not refresh:
        return pd.read_csv(DATA_PATH)

    # Combine customer features, treatment assignment, and outcomes into one table.
    bundle = fetch_hillstrom(target_col='all')
    frame = pd.concat([
        bundle.data.reset_index(drop=True),
        pd.Series(bundle.treatment).reset_index(drop=True).rename('segment'),
        bundle.target.reset_index(drop=True),
    ], axis=1)

    # Persist the assembled source once so later runs do not need to fetch it again.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame

"""Standalone regression test for strict-future target and past-only inputs."""
import pandas as pd
from build_and_evaluate import build_panel, cut_stats


def run():
    t = pd.Timestamp('2015-02-01 06:00:00')
    machines = pd.DataFrame({'machineID': [1], 'model': ['model1'], 'age': [5]})
    maint = pd.DataFrame({'datetime': [t - pd.Timedelta(days=1), t], 'machineID': [1, 1], 'comp': ['comp1', 'comp2']})
    errors = pd.DataFrame({'datetime': [t], 'machineID': [1], 'errorID': ['error1']})
    # A failure EXACTLY at t must not be treated as a future prediction target.
    same_time_only = pd.DataFrame({'datetime': [t], 'machineID': [1], 'failure': ['comp1']})
    panel = build_panel(machines, maint, same_time_only, errors)
    row = panel.loc[panel.prediction_time == t].iloc[0]
    assert row.label_failure_7d == 0, 'Prediction-time failure counted as future failure'
    assert row.maintenance_events_prior == 1, 'Prediction-time maintenance included as history'
    assert row.failure_events_prior == 0, 'Prediction-time failure included as history'
    assert row.error_events_prior == 0, 'Prediction-time error included as history'

    # A genuine future failure within the next seven days must be labelled positive.
    future = pd.DataFrame({'datetime': [t, t + pd.Timedelta(days=1)], 'machineID': [1,1], 'failure': ['comp1','comp2']})
    future_panel = build_panel(machines, maint, future, errors)
    row2 = future_panel.loc[future_panel.prediction_time == t].iloc[0]
    assert row2.label_failure_7d == 1, 'Future failure incorrectly excluded'
    print('PASS: strict-future labels and past-only maintenance/failure/error inputs')

if __name__ == '__main__':
    run()

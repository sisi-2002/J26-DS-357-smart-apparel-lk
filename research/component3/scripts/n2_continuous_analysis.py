#!/usr/bin/env python3
"""N2 PP1 exploratory continuous evidence--prediction-loss analysis.

Starts from *held-out predictions* produced by n2_history_pilot.py and
joins evidence variables rebuilt independently for each history condition.
It does not retrain/alter the Function 1 classifier or claim a per-machine
'reliability percentage'. All analysis is exploratory on synthetic Azure PdM.

Usage from research/component3/:
  python scripts/n2_continuous_analysis.py --data-dir RAW_PATH \
      --losses outputs_corrected/n2_paired_test_prediction_losses.csv \
      --out-dir outputs_n2_continuous
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_and_evaluate import build_panel, load_csv

CONDITIONS = {
    'Full observed history': None,
    '180-day history': 180,
    '90-day history': 90,
}
TARGET = 'label_failure_7d'
KEYS = ['machineID', 'prediction_time']
EVIDENCE = [
    'observed_maintenance_span_days',  # H (NOT confirmed logging duration)
    'maintenance_events_prior',        # M
    'failure_events_prior',            # F
    'days_since_last_maintenance',     # R
    'published_field_completeness_pct',# C: constant in unmasked Azure data
]
MODEL_INPUTS = EVIDENCE[:4]  # C is constant =100%, cannot be learned here


def load_and_link(data_dir: Path, loss_csv: Path) -> pd.DataFrame:
    """Join condition-specific, strictly past-only evidence to saved test losses."""
    if not loss_csv.is_file():
        raise FileNotFoundError(f'{loss_csv} missing. First run n2_history_pilot.py.')
    losses = pd.read_csv(loss_csv, parse_dates=['prediction_time'])
    required = {*KEYS, 'history_condition', TARGET, 'predicted_probability', 'sq_error'}
    missing = required - set(losses)
    if missing:
        raise ValueError(f'N2 losses are missing required columns: {sorted(missing)}')
    unknown = set(losses.history_condition.dropna().unique()) - set(CONDITIONS)
    if unknown:
        raise ValueError(f'Unknown history conditions: {sorted(unknown)}')
    if set(losses.history_condition.unique()) != set(CONDITIONS):
        raise ValueError('The N2 losses file must contain all three history conditions.')
    if losses.duplicated(KEYS + ['history_condition']).any():
        raise ValueError('Duplicate machine/date/condition rows in N2 losses.')
    if not losses['predicted_probability'].between(0, 1).all():
        raise ValueError('Probabilities outside [0, 1].')
    if not losses[TARGET].isin([0, 1]).all():
        raise ValueError('Failure labels must be binary 0/1.')
    calc_loss = (losses[TARGET] - losses.predicted_probability) ** 2
    if not np.allclose(calc_loss, losses.sq_error, atol=1e-10, rtol=1e-8):
        raise ValueError('Saved squared losses do not match labels and predictions.')

    machines = load_csv(data_dir, 'machines', False)
    maint = load_csv(data_dir, 'maint', True)
    failures = load_csv(data_dir, 'failures', True)
    errors = load_csv(data_dir, 'errors', True)
    frames = []
    reference_keys = None
    for name, days in CONDITIONS.items():
        panel = build_panel(machines, maint, failures, errors, max_history_days=days)
        test = panel.loc[panel.split.eq('test'), KEYS + [TARGET, 'machine_model'] + EVIDENCE].copy()
        if test.duplicated(KEYS).any():
            raise ValueError(f'Duplicate test snapshots for {name}.')
        here = losses.loc[losses.history_condition.eq(name)].copy()
        joined = here.merge(test, on=KEYS, how='outer', indicator=True, suffixes=('_saved', '_source'), validate='one_to_one')
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'Missing or extra machine-time examples for {name}.')
        if not joined[f'{TARGET}_saved'].eq(joined[f'{TARGET}_source']).all():
            raise ValueError(f'Target labels disagree with corrected strict-future source for {name}.')
        joined = joined.drop(columns=['_merge', f'{TARGET}_saved']).rename(columns={f'{TARGET}_source':TARGET})
        if reference_keys is None:
            reference_keys = set(map(tuple, joined[KEYS].to_numpy()))
        elif set(map(tuple, joined[KEYS].to_numpy())) != reference_keys:
            raise ValueError('History conditions do not share the same test machine-time examples.')
        frames.append(joined)
    table = pd.concat(frames, ignore_index=True).sort_values(['history_condition','machineID','prediction_time'])
    table['H_observed_span_days'] = table['observed_maintenance_span_days']
    table['M_maintenance_count'] = table['maintenance_events_prior']
    table['F_prior_failure_count'] = table['failure_events_prior']
    table['C_field_completeness_pct'] = table['published_field_completeness_pct']
    table['R_days_since_maintenance'] = table['days_since_last_maintenance']
    return table.reset_index(drop=True)


def descriptive_associations(table: pd.DataFrame) -> pd.DataFrame:
    """Descriptive Spearman correlations: NOT significance tests (repeated machines)."""
    recs = []
    for cond, group in table.groupby('history_condition'):
        for field in EVIDENCE:
            non_missing = group[[field, 'sq_error']].dropna()
            rho = (non_missing[field].corr(non_missing.sq_error, method='spearman')
                   if len(non_missing) > 2 and non_missing[field].nunique() > 1 else np.nan)
            recs.append({
                'history_condition':cond, 'evidence_variable':field,
                'observations_used':int(len(non_missing)),
                'unique_evidence_values':int(non_missing[field].nunique()),
                'spearman_rho_descriptive':rho,
                'warning':'Repeated machine snapshots; no independence-based p-value',
            })
    return pd.DataFrame(recs)


def grouped_loss(table: pd.DataFrame, variable: str, bins: int = 5) -> pd.DataFrame:
    """Quantile bins within each condition, with observed event prevalence for context."""
    records = []
    for condition, group in table.groupby('history_condition'):
        valid = group.dropna(subset=[variable]).copy()
        if valid[variable].nunique() < 2:
            continue
        # Quantile grouping uses actual values and does not split tied values.
        valid['evidence_bin'] = pd.qcut(valid[variable], q=bins, labels=False, duplicates='drop')
        for bin_id, block in valid.groupby('evidence_bin'):
            records.append({
                'history_condition':condition,
                'evidence_variable':variable,
                'bin_index':int(bin_id)+1,
                'n_snapshots':int(len(block)),
                'n_machines':int(block.machineID.nunique()),
                'evidence_mean':float(block[variable].mean()),
                'evidence_min':float(block[variable].min()),
                'evidence_max':float(block[variable].max()),
                'mean_squared_prediction_error_brier':float(block.sq_error.mean()),
                'observed_failure_prevalence':float(block[TARGET].mean()),
            })
    return pd.DataFrame(records)


def _loss_model() -> Pipeline:
    nums = Pipeline([
        ('impute',SimpleImputer(strategy='median', add_indicator=True)),
        ('scale',StandardScaler()),
    ])
    cols = ColumnTransformer([
        ('evidence',nums,MODEL_INPUTS),
        ('type',OneHotEncoder(handle_unknown='ignore'),['machine_model']),
    ])
    return Pipeline([('features',cols),('ridge',Ridge(alpha=20.0))])


def grouped_machine_cv(table: pd.DataFrame, folds: int = 5):
    """Cross-fit regression on squared losses, keeping entire machines in folds.

    The base failure predictor itself was previously trained on earlier rows of
    these same machines. Thus this evaluates the *second-stage relationship*
    across machine groups, NOT generalization of the entire pipeline to new machines.
    """
    summary, predictions = [], []
    for condition, group in table.groupby('history_condition'):
        group=group.copy().reset_index(drop=True)
        n_groups=group.machineID.nunique()
        if n_groups < folds:
            raise ValueError(f'Need >= {folds} machines per condition; found {n_groups}.')
        groups=group.machineID.to_numpy()
        outcomes=group.sq_error.to_numpy()
        oof = np.full(len(group), np.nan)
        base = np.full(len(group), np.nan)
        fold_ids = np.full(len(group), -1, dtype=int)
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(group, outcomes, groups), start=1):
            if set(groups[tr]).intersection(set(groups[te])):
                raise AssertionError('Machine group crossover detected.')
            model=_loss_model()
            model.fit(group.loc[tr, MODEL_INPUTS+['machine_model']],outcomes[tr])
            oof[te]=np.clip(model.predict(group.loc[te,MODEL_INPUTS+['machine_model']]),0,1)
            base[te]=float(outcomes[tr].mean())
            fold_ids[te]=fold
        if np.isnan(oof).any() or np.isnan(base).any():
            raise AssertionError('Not every observation received a held-out prediction.')
        summary.append({
            'history_condition':condition,
            'n_snapshots':len(group), 'n_machines':n_groups,
            'folds_grouped_by_machine':folds,
            'mean_observed_squared_error':float(outcomes.mean()),
            'mae_constant_mean_baseline':float(mean_absolute_error(outcomes,base)),
            'mae_evidence_ridge':float(mean_absolute_error(outcomes,oof)),
            'rmse_constant_mean_baseline':float(np.sqrt(mean_squared_error(outcomes,base))),
            'rmse_evidence_ridge':float(np.sqrt(mean_squared_error(outcomes,oof))),
            'r2_evidence_ridge':float(r2_score(outcomes,oof)),
            'improved_mae_vs_constant': bool(mean_absolute_error(outcomes,oof)<mean_absolute_error(outcomes,base)),
        })
        predictions.append(pd.DataFrame({
            'history_condition':condition,
            'machineID':groups,
            'prediction_time':group.prediction_time,
            'machine_group_fold':fold_ids,
            'observed_individual_sq_error':outcomes,
            'crossfit_estimated_conditional_mean_loss':oof,
            'crossfit_constant_baseline_mean_loss':base,
        }))
    return pd.DataFrame(summary),pd.concat(predictions,ignore_index=True)


def create_graphs(table: pd.DataFrame, binned: pd.DataFrame, figure_dir:Path):
    figure_dir.mkdir(parents=True,exist_ok=True)
    for variable, name, label in [
        ('maintenance_events_prior','maintenance_count','Prior maintenance events (M)'),
        ('observed_maintenance_span_days','history_span','Observed maintenance span, days (H)'),
        ('days_since_last_maintenance','recency','Days since latest maintenance (R)'),
    ]:
        full=table.loc[table.history_condition.eq('Full observed history')].dropna(subset=[variable])
        trend=binned.loc[binned.history_condition.eq('Full observed history') & binned.evidence_variable.eq(variable)].sort_values('bin_index')
        if trend.empty:
            continue
        fig,ax=plt.subplots(figsize=(8,4.7))
        ax.scatter(full[variable],full.sq_error,s=10,alpha=0.09,label='Individual held-out losses')
        ax.plot(trend.evidence_mean,trend.mean_squared_prediction_error_brier,
                marker='o',lw=2,label='Mean loss in quantile groups')
        ax.set_xlabel(label)
        ax.set_ylabel('Individual squared probability error / group mean')
        ax.set_title('N2 exploratory association: 7-day prediction loss')
        ax.legend(loc='best',fontsize=8)
        ax.grid(alpha=0.2)
        fig.text(0.5,0.01,'Synthetic Azure data; reused test period; no causal or factory-domain claim',
                 ha='center',fontsize=8)
        fig.tight_layout(rect=(0,0.04,1,1))
        fig.savefig(figure_dir/f'n2_{name}_vs_loss.png',dpi=180)
        plt.close(fig)

    mean=table.groupby('history_condition').sq_error.mean().reindex(CONDITIONS.keys())
    fig,ax=plt.subplots(figsize=(8,4.3))
    ax.bar(range(len(mean)),mean.values)
    ax.set_xticks(range(len(mean)),['Full','180-day','90-day'])
    ax.set_ylabel('Mean held-out squared loss (Brier Score)')
    ax.set_title('N2 pilot: separately retrained models, same test machine-time cases')
    ax.grid(axis='y',alpha=0.2)
    fig.text(0.5,0.01,'Differences combine evidence-window effects and model retraining',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,0.04,1,1))
    fig.savefig(figure_dir/'n2_window_brier_comparison.png',dpi=180)
    plt.close(fig)


def run(data_dir:Path, losses:Path, out_dir:Path, folds:int=5):
    out_dir.mkdir(parents=True,exist_ok=True)
    data = load_and_link(data_dir,losses)
    expected=3 * data.loc[data.history_condition.eq('Full observed history')].shape[0]
    if len(data)!=expected:
        raise AssertionError('History conditions should have equal machine-time sample counts.')
    if data.C_field_completeness_pct.nunique()!=1:
        raise ValueError('This first experiment assumes original unmasked Azure fields; completeness should be constant.')
    if not np.isclose(data.C_field_completeness_pct.iloc[0],100):
        raise ValueError('Expected 100% source completeness for this published benchmark.')

    associations=descriptive_associations(data)
    bins=pd.concat([grouped_loss(data,f) for f in MODEL_INPUTS],ignore_index=True)
    cv_summary,cv_predictions=grouped_machine_cv(data,folds=folds)
    data.to_csv(out_dir/'n2_prediction_evidence_and_losses.csv',index=False)
    associations.to_csv(out_dir/'n2_descriptive_spearman_correlations.csv',index=False)
    bins.to_csv(out_dir/'n2_evidence_loss_quantile_summary.csv',index=False)
    cv_summary.to_csv(out_dir/'n2_grouped_cv_relationship_metrics.csv',index=False)
    cv_predictions.to_csv(out_dir/'n2_grouped_cv_loss_predictions.csv',index=False)
    create_graphs(data,bins,out_dir/'figures')

    summary={
        'dataset':'Microsoft Azure PdM synthetic benchmark (NOT apparel factory data)',
        'prediction_horizon_days':7,
        'n_paired_machine_snapshots_per_condition':int(expected/3),
        'n_observation_rows_total':int(len(data)),
        'n_machines_per_condition':int(data.loc[data.history_condition.eq('Full observed history'),'machineID'].nunique()),
        'history_conditions':list(CONDITIONS),
        'H_definition':'Elapsed time since earliest observed maintenance event in retained window; NOT verified data collection start',
        'M_definition':'Prior recorded maintenance events in retained window',
        'F_definition':'Prior recorded failures in retained window',
        'C_definition':'Source fields complete (100% constant). Not used for fitted relationship; simulated masking needed later',
        'R_definition':'Days since most recent recorded maintenance event, within retained window; missing if no prior event',
        'loss_definition':'(observed 7-day failure label - original uncalibrated model probability)^2',
        'calibration_note':'Losses come from the ORIGINAL independently refitted N2 pilot models, not the later sigmoid-calibration experiment.',
        'relationship_estimator':'Ridge regression on H/M/F/R plus machine model; machine-grouped 5-fold out-of-fold predictions',
        'correlation_interpretation':'Spearman rho is descriptive only; repeated snapshots are correlated; no p-values reported',
        'scope_limitations':[
            'Repeated machine snapshots and overlapping outcomes; no row-wise independence',
            'Underlying original failure model trained on earlier periods of these same machines',
            'Test data inspected during earlier model development; not untouched final evaluation',
            'Brier loss can vary with failure prevalence and intrinsic case difficulty, not only sufficiency',
            'Machine-group CV assesses second-stage mapping only; not generalisation to wholly unseen assets',
            'Window models were refitted; differences do not isolate the causal effect of data removal',
            'No real-world naturally missing maintenance fields; C is unidentifiable',
        ],
    }
    (out_dir/'n2_methodology_and_limitations.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nN2 continuous analysis generated:',out_dir)
    print('\nMean squared probability error by condition:')
    print(data.groupby('history_condition').sq_error.agg(['count','mean']).round(6).to_string())
    print('\nContinuous evidence -> error model, grouped by machine:')
    print(cv_summary.round(5).to_string(index=False))
    print('\nDescriptive correlations (rho); completeness cannot be estimated:')
    print(associations[['history_condition','evidence_variable','spearman_rho_descriptive']].round(4).to_string(index=False))
    return data, associations, bins, cv_summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True,help='Folder containing original four Azure PdM CSVs')
    p.add_argument('--losses',type=Path,default=Path('outputs_corrected/n2_paired_test_prediction_losses.csv'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs_n2_continuous'))
    p.add_argument('--folds',type=int,default=5)
    a=p.parse_args()
    run(a.data_dir,a.losses,a.out_dir,folds=a.folds)

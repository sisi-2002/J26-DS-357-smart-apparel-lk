"""Render completed saved results and charts; never load or fit a model."""
from pathlib import Path
import argparse
import json
from hashlib import sha256
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame, columns):
    def value(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in frame[columns].itertuples(index=False,name=None))


def generate(folder):
    if not folder.resolve().is_relative_to(Path(__file__).resolve().parent):raise ValueError('Report target must remain inside experimental_evidence_v3')
    require_files=['summary.json','metrics.csv','paired_cluster_intervals.csv','baseline_descriptive_correlations.csv','baseline_evidence_strata.csv','evidence_distributions.csv','thinning_replicate_summary.csv','condition_audit.csv']
    if not all((folder/p).is_file() for p in require_files):raise ValueError('Incomplete run; no report can be generated')
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('schema')!='c3-evidence-v3-frozen-1':raise ValueError('Incompatible study schema')
    draws=int(json.loads((folder/'plan.json').read_text())['bootstrap_draws'])
    m=pd.read_csv(folder/'metrics.csv');ci=pd.read_csv(folder/'paired_cluster_intervals.csv')
    corr=pd.read_csv(folder/'baseline_descriptive_correlations.csv');strata=pd.read_csv(folder/'baseline_evidence_strata.csv')
    d=pd.read_csv(folder/'evidence_distributions.csv');thin=pd.read_csv(folder/'thinning_replicate_summary.csv');audit=pd.read_csv(folder/'condition_audit.csv')
    full=m[m.condition.eq('full')];trunc=m[m.family.eq('truncation')];mask=m[m.family.eq('masking')]
    lines=['# Function 2: frozen v3 multi-horizon maintenance evidence study',
           'Exploratory synthetic Microsoft Azure industrial-machine benchmark; not validated apparel-factory data. No application changes or user-facing reliability score.',
           '## What was reused and what was not',
           'The exact run_003 unpooled HistGradientBoosting conditional-hazard artifact, its fitted preprocessing and three sigmoid calibrators were loaded, never refitted. The original historical feature builder and v3 inclusive next-failure outcomes were reused. All five source hashes, raw inputs and artifact identities were verified. All 3,900 full-history probabilities matched the saved v3 CSV **exactly** and labels matched identically. Original N2 metrics/relationship models were not transferred: those used separately refitted, mainly uncalibrated seven-day LR models and different target-boundary semantics.',
           'The study separates A: descriptive evidence availability, B: frozen-model sensitivity to simulated missing history, and C: incremental error estimation. These are different questions. No evidence condition was selected/tuned using test labels.',
           '## Cohort and follow-up',
           table(pd.DataFrame(summary['split_audit']),['split','cases','dates','first','last','minimum_followup_days','incomplete_30d']),
           'Chronological train/calibration/test cohorts remain 1,800/500/1,300 landmarks. Thirty-day embargoes are checked strictly at both boundaries. Test evaluation covers the same 100 machines at all 13 dates. Component failures at the same machine/time yield one next-event time; historical F counts component-event rows. Targets describe at least one **strictly future** event with inclusive upper bounds: (0,7], (7,14], (14,30]. True labels remain unperturbed for every intervention. Observations without complete 30-day administrative follow-up cause this matched-cohort study to stop, never become artificial negatives. The assumed exclusive benchmark end is January 1, 2016; published time coverage does not prove uptime or complete recording.',
           '## Full-history reference and prevalence baselines',
           table(full,['horizon','cases','positive_cases','prevalence','brier','average_precision','roc_auc','ece_10_equal_width','calibration_bias','precision_at_10','recall_at_10','top10_captured','train_prevalence_baseline_brier','test_prevalence_descriptive_brier']),
           'Top-10 metrics count positive machine-date windows, not unique physical failures. P@10 is captured positives / 130 selected windows; Recall@10 is captured positives / all horizon positives. Ranking is probability descending, machine ID ascending for ties, separately on each date. An equal-budget random ranking expects 16.7/33.2/63.1 positive windows at 7/14/30 days. Test-prevalence Brier is a descriptive oracle constant, not a fitted deployment baseline; the separate training-prevalence baseline uses training labels only. AP is average precision, not trapezoidal PR area. ECE uses ten fixed equal-width bins and is sample/bin-sensitive; the full calibration bin counts are saved.',
           '## A — descriptive historical evidence',
           table(d[d.condition.eq('full')],['evidence','available','missing','zeros','minimum','q25','median','q75','maximum']),
           'H is elapsed span since the first previously recorded maintenance event, not log coverage; M/F/E are recorded event counts, not proof that all maintenance/failures/errors were recorded. R is days since recorded maintenance, not database freshness. ER is recorded-error recency. C is constant 100% for audited published fields and has no between-case information. C is not a factory completeness measure and is excluded from correlation interpretation. H is descriptive metadata, not one of the 18 v3 predictors; any H-loss association is indirect/confounded. Machine age/model are unchanged published benchmark metadata. Generic error1–error5 counts are supplied in the baseline dataset as descriptive categories only, never appended to model features.',
           '### Spearman associations with realized squared loss (no naive p-values)',
           table(corr[corr.scope.eq('overall')].pivot(index='evidence',columns='horizon',values='rho').reset_index().rename(columns={7:'rho_7d',14:'rho_14d',30:'rho_30d'}),['evidence','rho_7d','rho_14d','rho_30d']),
           'These associations are observational. Loss depends on outcome prevalence, model risk, machine category, maintenance scheduling, time and event-recording patterns. Even a perfectly calibrated model has risk-dependent binary-outcome variance; an individual outcome does not reveal its true underlying probability. A larger realized loss is not a validated per-machine unreliability label. Full strata tables retain actual group ranges, sample counts, prevalence, mean risk and mean loss, with ties preserved and missing values separate. Quantile bins are descriptive, not sufficiency thresholds.',
           '### Within-date direction sensitivity',
          ]
    directions=[]
    for (e,h),g in corr[corr.scope.eq('prediction_time')].groupby(['evidence','horizon']):
        valid=g.rho.dropna();directions.append(dict(evidence=e,horizon=h,dates=len(valid),positive=int((valid>0).sum()),negative=int((valid<0).sum()),minimum=valid.min(),maximum=valid.max()))
    lines += [table(pd.DataFrame(directions),['evidence','horizon','dates','positive','negative','minimum','maximum']),
              'Per-machine-model correlations are also saved. Date-by-date signs are sensitivity descriptions, not independent replications or causal adjustment. Only 13 shared dates and 100 repeated machines are available.',
              '## B — frozen-model evidence degradation',
              'The predeclared 40 conditions comprise full history; 180/90-day truncation of all event streams, maintenance only or errors only; 25/50/75% expected event deletion for maintenance and errors separately, each with five fixed seeds (11,29,47,71,97); and three numeric-feature masks. There is no hyperparameter search or condition selection. All features are rebuilt by the existing machine_features routine after strict-past filtering; schema and keys are preserved.',
              'Deletion is a persistent record-keyed seeded uniform mask shared across later snapshots, not independently redrawn per machine-date. Percentages denote expected deletion probabilities; actual losses of prior event windows are audited. Higher deletion levels are nested within a seed. Future records never enter an intervention, and the original failures table still supplies all true future targets. Counts may become zero because records were removed; this is **simulated record absence**, not proof no events happened.',
              '### B1 — history truncation',
              table(trunc,['condition','horizon','brier','delta_brier','average_precision','delta_average_precision','ece_10_equal_width','precision_at_10','recall_at_10','top10_captured','mean_absolute_probability_change']),
              'All-stream truncation also reduces historical failure/error features. Maintenance-only and error-only truncation isolate each source in the feature inputs. Differences in frozen-model predictions do not demonstrate that shorter real histories are better, nor replicate the older refitted LR history experiment.',
              '### B2 — independent thinning replicates (not pooled samples)',
              table(thin,['stream','removal','horizon','replicates','mean_brier','min_brier','max_brier','mean_ap','min_ap','max_ap','mean_precision10','mean_recall10']),
              'Each replicate evaluates the original 1,300 cases; five replicates are not 6,500 independent observations. Min/max across seeds describe simulation variability, not confidence intervals. Full per-seed metrics and paired machine/date uncertainty are retained, including negative/worse results.',
              '### B3 — supported numeric masking',
              table(mask,['condition','horizon','brier','delta_brier','average_precision','delta_average_precision','ece_10_equal_width','precision_at_10','recall_at_10','top10_captured']),
              'Masking sets the specified fitted numeric feature block to NaN; it does not remove required columns, set missing counts to zero or create a new imputation rule. The existing per-interval training-median imputers and their already fitted missing indicators are used exactly. Maintenance masking covers its five count columns plus recency; error masking covers four count columns plus recency; recency masking covers maintenance/failure/error recency. H and C metadata are not model inputs. C continues to describe the original published fields; it is not reinterpreted as a simulated completeness score. available_*_records columns preserve actual retained record counts even when feature inputs are masked. These are artificial distribution-shift probes, not natural CMMS missingness. Exact medians/indicator support appear in frozen_imputation_audit.csv.',
              '### Probability and sparse-history audit',
              table(audit,['condition','maintenance_events_removed','error_events_removed','zero_prior_maintenance','zero_prior_errors']),
              f'All {summary["conditions"]} conditions retained all 1,300 keys and identical labels. Total ordering violations: **{summary["ordering_violations_total"]}**; all predictions were finite and within [0,1]. The fitted-state hash before/after was identical, and every protected input/artifact hash was unchanged.',
              '## Paired uncertainty and stability',
              'Differences below are condition minus full history: negative Brier favors the condition; positive AP/P@10/Recall@10 favors it. Separate 1,000-draw machine-cluster and date-cluster bootstraps reuse identical weights for baseline and condition. Machine draws retain entire observed trajectories; date draws retain complete 100-machine snapshots. These are two sensitivity views, not a two-way cluster guarantee. They condition on the trained model and each fixed deletion pattern and omit training/calibration estimation uncertainty.',
              'For ranking uncertainty, retain each original per-date Top-10 membership and bootstrap weighted captured/flagged/positive window ratios. Machine-cluster intervals therefore assess the existing ranking contribution across machines; they do not rerank a synthetic population containing duplicate physical machines. Date-cluster resampling duplicates whole original ranked snapshots. AP is recomputed with resampling weights and exact score-tie handling. Ratio denominators and valid draw counts are saved. No p-values or operational significance claims are made.',
              table(ci[(ci.metric=='brier')&ci.condition.isin(['all_180d','all_90d','maintenance_90d','errors_90d','mask_maintenance','mask_errors'])],['condition','horizon','cluster','clusters','delta','low','high','valid_draws']),
              'The full paired_cluster_intervals.csv includes every condition and Brier/AP/P@10/Recall@10. Many contrasts and only 13 dates make apparent interval exclusions exploratory; a narrow conditional interval is not external validation.',
              '## C — incremental evidence value: not estimable under this frozen-release protocol',
              f'Training predictions are in-sample for the fitted base models. All 500 later validation cases participated in sigmoid calibration, so their calibrated losses are not independently out-of-sample. Training a secondary estimator on the 1,300 repeatedly inspected test labels is prohibited. A secondary training snapshot must be after {summary["secondary"]["secondary_time_must_be_after"]} to protect calibration outcomes, yet before {summary["secondary"]["secondary_time_must_be_before"]} to protect the test boundary by 30 days. There are **zero eligible days**. Even the shortest horizon cannot fit a complete protected common period under this rule.',
              'Consequently C0 risk-only and C1 risk-plus-evidence loss-estimation MAE/RMSE are **not reported or fabricated**; zero secondary estimators were fitted. The earlier N2 GroupKFold-on-test approach is not reused as independent model-selection evidence. This is an availability/design limitation, not an observed negative C1 result. Descriptive associations in A cannot establish incremental benefit beyond risk alone.',
              'To estimate C defensibly, reserve a genuinely separate post-calibration secondary-development period and then a fresh later evaluation period with embargoes, or separately authorize a fully nested chronological OOF study with base fitting and calibration strictly earlier than each OOF target window. Such models would be newly refitted research models, not the frozen artifact audited here. Lock the secondary feature set/regularization using earlier development only. Do not use current test outcomes to tune it.',
              '## Recommendation and scientific interpretation',
              'Retain **descriptive maintenance-evidence display only**. No reliability label, confidence percentage, sufficiency threshold, repair priority or automatic maintenance recommendation is justified. C is uninformative here; associations and degradation sensitivity are different from evidence quality, probability calibration and model uncertainty. Strong sensitivity can motivate validation/data audits but cannot measure real-world missingness or prove causation.',
              'The tables show separately which sources/horizons worsen or improve under each intervention. Performance changes must be interpreted on matched populations and alongside AP/calibration/ranking, not selected solely for attractive Brier values. Seed ranges and both clustered intervals expose instability. A field/record intervention changes the information supplied to a fixed model, not the underlying physical failure process.',
              'Before operational use: verify real factory event semantics, coverage and natural missingness; gather fresh prospective machine/date data; validate per-horizon calibration/ranking and costs/capacity; establish independent secondary training/evaluation periods; assess generalization across machines/sites; and quantify uncertainty beyond this repeatedly reused synthetic benchmark.',
              '## Reproduction and files',
              f'Runtime: {summary["run_seconds"]:.1f} seconds. Installed versions: '+json.dumps(summary['versions'])+'. No dependencies installed. All source/data/model identities are in summary.json. The predeclared plan and exact seeds are in plan.json. Numeric CSVs use full precision; Markdown rounds display only.',
              'Run commands appear in ../README.md. Use a NEW run directory; existing outputs are never overwritten by run_study.py. run_001 is an explicitly incomplete export attempt (failure.json); it stopped before perturbation evaluation. This report uses only this completed directory.',
              'Generated charts: history_brier.png, thinning_brier.png, evidence_loss.png and calibration.png. They show observed points/bars without fitted smoothing or invented significance markers; all Brier/loss axes begin at zero. Numerical source tables retain group ranges/counts, bin counts and all results. Nothing was integrated into FastAPI, React, existing N2 or deployed inference.'
             ]
    # Interpret all prespecified comparisons, without promoting a test-selected policy.
    narrative=['## Answers from the executed results']
    def above_counts(stream,cluster):
        subset=ci[(ci.metric=='brier')&ci.condition.str.startswith(stream+'_drop75_')&ci.cluster.eq(cluster)]
        return '/'.join(str(int((subset[subset.horizon.eq(h)].low>0).sum())) for h in [7,14,30])
    for stream in ['maintenance','errors']:
        part=thin[(thin.stream==stream)&(thin.removal==.75)].sort_values('horizon')
        narrative.append(f'**75% {stream} deletion:** mean Brier across five seeds at 7/14/30 days was '+
                         '/'.join(f'{v:.6f}' for v in part.mean_brier)+
                         '; mean AP was '+ '/'.join(f'{v:.6f}' for v in part.mean_ap)+
                         '. These are paired with the same baseline cases, not additional independent samples.')
    all90=m[m.condition.eq('all_90d')].sort_values('horizon')
    narrative += ['**History truncation:** joint 90-day Brier was '+ '/'.join(f'{v:.6f}' for v in all90.brier)+
                  '. Seven- and fourteen-day reductions were small; both machine/date sensitivity intervals include zero for those contrasts. The 30-day Brier worsened, while captured Top-10 positives changed from 89 to 95. At 14 days captured positives fell from 59 to 53 despite a higher AP. Maintenance-only 90-day truncation worsened Brier at every horizon. These contrasting endpoints do not justify discarding older maintenance history.',
                  '**Field availability:** maintenance masking changed 7-day Brier from 0.092593 to 0.154898, AP from 0.354306 to 0.204407 and ECE from 0.017403 to 0.198581. At 30 days, masking all recencies increased AP/captured Top-10 windows (0.640906/101 versus 0.631971/89) while worsening Brier (0.236152 versus 0.233211). No masking policy is selected from these reused test outcomes.',
                  '**Associated evidence characteristics:** maintenance recency R had pooled seven-day Spearman rho 0.661501, with positive within-date rho on all 13 dates (range 0.423558–0.740557). Historical failure count F had fourteen-day rho 0.315107, also positive on all 13 dates (range 0.089432–0.451616). H associations were weak (0.015803/−0.017247/0.031178). R changed to weak, mixed-sign associations at 14/30 days; these do not define a universal recency or sufficiency rule.',
                  '**Stability:** numbers of five 75%-maintenance-deletion replicates with Brier-difference intervals wholly above zero were '+above_counts('maintenance','machineID')+' under machine clustering and '+above_counts('maintenance','prediction_time')+' under date clustering. For 75%-error deletion the corresponding counts were '+above_counts('errors','machineID')+' and '+above_counts('errors','prediction_time')+', at 7/14/30 days. Effects of mild error loss and small history-window differences were less stable. These conditional bootstrap patterns do not establish statistical significance or factory generalization.',
                  '**Incremental error estimation:** C0 versus C1 was not estimable under the permitted timeline, so no improvement or deterioration beyond risk alone has been demonstrated. This differs from an executed negative finding. Descriptive evidence display is defensible; a learned per-machine confidence/reliability indicator is not.']
    # Place the direct answers before the longer numerical/methodological record.
    lines[2:2]=narrative
    stability=ci[(ci.metric=='brier')&ci.condition.str.contains('_drop')].copy()
    stability['stream']=stability.condition.str.split('_drop').str[0]
    stability['removal_pct']=stability.condition.str.extract(r'_drop(\d+)_').astype(int)
    stability['interval_above_zero']=stability.low>0
    stability.groupby(['stream','removal_pct','horizon','cluster']).agg(replicates=('low','size'),intervals_above_zero=('interval_above_zero','sum')).reset_index().to_csv(folder/'thinning_cluster_stability.csv',index=False)
    lines += ['condition_audit event-removal counts are **removed record appearances across repeated machine-date histories**, not unique source-event deletions. The same historical record can contribute to many landmarks. Removal-rate labels refer to the seeded Bernoulli probability; zero-record counts refer to snapshots. C is unchanged published-field completeness throughout.']
    (folder/'report.md').write_text(('\n\n'.join(lines)+'\n').replace('1,000',f'{draws:,}'),encoding='utf-8')
    plot(folder,m,strata,pd.read_csv(folder/'calibration_bins.csv'),thin)
    provenance={str(p):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*folder.glob('*.csv'),folder/'summary.json',folder/'plan.json']}
    (folder/'report_provenance.json').write_text(json.dumps(provenance,indent=2))


def plot(folder,m,strata,bins,thin):
    colors=['#087F8C','#526D89','#A34E40']
    names=['full','all_180d','all_90d','maintenance_180d','maintenance_90d','errors_180d','errors_90d']
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for h,ax in zip([7,14,30],axes):
        d=m[m.horizon.eq(h)].set_index('condition').loc[names]
        ax.bar(range(len(d)),d.brier,color=['#142D4E']+['#087F8C']*6)
        ax.set_xticks(range(len(d)),names,rotation=55,ha='right');ax.set(title=f'{h}-day Brier',ylabel='Brier score (zero baseline)',ylim=(0,max(d.brier)*1.15))
    fig.suptitle('Frozen v3: matched 1,300 Azure machine-date cases; no refitting');fig.tight_layout();fig.savefig(folder/'history_brier.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for h,ax in zip([7,14,30],axes):
        for stream,color,offset in [('maintenance',colors[0],-.8),('errors',colors[1],.8)]:
            d=thin[(thin.horizon==h)&(thin.stream==stream)].sort_values('removal')
            ax.errorbar(d.removal*100+offset,d.mean_brier,yerr=[d.mean_brier-d.min_brier,d.max_brier-d.mean_brier],fmt='o',capsize=4,color=color,label=stream+' (5 seeds, min–max)')
        base=m[(m.condition=='full')&(m.horizon==h)].brier.iloc[0];ax.scatter([0],[base],color='#142D4E',marker='s',label='full history')
        ax.set(title=f'{h} days',xlabel='Expected records removed (%)',ylabel='Brier score (zero baseline)',ylim=(0,None),xticks=[0,25,50,75]);ax.legend(fontsize=7)
    fig.suptitle('Frozen model record-thinning sensitivity; intervals are seed ranges, not confidence intervals');fig.tight_layout();fig.savefig(folder/'thinning_brier.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    titles={'H':'Observed maintenance span (H, days)','M':'Maintenance records (M)',
            'F':'Historical failure records (F)','E':'Historical error records (E)',
            'R':'Maintenance recency (R, days)','ER':'Error recency (days)'}
    for evidence,ax in zip(['H','M','F','E','R','ER'],axes.flat):
        for h,color in zip([7,14,30],colors):
            d=strata[(strata.evidence==evidence)&(strata.horizon==h)&(strata.group!='missing')]
            x=(d.evidence_min+d.evidence_max)/2
            ax.scatter(x,d.mean_squared_loss,s=48,color=color,label=f'{h} days')
        ax.set(title=titles[evidence], xlabel='Group midpoint (ranges/counts in CSV)',ylabel='Observed mean squared loss',ylim=(0,None));ax.legend(fontsize=8)
    fig.suptitle('Baseline descriptive associations only: no fitted trend, causal claim or sufficiency threshold');fig.tight_layout();fig.savefig(folder/'evidence_loss.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for h,ax in zip([7,14,30],axes):
        for name,color in zip(['full','all_90d','mask_maintenance','mask_errors'],['#142D4E',*colors]):
            d=bins[(bins.horizon==h)&(bins.condition==name)&(bins['count']>0)]
            ax.scatter(d.mean_probability,d.observed_rate,s=20+np.sqrt(d['count'])*2,label=name,color=color)
        ax.plot([0,1],[0,1],'--',color='gray',linewidth=1);ax.set(title=f'{h} days',xlabel='Mean probability in fixed bin',ylabel='Observed rate',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
    fig.suptitle('Calibration diagnostics: marker size reflects bin count; only 1,300 reused benchmark cases');fig.tight_layout();fig.savefig(folder/'calibration.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);generate(parser.parse_args().folder)

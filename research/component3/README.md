\# Component 3 — Reliability-Aware Predictive Maintenance



\*\*Project:\*\* SMART APPAREL LK (J26-DS-357)



\## Research Objective



Develop and evaluate short-horizon machine failure-risk prediction using historical maintenance records and investigate the relationship between available maintenance evidence and predictive reliability.



\## Function 1 — Machine Failure-Risk Prediction



Implemented:

\- Historical maintenance, failure and error feature extraction

\- 7-day, 14-day and 30-day binary failure prediction

\- Logistic Regression baseline

\- Chronological training, validation and testing

\- Validation-based alert threshold selection

\- Comparison against a prevalence baseline

\- Preliminary probability calibration experiments



\## Function 2 — Maintenance-Evidence Sufficiency



Implemented:

\- Controlled full-history, 180-day and 90-day experiments

\- Separate model training for each history condition

\- Comparison of held-out prediction performance



In progress:

\- Continuous evidence–prediction-error relationship analysis

\- Field completeness and event thinning experiments



\## Dataset



Microsoft Azure Predictive Maintenance benchmark.



This is \*\*synthetic industrial data\*\*, not actual apparel-factory machinery data. Results demonstrate preliminary methodology and cannot establish deployment performance in apparel factories.



Required local files:

\- `PdM\_machines.csv`

\- `PdM\_maint.csv`

\- `PdM\_failures.csv`

\- `PdM\_errors.csv`



\## Reproducibility



Install dependencies using `requirements.txt`.



Run commands from the `research/component3` directory:



```bash

python scripts/test\_prediction\_time\_boundary.py

python scripts/build\_and\_evaluate.py --data-dir PATH\_TO\_RAW\_DATA --out-dir outputs\_corrected

python scripts/n2\_history\_pilot.py --data-dir PATH\_TO\_RAW\_DATA --out-dir outputs\_corrected

python C3\_calibration\_validation\_experiment.py --data-dir PATH\_TO\_RAW\_DATA --out-dir outputs\_calibration

```



Replace `PATH\_TO\_RAW\_DATA` with the local directory containing the four source CSV files.



\## Important Limitations



\- No real apparel-machine validation yet

\- Repeated observations from the same machines

\- Initial N2 experiments do not establish a continuous evidence–reliability relationship

\- No natural field missingness in the audited benchmark fields

\- Model performance is preliminary and should not be interpreted as factory-ready

\- API and frontend integration are future implementation steps



\## Development Status



PP1 research proof-of-concept implementation. Further model evaluation, testing, evidence analysis and integration are in progress.
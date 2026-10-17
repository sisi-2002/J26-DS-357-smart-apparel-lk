# Component 2 – Research

This folder holds the research code for Smart Apparel-LK component 2: dataset preparation, preprocessing, model training, and the source-sufficiency, style-novelty, high-waste-tail and marker experiments, plus evaluation and the reports and figures that serve as assessment evidence. Raw factory data lives in the repo-root `data/` folder and is never committed.

Set up on Windows (PowerShell), from `research/component2`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env   # then fill in local values
pytest
```

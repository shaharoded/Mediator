"""Create a reviewable report and vector plot from the aggregate audit outputs."""
from pathlib import Path
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = RESULTS_ROOT / 'threshold-proximity'
ROOT.mkdir(parents=True, exist_ok=True)
states = pd.read_csv(ROOT/'state_summary.csv')
trends = pd.read_csv(ROOT/'trend_summary.csv')
bands = pd.read_csv(ROOT/'state_boundary_bands.csv')
meta = json.loads((ROOT/'metadata.json').read_text())
db_validation = json.loads((ROOT/'database_validation.json').read_text()) if (ROOT/'database_validation.json').exists() else None

def weighted(frame, denominator, column):
    return np.average(frame[column], weights=frame[denominator])

lines = ['# TAK threshold-proximity audit', '',
         f'Scope: {meta["matched_admissions"]:,} admissions, numeric observations '
         '(physiological measurements and insulin doses) within 0–48 hours of admission, '
         'pooled over all train/validation/test admissions. Data source: the local '
         'mimic-dataset/output/concept_events.csv export, not a MySQL query. '
         'Relationships and input aliases are resolved from the saved Mediator TAKRepository.', '',
         'The repository contains 174 TAKs and 26 eligible direct scalar numeric state concepts. '
         '25 have valid observations in this window; serum ketones have none. 22 of the observed '
         'concepts also have trend definitions; three insulin-dose state concepts do not. '
         'Numeric inputs without state cutoffs are outside this analysis. The derived M-SHR '
         'ratio is separately identified in repository_scope.json; its transformation is '
         'not audited as a direct raw-measurement cutoff.', '',
         'This measures proximity to decision boundaries. It does not perturb observations, '
         'rebuild intervals, evaluate missingness, or establish downstream model robustness.', '',
         'State bands use ±1%, ±2.5%, and ±5% of the smaller adjacent cutoff spacing; '
         'outer cutoffs use the nearest interior spacing. These are scale-normalized geometric '
         'bands, not assumed assay-error bounds. Exact matches use a numerical tolerance of 1e-10.', '',
         'Trend bands use the same fractions of the TAK significant-variation threshold, '
         'around either positive or negative OLS-derived change over the TAK lookback window. '
         'Windows with fewer than two distinct times are excluded from the trend-band denominator '
         'and counted separately; Mediator classifies them as Steady.', '',
         '## Aggregate counts', '',
         f'- State measurements: {states.observations.sum():,} across {len(states)} concepts.',
         f'- Eligible trend windows: {trends.eligible_windows.sum():,} across {len(trends)} concepts.',
         f'- Insufficient/degenerate trend windows: {trends.insufficient_or_degenerate.sum():,}.',
         f'- State observations exactly at a cutoff: {weighted(states,"observations","exact_boundary_pct"):.2f}%.', '']
for f in (.01, .025, .05):
    col = f'near_{f:g}_pct'
    lines.append(f'- Within the ±{100*f:g}% band: '
                 f'{weighted(states,"observations",col):.2f}% of state measurements; '
                 f'{weighted(trends,"eligible_windows",col):.2f}% of eligible trend windows.')
lines += ['', 'Pooled percentages weight frequent measurements more heavily. '
          'Concept-level values below preserve differences across variables.', '',
          '## Concept-level results', '',
          '| Concept | State observations | Exactly on cutoff (%) | State ±5% (%) | '
          'Eligible trend windows | Trend ±5% (%) |',
          '|---|---:|---:|---:|---:|---:|']
merged = states.merge(trends[['concept','eligible_windows','near_0.05_pct']],
                       on='concept', suffixes=('_state','_trend'), how='left')
for _, r in merged.iterrows():
    trend_count = f'{r.eligible_windows:,.0f}' if pd.notna(r.eligible_windows) else '—'
    trend_pct = f'{r["near_0.05_pct_trend"]:.2f}' if pd.notna(r['near_0.05_pct_trend']) else '—'
    lines.append(f'| {r.concept} | {r.observations:,} | {r.exact_boundary_pct:.2f} | '
                 f'{r["near_0.05_pct_state"]:.2f} | {trend_count} | {trend_pct} |')
lines += ['', '## Boundary density comparison', '',
          'state_boundary_bands.csv compares the central interval [b−w,b+w] with '
          'the two adjacent intervals [b−3w,b−w) and (b+w,b+3w]. Each interval has '
          'the same width 2w. The ratio is central count divided by the mean of the two '
          'flank counts. A ratio above one indicates greater local concentration, not '
          'a robustness failure by itself. Discrete measurement precision can strongly '
          'affect these ratios. Zero flank counts leave the ratio undefined.', '',
          '## Reproduction and validation', '',
          'First run export_repository.py with the Mediator virtual environment to resolve '
          'the saved TAKRepository object and full dependency graph. Run analyze.py '
          'with Python, pandas, and NumPy; then report.py with Matplotlib. '
          'validate_trends.py uses the local Mediator virtual environment to compare '
          'classification against its actual Trend._compute_trends implementation. '
          'All 11,500 irregularly sampled synthetic positions agreed across 23 concepts; '
          'analyze.py additionally checked 3,000 actual-data OLS calculations against '
          'the direct dot-product formula. Metadata records paths, repository hash, scope, '
          'and excluded rows. No patient-level outputs are written.', '',
          'No manuscript changes, commits, or pushes were performed for this analysis.']
if db_validation:
    lines += ['', '## Database agreement', '',
              f'The configured Mediator database is SQLite, not MySQL. All '
              f'{db_validation["database_rows"]:,} InputPatientData records matched '
              'the CSV exactly in PatientId, ConceptName, StartDateTime, EndDateTime, '
              f'and Value; mismatching rows: {db_validation["mismatching_rows"]}. '
              'The comparison covered the entire export, not a sample or aggregate-count check. '
              'The database was accessed in read-only mode. See database_validation.json.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

display = {'ALANINE-AMINOTRANSFERASE_MEASURE':'ALT',
           'ASPARATE-AMINOTRANSFERASE_MEASURE':'AST',
           'BASAL_BITZUA':'Basal insulin dose', 'BOLUS_BITZUA':'Bolus insulin dose',
           'INSULIN_BITZUA':'Insulin dose',
           'BLOOD_PRESSURE_DIASTOLIC_MEASURE':'Diastolic blood pressure',
           'BLOOD_PRESSURE_SYSTOLIC_MEASURE':'Systolic blood pressure',
           'CREATININE_SERUM_MEASURE':'Serum creatinine',
           'INFECTION_WBC_MEASURE':'White blood cell count',
           'PLT_MEASURE':'Platelet count', 'PH_MEASURE':'pH', 'E-GFR_MEASURE':'eGFR'}
labels = [display.get(c, c.replace('_MEASURE','').replace('_',' ').capitalize())
          for c in merged.concept]
y = np.arange(len(merged))
fig, axes = plt.subplots(1,2,figsize=(10,9.5),sharey=True,layout='constrained')
axes[0].barh(y, merged['near_0.05_pct_state'], color='#236A91', label='Within ±5% band')
axes[0].barh(y, merged.exact_boundary_pct, color='#94BBCF', label='Exactly on cutoff')
axes[1].barh(y, merged['near_0.05_pct_trend'], color='#79539B')
for i in np.flatnonzero(merged['near_0.05_pct_trend'].isna()):
    axes[1].text(.05, i, 'No trend definition', va='center', fontsize=9, color='#627080')
axes[0].set_yticks(y, labels, fontsize=11)
axes[0].invert_yaxis()
axes[0].set_title('State cutoffs',fontweight='bold',fontsize=12)
axes[1].set_title('Trend change thresholds',fontweight='bold',fontsize=12)
axes[0].set_xlabel('Numeric observations within band (%)',fontsize=11)
axes[1].set_xlabel('Eligible trend windows within band (%)',fontsize=11)
axes[0].legend(loc='lower right',fontsize=9)
for ax in axes:
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'threshold_proximity.pdf')
fig.savefig(ROOT/'threshold_proximity.png',dpi=150)
print('\n'.join(lines[12:24]))

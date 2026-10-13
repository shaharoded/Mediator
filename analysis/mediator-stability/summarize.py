"""Aggregate matched interval comparisons; bootstrap admission-level sampling units."""
from pathlib import Path
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import json
import numpy as np
import pandas as pd

ROOT = RESULTS_ROOT / 'mediator-stability'
ROOT.mkdir(parents=True, exist_ok=True)
METRICS={'labeled_interval_agreement':('same_label_seconds','union_seconds'),
         'label_agreement_on_shared_coverage':('same_label_seconds','overlap_seconds'),
         'baseline_coverage_retained':('overlap_seconds','baseline_seconds'),
         'altered_to_baseline_coverage':('altered_seconds','baseline_seconds')}
LABELS={'noise_1pct':'Value perturbations: ±1% of local cutoff spacing',
        'noise_5pct':'Value perturbations: ±5% of local cutoff spacing',
        'remove_5pct':'Random observation removal: 5%',
        'remove_10pct':'Random observation removal: 10%'}


def main():
    data=pd.read_csv(ROOT/'patient_metrics.csv')
    meta=json.loads((ROOT/'metadata.json').read_text())
    sample=pd.read_csv(ROOT/'sample_summary.csv')
    numeric=['baseline_seconds','altered_seconds','overlap_seconds','same_label_seconds','union_seconds']
    # Average the two realizations within each admission, preserving paired sampling.
    paired=data.groupby(['sample_index','stratum','weight','condition','family'],as_index=False)[numeric].mean()
    rng=np.random.default_rng(20261012)
    output=[]
    for (condition,family),group in paired.groupby(['condition','family'],sort=True):
        group=group.reset_index(drop=True)
        strata=[g.index.to_numpy() for _,g in group.groupby('stratum',sort=True)]
        values=group[numeric].to_numpy()*group.weight.to_numpy()[:,None]
        columns={column:i for i,column in enumerate(numeric)}
        draws=np.vstack([np.concatenate([rng.choice(indices,len(indices),replace=True)
                                         for indices in strata]) for _ in range(2000)])
        boot=values[draws].sum(axis=1)
        total=values.sum(axis=0)
        for name,(num,den) in METRICS.items():
            estimate=total[columns[num]]/total[columns[den]]
            ratios=boot[:,columns[num]]/boot[:,columns[den]]
            low,high=np.quantile(ratios,[.025,.975])
            output.append(dict(condition=condition,family=family,metric=name,percentage=100*estimate,
                               ci_low=100*low,ci_high=100*high,admissions=len(group)))
    summary=pd.DataFrame(output)
    summary.to_csv(ROOT/'results.csv',index=False)
    concepts=pd.read_csv(ROOT/'concept_metrics.csv')
    concepts['weighted_same']=concepts.same_label_seconds*concepts.weight
    concepts['weighted_union']=concepts.union_seconds*concepts.weight
    concept_summary=concepts.groupby(['concept','condition','family'],as_index=False).agg(
        weighted_same=('weighted_same','sum'),weighted_union=('weighted_union','sum'),
        sample_admissions=('sample_index','nunique'))
    concept_summary['labeled_interval_agreement_pct']=100*concept_summary.weighted_same/concept_summary.weighted_union
    concept_summary.to_csv(ROOT/'concept_results.csv',index=False)
    lines=['# Sampled Mediator abstraction-stability study','',
           f'Completed: {len(sample)} admissions, {int(sample.observations.sum()):,} valid direct numeric '
           f'observations in the first 48 hours; {int(sample.baseline_state_intervals.sum()):,} baseline '
           f'state intervals and {int(sample.baseline_trend_intervals.sum()):,} baseline trend intervals.', '',
           '## Study design','',
           'Admissions were selected with a fixed seed across joint quartiles of first-48-hour '
           'numeric input counts and recorded admission-to-terminal-event span. The latter is '
           'not necessarily hospital length of stay because a post-discharge death can replace '
           'the RELEASE event. Each of 16 strata contributed 31 or 32 admissions. Stratum '
           'population/sample ratios weight the aggregate results to the cohort.', '',
           'The saved TAKRepository resolved raw inputs, aliases, and state/trend definitions. '
           'Actual RawConcept.apply, State.apply, Trend.apply, and the Mediator global-clipping '
           'method ran in memory; the original database was read-only. State persistence, '
           'interpolation and merging, and trend regression and interval construction were '
           'included. Derived concepts and other abstraction families were not evaluated.', '',
           'For each condition, two independent random realizations were averaged within '
           'admissions. Values received zero-mean uniform additive perturbations bounded by '
           '±1% or ±5% of the smaller cutoff spacing adjacent to the nearest state cutoff. '
           'These controlled scales are not clinically validated assay-error distributions. '
           'Perturbed values outside raw validity limits were handled by the unchanged Mediator '
           'filtering rules, without artificial clipping. Observation removal used independent '
           'Bernoulli probabilities of 5% or 10%; admission and terminal markers were preserved.', '',
           'Baseline and altered abstractions used only observations in the first 48 hours; '
           'output intervals were restricted to the same admission/terminal/48-hour window. '
           'Unchanged-input reruns of the first three sampled admissions agreed exactly.', '',
           '## Metrics','',
           '- Labeled interval agreement: duration with matching labels divided by the '
           'duration covered by either baseline or altered output, summed over patient-concept '
           'pairs. It penalizes changed labels and lost or added coverage; time absent from '
           'both outputs contributes neither agreement nor disagreement.',
           '- Shared-coverage label agreement: matching-label duration divided by overlap duration.',
           '- Baseline coverage retained: overlap duration divided by baseline duration.',
           '- Altered/baseline coverage: total altered duration divided by baseline duration; '
           'values above 100% indicate increased coverage.', '',
           '95% intervals use 2,000 admission bootstrap resamples within sampling strata. '
           'Intervals characterize uncertainty over admissions, conditional on the two '
           'perturbation realizations; they are not model-training confidence intervals.', '',
           '## Results','',
           '| Condition | Family | Labeled interval agreement (%) | Shared-coverage label agreement (%) | '
           'Baseline coverage retained (%) | Altered/baseline coverage (%) |',
           '|---|---|---:|---:|---:|---:|']
    for condition in LABELS:
        for family in ['state','trend']:
            cells=[]
            for metric in METRICS:
                r=summary[(summary.condition==condition)&(summary.family==family)&(summary.metric==metric)].iloc[0]
                cells.append(f'{r.percentage:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]')
            lines.append('| '+LABELS[condition]+' | '+family+' | '+' | '.join(cells)+' |')
    def estimate(condition, family, metric):
        return float(summary[(summary.condition==condition)&(summary.family==family)&
                             (summary.metric==metric)].iloc[0].percentage)
    lines+=['', '## Interpretation', '',
            f'Under the larger bounded value perturbation, pooled interval agreement was '
            f'{estimate("noise_5pct", "state", "labeled_interval_agreement"):.2f}% for states and '
            f'{estimate("noise_5pct", "trend", "labeled_interval_agreement"):.2f}% for trends. '
            'Per-concept variation and support counts are provided in concept_results.csv.', '',
            f'Under 10% random removal, trend interval agreement was '
            f'{estimate("remove_10pct", "trend", "labeled_interval_agreement"):.2f}%, while baseline '
            f'coverage retained was {estimate("remove_10pct", "trend", "baseline_coverage_retained"):.2f}% '
            f'and label agreement on shared coverage was '
            f'{estimate("remove_10pct", "trend", "label_agreement_on_shared_coverage"):.2f}%. '
            'Thus, the interval-level result reflects both coverage loss and label changes. '
            'These distinct metrics should not be presented interchangeably.']
    lines+=['','## Scope of the conclusion','',
            'This directly measures state/trend stability under the specified bounded '
            'perturbations and independent random observation removal in the sampled cohort. '
            'It does not establish robustness to systematic missingness, assay-specific '
            'noise, other institutions, or downstream model predictions. Per-concept results '
            'and their admission counts are in concept_results.csv; sparse concepts should '
            'not be interpreted as strongly validated by pooled results.', '',
            'No manuscript changes, commits, or pushes have been made for this study.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proposed=['# Proposed addition to the threshold-proximity appendix', '',
              'After the full-cohort threshold-proximity paragraph, add the following '
              'sampled stability analysis and a compact table. This replaces the blanket '
              'statement that perturbation and missingness sensitivity remain unevaluated.', '',
              'We additionally evaluated abstraction stability on 500 admissions sampled '
              'across joint quartiles of observation count and recorded terminal-event span. '
              'The unchanged Mediator state/trend implementations were rerun on the first '
              '48 hours after bounded value perturbations (uniform ±1% or ±5% of local '
              'state-cutoff spacing) or independent removal of 5% or 10% of numeric '
              'observations, with two random realizations per condition. We compared '
              'the duration carrying the same clinical label with the union of baseline '
              'and altered interval coverage, thereby accounting for label changes and '
              'lost or added intervals. Results were weighted by sampling strata; 95% '
              'intervals used 2,000 admission-level bootstrap resamples.', '',
              '| Condition | State interval agreement (%) | Trend interval agreement (%) |',
              '|---|---:|---:|']
    for condition in LABELS:
        cells=[]
        for family in ['state','trend']:
            r=summary[(summary.condition==condition)&(summary.family==family)&
                      (summary.metric=='labeled_interval_agreement')].iloc[0]
            cells.append(f'{r.percentage:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}]')
        proposed.append('| '+LABELS[condition]+' | '+' | '.join(cells)+' |')
    proposed+=['', 'These results characterize stability under the specified small '
               'perturbations and random observation removal; assay-specific noise and '
               'structured missingness were not evaluated. No downstream robustness '
               'claim is made.', '',
               'Suggested main-text reference: “Appendix [reference] reports threshold '
               'proximity and sampled state/trend stability under controlled value '
               'perturbations and random observation removal.”']
    (ROOT/'manuscript-proposal.md').write_text('\n'.join(proposed)+'\n',encoding='utf-8')
    print('\n'.join(lines[lines.index('## Results'):lines.index('## Scope of the conclusion')]))


if __name__=='__main__': main()

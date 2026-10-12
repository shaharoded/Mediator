"""Read-only TAK threshold exposure audit; aggregate outputs contain no patient IDs.

State margins use the smaller adjacent cutoff spacing, not a clinical noise model.
Trend margins use the TAK significant-variation threshold. Scope: numeric raw
measurements within the first 48 hours of admissions in the local context table.
No intervals, predictions, model weights, or paper files are modified.
"""
from pathlib import Path
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import json
import re
import numpy as np
import pandas as pd

KB = MEDIATOR_ROOT / 'core/knowledge-base'
SOURCE = PERSONAL / 'mimic-dataset/output/concept_events.csv'
OUT = RESULTS_ROOT / 'threshold-proximity'
OUT.mkdir(parents=True, exist_ok=True)
FRACTIONS = (0.01, 0.025, 0.05)


def seconds(text):
    match = re.fullmatch(r'([\d.]+)([smhd])', text)
    if not match:
        raise ValueError(text)
    return float(match[1]) * dict(s=1, m=60, h=3600, d=86400)[match[2]]


def definitions():
    scope = OUT / 'repository_scope.json'
    if not scope.exists():
        raise RuntimeError('Run export_repository.py with the Mediator environment first')
    return json.loads(scope.read_text())['definitions']


def trend_variations(times_ns, values, window):
    """Same OLS window and units as Trend._compute_trends, using prefix sums."""
    x = (times_ns - times_ns[0]) // 1_000_000_000
    x = x.astype(float)
    j = np.searchsorted(times_ns, times_ns - int(window * 1e9), side='left')
    end = np.arange(len(x)) + 1
    n = end - j
    # Center values to reduce cancellation; OLS slopes are shift-invariant.
    y = values - values[0]
    def sums(z):
        p = np.r_[0., np.cumsum(z)]
        return p[end] - p[j]
    sx, sy = sums(x), sums(y)
    den = sums(x*x) - sx*sx/n
    num = sums(x*y) - sx*sy/n
    eligible = (n >= 2) & (den > 1e-10)
    result = np.full(len(x), np.nan)
    result[eligible] = num[eligible] / den[eligible] * window
    return result, eligible, j


def main():
    specs = definitions()
    aliases = {}
    for name, spec in specs.items():
        for attribute in spec['input_attributes']:
            if attribute in aliases and aliases[attribute] != name:
                raise ValueError(f'Ambiguous input alias: {attribute}')
            aliases[attribute] = name
    cohort = set(pd.read_csv(PERSONAL / 'INTERVenE-Enc/data/source/context_data.csv',
                            usecols=['PatientId']).PatientId)
    admissions = {}
    # Separate first pass avoids assuming input row ordering or chunk boundaries.
    for chunk in pd.read_csv(SOURCE, usecols=['PatientId', 'ConceptName', 'StartDateTime'],
                             chunksize=500_000):
        a = chunk[(chunk.ConceptName == 'ADMISSION') & chunk.PatientId.isin(cohort)]
        for pid, ts in zip(a.PatientId, pd.to_datetime(a.StartDateTime)):
            if pid in admissions and admissions[pid] != ts:
                raise ValueError('Multiple different admission timestamps')
            admissions[pid] = ts
    print(f'Admission timestamps: {len(admissions)} / {len(cohort)}', flush=True)
    frames = {name: [] for name in specs}
    excluded = dict(nonfinite=0, out_of_raw_range=0)
    for k, chunk in enumerate(pd.read_csv(SOURCE, chunksize=500_000,
                                          usecols=['PatientId', 'ConceptName', 'StartDateTime', 'Value'])):
        c = chunk[chunk.ConceptName.isin(aliases) & chunk.PatientId.isin(admissions)].copy()
        c['time'] = pd.to_datetime(c.StartDateTime)
        elapsed = (c.time - c.PatientId.map(admissions)).dt.total_seconds()
        c = c[(elapsed >= 0) & (elapsed <= 48*3600)].copy()
        c['number'] = pd.to_numeric(c.Value, errors='coerce')
        excluded['nonfinite'] += int((~np.isfinite(c.number)).sum())
        c = c[np.isfinite(c.number)]
        for attribute, group in c.groupby('ConceptName', sort=False):
            name = aliases[attribute]
            valid = np.zeros(len(group), bool)
            for lo, hi in specs[name]['input_attributes'][attribute]:
                valid |= (group.number >= lo) & (group.number <= hi)
            excluded['out_of_raw_range'] += int((~valid).sum())
            group = group[valid]
            frames[name].append(group[['PatientId', 'time', 'number']])
        if k % 10 == 0:
            print(f'Read {(k+1)*500_000:,} raw rows', flush=True)
    state_rows, boundary_rows, trend_rows = [], [], []
    parity_checks = 0
    for name, parts in frames.items():
        if not parts:
            continue
        data = pd.concat(parts, ignore_index=True).sort_values(['PatientId', 'time'], kind='stable')
        v = data.number.to_numpy()
        cuts = np.array(specs[name]['thresholds'])
        gap = np.diff(cuts)
        scale = np.minimum(np.r_[gap[0], gap], np.r_[gap, gap[-1]])
        dist = np.abs(v[:, None] - cuts[None, :])
        norm = dist / scale
        base = dict(concept=name, observations=len(v), admissions=data.PatientId.nunique())
        state_rows.append(dict(**base, exact_boundary_pct=100*np.any(dist <= 1e-10, axis=1).mean(),
                               **{f'near_{f:g}_pct':100*np.any(norm <= f+1e-10, axis=1).mean()
                                  for f in FRACTIONS}))
        for b, cutoff in enumerate(cuts):
            for f in FRACTIONS:
                width = f*scale[b]
                delta = v-cutoff
                center = np.abs(delta) <= width+1e-10
                left = (delta >= -3*width-1e-10) & (delta < -width-1e-10)
                right = (delta > width+1e-10) & (delta <= 3*width+1e-10)
                flank_mean = (left.sum()+right.sum())/2
                boundary_rows.append(dict(concept=name, cutoff=cutoff, fraction=f, half_width=width,
                                          observations=len(v), center_count=int(center.sum()),
                                          left_count=int(left.sum()), right_count=int(right.sum()),
                                          center_to_flank_ratio=float(center.sum()/flank_mean) if flank_mean else None))
        if 'variation' not in specs[name]:
            continue
        variations = []
        ineligible = 0
        variation = specs[name]['variation']
        for _, group in data.groupby('PatientId', sort=False):
            times = group.time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
            values = group.number.to_numpy()
            result, eligible, starts = trend_variations(times, values, specs[name]['window'])
            ineligible += int((~eligible).sum())
            variations.append(result[eligible])
            # Deterministic comparisons with the Mediator's direct dot-product OLS.
            if parity_checks < 3000:
                for i in np.flatnonzero(eligible)[::max(1, len(group)//3)][:3]:
                    xs = ((times[starts[i]:i+1]-times[0])//1_000_000_000).astype(float)
                    ys = values[starts[i]:i+1]
                    dx, dy = xs-xs.mean(), ys-ys.mean()
                    expected = np.dot(dx,dy)/np.dot(dx,dx)*specs[name]['window']
                    if not np.isclose(result[i], expected, rtol=1e-7, atol=1e-7):
                        raise AssertionError((name, result[i], expected))
                    parity_checks += 1
        tv = np.concatenate(variations)
        margin = np.abs(np.abs(tv)-variation)/variation
        trend_rows.append(dict(**base, eligible_windows=len(tv), insufficient_or_degenerate=ineligible,
                               variation_threshold=variation, window_hours=specs[name]['window']/3600,
                               **{f'near_{f:g}_pct':100*np.mean(margin <= f+1e-10) for f in FRACTIONS}))
        print(f'{name}: {len(v):,} measurements; {len(tv):,} eligible trend windows', flush=True)
    pd.DataFrame(state_rows).to_csv(OUT/'state_summary.csv', index=False)
    pd.DataFrame(boundary_rows).to_csv(OUT/'state_boundary_bands.csv', index=False)
    pd.DataFrame(trend_rows).to_csv(OUT/'trend_summary.csv', index=False)
    metadata = dict(source=SOURCE.relative_to(PERSONAL).as_posix(), source_bytes=SOURCE.stat().st_size,
                    cohort_admissions=len(cohort), matched_admissions=len(admissions),
                    observation_hours=48, excluded=excluded, ols_parity_checks=parity_checks,
                    definitions=specs, fractions=FRACTIONS,
                    repository_scope=json.loads((OUT/'repository_scope.json').read_text()))
    (OUT/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Aggregate results written to threshold-proximity output directory', flush=True)


if __name__ == '__main__':
    main()

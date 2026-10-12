"""Check the audit's window calculations against the actual Mediator classifier."""
import json
import sys
from datetime import timedelta
import numpy as np
import pandas as pd
from analyze import MEDIATOR_ROOT, OUT, definitions, trend_variations

sys.path.insert(0, str(MEDIATOR_ROOT))
from core.tak.trend import Trend

rng = np.random.default_rng(42)
checks = []
for name, spec in definitions().items():
    if 'variation' not in spec:
        continue
    trend = Trend.__new__(Trend)
    trend.attribute_idx = 0
    trend.time_steady = timedelta(seconds=spec['window'])
    trend.significant_variation = spec['variation']
    # Irregular sampling, large gaps, and both slope signs. Input tables require
    # unique patient/concept/timestamp keys, so timestamps are strictly increasing.
    times = pd.Timestamp('2020-01-01') + pd.to_timedelta(
        np.cumsum(rng.choice([60, 600, 3600, 86400], 500)), unit='s')
    values = np.cumsum(rng.normal(0, spec['variation']/2, 500))
    frame = pd.DataFrame(dict(PatientId=1, ConceptName=name,
                              StartDateTime=times, EndDateTime=times,
                              Value=values, AbstractionType='raw-concept'))
    expected = trend._compute_trends(frame).Value.to_numpy()
    score, eligible, _ = trend_variations(times.to_numpy().astype(np.int64), values, spec['window'])
    actual = np.where(score >= spec['variation'], 'Increasing',
                      np.where(score <= -spec['variation'], 'Decreasing', 'Steady'))
    assert np.array_equal(expected, actual), name
    checks.append(dict(concept=name, positions=len(actual), matched=True))
(OUT/'implementation_validation.json').write_text(json.dumps(checks, indent=2))
print(f'Mediator label agreement: {sum(c["positions"] for c in checks)} / '
      f'{sum(c["positions"] for c in checks)} positions across {len(checks)} concepts')

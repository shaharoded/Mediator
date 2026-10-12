"""Resolve the audit scope from the saved Mediator TAKRepository object."""
from pathlib import Path
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import hashlib
import json
import math
import pickle
import sys

sys.path.insert(0, str(MEDIATOR_ROOT))
from core.tak.raw_concept import RawConcept, ParameterizedRawConcept

ROOT = RESULTS_ROOT / 'threshold-proximity'
ROOT.mkdir(parents=True, exist_ok=True)
source = MEDIATOR_ROOT/'backend/data/tak-repo.pkl'
with source.open('rb') as f:
    repo = pickle.load(f)
specs, coverage = {}, []
for tak in sorted(repo.taks.values(), key=lambda t:t.name):
    if tak.family != 'state':
        continue
    raw = repo.get(tak.derived_from)
    row = dict(state=tak.name, parent=tak.derived_from,
               dependencies=sorted(repo.graph.get(tak.name, set())))
    if isinstance(raw, ParameterizedRawConcept):
        row.update(status='Derived numeric quantity; requires parameter transformation',
                   parent_dependencies=sorted(repo.graph.get(raw.name,set())))
        coverage.append(row)
        continue
    if not isinstance(raw, RawConcept) or raw.concept_type != 'raw-numeric':
        row['status'] = 'Not a direct scalar numeric state'
        coverage.append(row)
        continue
    rules = tak.abstraction_rules
    if any(set(rule.constraints) != {0} or rule.constraints[0]['type'] != 'numeric'
           for rule in rules):
        row['status'] = 'Not a scalar numeric threshold rule'
        coverage.append(row)
        continue
    cuts = sorted({float(v[key]) for rule in rules
                   for v in rule.constraints[0]['rules'] for key in ('min','max')
                   if key in v and math.isfinite(v[key])})
    if len(cuts) < 2:
        row['status'] = 'No adjacent finite cutoff spacing'
        coverage.append(row)
        continue
    attrs = {a['name']:[[a['min'], a['max']]] for a in raw.attributes}
    spec = dict(thresholds=cuts, input_attributes=attrs, state=tak.name,
                ranges=list(attrs.values())[0],
                state_dependencies=sorted(repo.graph.get(tak.name,set())))
    trends = [t for t in repo.taks.values() if t.family=='trend' and t.derived_from==raw.name]
    if len(trends)>1:
        raise ValueError(f'Multiple trend definitions for {raw.name}')
    if trends:
        t=trends[0]
        spec.update(trend=t.name, variation=t.significant_variation,
                    window=t.time_steady.total_seconds(),
                    trend_dependencies=sorted(repo.graph.get(t.name,set())))
    specs[raw.name]=spec
    row.update(status='Eligible', input_attributes=sorted(attrs), trends=[t.name for t in trends])
    coverage.append(row)
payload=dict(repository_source=source.relative_to(MEDIATOR_ROOT).as_posix(), repository_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             total_taks=len(repo.taks), full_graph={k:sorted(v) for k,v in repo.graph.items()},
             definitions=specs, state_coverage=coverage,
             raw_numeric_without_eligible_state=sorted(t.name for t in repo.taks.values()
                 if t.family=='raw-concept' and getattr(t,'concept_type',None)=='raw-numeric' and t.name not in specs))
(ROOT/'repository_scope.json').write_text(json.dumps(payload,indent=2))
print(f'Repository: {len(repo.taks)} TAKs; {len(specs)} eligible numeric state concepts')

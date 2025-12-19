from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, TYPE_CHECKING
import pandas as pd
import json
import openpyxl
import logging
import pickle
import gzip
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from .tak import TAK
    from .raw_concept import RawConcept
    from .event import Event
    from .state import State
    from .trend import Trend
    from .context import Context
    from .pattern import Pattern

# Global singleton TAKRepository instance (set by mediator at startup)
_GLOBAL_TAK_REPO: Optional["TAKRepository"] = None

def get_tak_repository() -> "TAKRepository":
    """Access the global TAK repository."""
    if _GLOBAL_TAK_REPO is None:
        raise RuntimeError("TAK repository not initialized. Call set_tak_repository() first.")
    return _GLOBAL_TAK_REPO

def set_tak_repository(repo: "TAKRepository") -> None:
    """Set the global TAK repository (called once by mediator)."""
    global _GLOBAL_TAK_REPO
    _GLOBAL_TAK_REPO = repo


@dataclass
class TAKRepository:
    """
    Registry of all parsed TAK objects (raw-concepts, states, trends, etc.).
    Does not cache .df (computed on-demand to save memory).
    """
    taks: Dict[str, TAK] = field(default_factory=dict)  # {tak_name: TAK_instance}
    graph: Dict[str, Set[str]] = field(default_factory=dict)  # Dependency graph
    execution_order: List[str] = field(default_factory=list)  # Topological sort result

    def register(self, tak: TAK) -> None:
        """Register a TAK by name."""
        if tak.name in self.taks:
            raise ValueError(f"Duplicate TAK name: {tak.name}")
        self.taks[tak.name] = tak

    def get(self, name: str) -> Optional[TAK]:
        """Retrieve a TAK by name."""
        return self.taks.get(name)

    def finalize_repository(self) -> None:
        """
        Finalize the TAK repository after all TAKs are registered.
        Detects circular references and computes topological execution order.
        (Individual TAK validation already happens in each TAK's .parse() method.)
        """
        # 1) Build dependency graph and detect circular references
        self.graph = self._build_dependency_graph()  # Store as instance attribute
        self._detect_circular_references(self.graph)

        # 2) Compute topological sort order
        self.execution_order = self._topological_sort(self.graph)
        logger.info(f"TAK execution order: {self.execution_order}")

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a dependency graph: {tak_name: set_of_dependent_tak_names}
        A TAK depends on all TAKs it references (derived-from, parameters, clippers, etc.)
        
        Returns deduplicated dependency sets (Pattern references same TAK twice → appears once).
        """
        graph: Dict[str, Set[str]] = {name: set() for name in self.taks}

        for tak_name, tak in self.taks.items():
            dependencies = self._extract_dependencies(tak)
            graph[tak_name] = dependencies  # set() automatically deduplicates

        return graph

    def _extract_dependencies(self, tak: TAK) -> Set[str]:
        """Extract all TAK names that this TAK depends on (deduplicated)."""
        # Import here to avoid circular imports
        from .raw_concept import RawConcept, ParameterizedRawConcept
        from .event import Event
        from .state import State
        from .trend import Trend
        from .context import Context
        from .pattern import Pattern

        deps = set()

        # Raw Concept: no dependencies (data source)
        if isinstance(tak, RawConcept) and not isinstance(tak, ParameterizedRawConcept):
            return deps
        
        # ParameterizedRawConcept: depends on derived-from + parameters (check BEFORE RawConcept!)
        if isinstance(tak, ParameterizedRawConcept):
            deps.add(tak.derived_from)
            for param in tak.parameters:
                deps.add(param["name"])
            return deps

        # Event: depends on derived-from raw-concepts
        if isinstance(tak, Event):
            for df in tak.derived_from:
                deps.add(df["name"])
            return deps

        # State: depends on derived-from (raw-concept or event)
        if isinstance(tak, State):
            deps.add(tak.derived_from)
            return deps

        # Trend: depends on derived-from raw-concept
        if isinstance(tak, Trend):
            deps.add(tak.derived_from)
            return deps

        # Context: depends on derived-from + clippers
        if isinstance(tak, Context):
            for df in tak.derived_from:
                deps.add(df["name"])
            for clipper in tak.clippers:
                deps.add(clipper["name"])
            return deps

        # Pattern: depends on derived-from + parameters TAKs
        if isinstance(tak, Pattern):
            for df in tak.derived_from:
                deps.add(df["name"])
            for param in tak.parameters:
                deps.add(param["name"])
            return deps

        return deps

    def _detect_circular_references(self, graph: Dict[str, Set[str]]) -> None:
        """
        Detect circular references using DFS.
        Raises ValueError if a cycle is found.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in self.taks:
                    # Neighbor not in repository (missing TAK)
                    raise ValueError(f"TAK '{node}' references missing TAK '{neighbor}'")

                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    raise ValueError(
                        f"Circular reference detected: {' → '.join(cycle_path)}"
                    )

            rec_stack.discard(node)

        # Run DFS from each unvisited node
        for tak_name in self.taks:
            if tak_name not in visited:
                dfs(tak_name, [])

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        Compute topological sort order using Kahn's algorithm.
        Returns list of TAK names in execution order (dependencies first).
        
        graph: {tak_name: set_of_dependencies_it_needs}
        We need to reverse this to get {tak_name: set_of_dependents_that_need_it}
        """
        # 1. Build reverse graph: {dependency: set_of_taks_that_need_it}
        reverse_graph: Dict[str, Set[str]] = {name: set() for name in self.taks}
        for tak_name, deps in graph.items():
            for dep in deps:
                reverse_graph[dep].add(tak_name)
        
        # 2. Calculate in-degree (number of dependencies each TAK has)
        in_degree = {name: len(deps) for name, deps in graph.items()}
        
        # 3. Initialize queue with TAKs that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            # Sort queue by TAK family for predictable ordering
            queue.sort(key=lambda name: (self._family_priority(self.taks[name].family), name))
            node = queue.pop(0)
            order.append(node)
            
            # For each TAK that depends on this node, decrement its in-degree
            for dependent in reverse_graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(order) != len(self.taks):
            # Should never happen if circular reference detection works
            missing = set(self.taks.keys()) - set(order)
            raise RuntimeError(f"Topological sort failed. Missing TAKs: {missing}")
        
        return order

    @staticmethod
    def _family_priority(family: str) -> int:
        """Return priority for execution order (lower = earlier)."""
        priority_map = {
            "raw-concept": 0,
            "event": 1,
            "state": 2,
            "trend": 3,
            "context": 4,
            "pattern": 5,
            "local-pattern": 5,
            "global-pattern": 6,
        }
        return priority_map.get(family, 99)
    
    def save(self, file_path: str, compress: bool = True) -> None:
        """
        Serialize the TAKRepository to a pickle file.
        If compress=True, use gzip compression.
        """
        if compress:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export_metadata(self) -> "pd.DataFrame":
        """
        Export TAK metadata to a pandas DataFrame for documentation/analysis.
        
        Returns a DataFrame with columns:
        - TAK_Name: TAK identifier
        - Family: TAK family (raw-concept, event, state, trend, context, pattern, etc.)
        - Categories: List of category tags
        - Description: Human-readable description
        - Attributes: For raw-concepts (formatted as readable string)
        - Derived_From: Dependencies (formatted as readable string)
        - Parameters: For parameterized TAKs (formatted as readable string)
        - Persistence: For states/trends (good_after_hours, interpolate, max_skip)
        - Trend_Params: For trends (attribute_idx, significant_variation, time_steady_hours)
        - Context_Windows: For contexts (value-specific before/after hours)
        - Clippers: For contexts (formatted as readable string)
        - Compliance_Function: For patterns (function name and trapezoid values)
        - Abstraction_Rules: For states/events/contexts/patterns (rules with constraints/relation)
        """
        
        def _duration_to_hours(val: str) -> float:
            """Parse compact duration string (e.g., '72h', '2d', '15m') into hours (float).
            Falls back to float(val) if already numeric; returns 0.0 on failure.
            """
            try:
                from .utils import parse_duration  # local import to avoid cycles
                return parse_duration(val).total_seconds() / 3600.0
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return 0.0

        def _json_default(o):
            """Best-effort serializer for non-JSON-native objects (timedelta → hours, sets, numpy/pandas, etc.)."""
            # Timedelta → hours (float)
            try:
                import pandas as _pd
                if isinstance(o, _pd.Timedelta):
                    return o.total_seconds() / 3600.0
            except Exception:
                pass
            try:
                from datetime import timedelta as _td
                if isinstance(o, _td):
                    return o.total_seconds() / 3600.0
            except Exception:
                pass
            # Sets → lists
            try:
                from collections.abc import Set as _Set
                if isinstance(o, _Set):
                    return list(o)
            except Exception:
                pass
            # Numpy → Python types
            try:
                import numpy as np  # type: ignore
                if isinstance(o, (np.integer, np.floating)):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
            except Exception:
                pass
            # Pandas timestamps → isoformat
            try:
                if isinstance(o, pd.Timestamp):
                    return o.isoformat()
            except Exception:
                pass
            # Datetime-like (has isoformat) → isoformat; else str fallback
            if hasattr(o, 'isoformat'):
                try:
                    return o.isoformat()
                except Exception:
                    pass
            return str(o)
        
        rows = []
        
        for tak_name in sorted(self.taks.keys()):
            tak = self.taks[tak_name]
            
            # Extract basic fields
            row = {
                'TAK_Name': tak.name,
                'Family': tak.family,
                'Categories': ', '.join(tak.categories) if hasattr(tak, 'categories') and tak.categories else '',
                'Description': tak.description if hasattr(tak, 'description') else '',
            }
            
            # Attributes (raw-concepts)
            if hasattr(tak, 'attributes') and tak.attributes:
                # Format as readable JSON
                row['Attributes'] = json.dumps(tak.attributes, indent=2, ensure_ascii=False, default=_json_default)
            else:
                row['Attributes'] = ''
            
            # Derived_From (events, states, trends, contexts, patterns)
            if hasattr(tak, 'derived_from'):
                if isinstance(tak.derived_from, list):
                    # Event/Context/Pattern: list of dicts
                    row['Derived_From'] = json.dumps(tak.derived_from, indent=2, ensure_ascii=False, default=_json_default)
                else:
                    # State/Trend: single string
                    row['Derived_From'] = tak.derived_from
            else:
                row['Derived_From'] = ''
            
            # Parameters (parameterized-raw-concepts, patterns)
            if hasattr(tak, 'parameters') and tak.parameters:
                row['Parameters'] = json.dumps(tak.parameters, indent=2, ensure_ascii=False, default=_json_default)
            else:
                row['Parameters'] = ''
            
            # Initialize family-specific columns
            row['Persistence'] = ''
            row['Trend_Params'] = ''
            row['Context_Windows'] = ''
            row['Clippers'] = ''
            compliance_str = ''
            rules_str = ''
            
            # Branch by TAK family (avoid runtime imports/cycles)
            fam = getattr(tak, 'family', '')

            # ===== STATE TAKs =====
            if fam == 'state':
                # Persistence parameters
                persistence_info = {
                    'good_after_hours': tak.good_after.total_seconds() / 3600.0,
                    'interpolate': tak.interpolate,
                    'max_skip': tak.max_skip
                }
                row['Persistence'] = json.dumps(persistence_info, indent=2)
                
                # Abstraction rules (discretization logic)
                if hasattr(tak, 'abstraction_rules') and tak.abstraction_rules:
                    state_rules = []
                    for rule in tak.abstraction_rules:
                        # Serialize constraints dict: {attr_idx: {type, rules:[...]}}
                        constraints_serialized = {}
                        try:
                            for idx, spec in (getattr(rule, 'constraints', {}) or {}).items():
                                constraints_serialized[int(idx)] = {
                                    'type': spec.get('type'),
                                    'rules': spec.get('rules', [])
                                }
                        except Exception:
                            constraints_serialized = getattr(rule, 'constraints', {}) or {}
                        rule_entry = {
                            'value': getattr(rule, 'value', None),
                            'operator': getattr(rule, 'operator', 'and'),
                            'constraints': constraints_serialized
                        }
                        state_rules.append(rule_entry)
                    rules_str = json.dumps(state_rules, indent=2, ensure_ascii=False, default=_json_default)
            
            # ===== TREND TAKs =====
            elif fam == 'trend':
                # Trend-specific parameters
                trend_params = {
                    'attribute_idx': tak.attribute_idx,
                    'significant_variation': tak.significant_variation,
                    'time_steady_hours': tak.time_steady.total_seconds() / 3600.0
                }
                row['Trend_Params'] = json.dumps(trend_params, indent=2)
                
                # Persistence
                persistence_info = {
                    'good_after_hours': tak.good_after.total_seconds() / 3600.0
                }
                row['Persistence'] = json.dumps(persistence_info, indent=2)
            
            # ===== EVENT TAKs =====
            elif fam == 'event':
                # Abstraction rules (value mapping with ref-based constraints)
                if hasattr(tak, 'abstraction_rules') and tak.abstraction_rules:
                    event_rules = []
                    for rule in tak.abstraction_rules:
                        rule_entry = {
                            'value': rule.value,
                            'operator': rule.operator,
                            'constraints': {}
                        }
                        for ref, constraint_list in rule.constraints.items():
                            rule_entry['constraints'][ref] = constraint_list
                        event_rules.append(rule_entry)
                    rules_str = json.dumps(event_rules, indent=2, ensure_ascii=False, default=_json_default)
            
            # ===== CONTEXT TAKs =====
            elif fam == 'context':
                # Clippers
                if hasattr(tak, 'clippers') and tak.clippers:
                    row['Clippers'] = json.dumps(tak.clippers, indent=2, ensure_ascii=False, default=_json_default)
                
                # Context windows (convert timedeltas to hours)
                if hasattr(tak, 'context_windows') and tak.context_windows:
                    windows_export = {}
                    for value, window in tak.context_windows.items():
                        windows_export[str(value) if value is not None else 'default'] = {
                            'before_hours': window['before'].total_seconds() / 3600.0,
                            'after_hours': window['after'].total_seconds() / 3600.0
                        }
                    row['Context_Windows'] = json.dumps(windows_export, indent=2)
                
                # Abstraction rules (value mapping with ref-based constraints)
                if hasattr(tak, 'abstraction_rules') and tak.abstraction_rules:
                    context_rules = []
                    for rule in tak.abstraction_rules:
                        rule_entry = {
                            'value': rule.value,
                            'operator': rule.operator,
                            'constraints': {}
                        }
                        for ref, constraint_list in rule.constraints.items():
                            rule_entry['constraints'][ref] = constraint_list
                        context_rules.append(rule_entry)
                    rules_str = json.dumps(context_rules, indent=2, ensure_ascii=False, default=_json_default)
            
            # ===== PATTERN TAKs =====
            elif fam in ('pattern', 'local-pattern', 'global-pattern') and hasattr(tak, 'abstraction_rules') and tak.abstraction_rules:
                entries = []
                rules_entries = []
                for rule in tak.abstraction_rules:
                    tcc = getattr(rule, 'time_constraint_compliance', None)
                    if tcc:
                        trapez_raw = tcc.get('trapeze')
                        trapeze = None
                        if trapez_raw is not None:
                            try:
                                # Convert compact duration strings to hours (floats)
                                trapeze = [
                                    _duration_to_hours(trapez_raw[0]),
                                    _duration_to_hours(trapez_raw[1]),
                                    _duration_to_hours(trapez_raw[2]),
                                    _duration_to_hours(trapez_raw[3]),
                                ]
                            except Exception:
                                trapeze = trapez_raw  # fallback to raw
                        entries.append({
                            'type': 'time-constraint',
                            'func_name': tcc.get('func_name'),
                            'trapeze': trapeze,
                            'parameters': tcc.get('parameters', []),
                        })
                    vcc = getattr(rule, 'value_constraint_compliance', None)
                    if vcc:
                        entries.append({
                            'type': 'value-constraint',
                            'func_name': vcc.get('func_name'),
                            'trapeze': vcc.get('trapeze'),
                            'targets': vcc.get('targets', []),
                            'parameters': vcc.get('parameters', []),
                        })

                    # Summarize the rule's temporal relation and constraints
                    rel = getattr(rule, 'relation_spec', {}) or {}
                    relation_summary = {
                        'how': rel.get('how'),
                        'max_distance_hours': _duration_to_hours(rel.get('max_distance')) if rel.get('max_distance') else None,
                        'min_distance_hours': _duration_to_hours(rel.get('min_distance')) if rel.get('min_distance') else None,
                    }
                    def _summarize_part(part: dict):
                        if not part:
                            return None
                        attrs = part.get('attributes', {}) or {}
                        return {
                            'select': part.get('select'),
                            'attributes': {
                                name: {
                                    'idx': spec.get('idx'),
                                    'allowed_values': list(spec.get('allowed_values', [])) if spec.get('allowed_values') else [],
                                    'min': spec.get('min'),
                                    'max': spec.get('max'),
                                }
                                for name, spec in attrs.items()
                            }
                        }
                    anchor_summary = _summarize_part(rel.get('anchor') or {})
                    event_summary = _summarize_part(rel.get('event') or {})
                    context_summary = _summarize_part(getattr(rule, 'context_spec', {}) or {})
                    rules_entries.append({
                        'relation': relation_summary,
                        'anchor': anchor_summary,
                        'event': event_summary,
                        'context': context_summary,
                    })
                if entries:
                    compliance_str = json.dumps(entries, indent=2, ensure_ascii=False, default=_json_default)
                if rules_entries:
                    rules_str = json.dumps(rules_entries, indent=2, ensure_ascii=False, default=_json_default)
            row['Compliance_Function'] = compliance_str
            row['Abstraction_Rules'] = rules_str
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
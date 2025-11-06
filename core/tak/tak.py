from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, List, Any
import pandas as pd
from datetime import timedelta
from typing import Dict, Any, List, Optional, Iterable, Set
from abc import ABC, abstractmethod
from lxml import etree as lxml_etree
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# Import schema path from config
from ..config import TAK_SCHEMA_PATH
from .utils import parse_duration

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
class TAK:
    """Base class for all TAK families. Subclasses must implement parse(), validate(), apply()."""
    name: str
    categories: Tuple[str, ...] = ()
    description: str = ""
    family: Optional[str] = None # One of: "raw-concept", "state", "trend", "context", "event", "pattern"
    df: Optional[pd.DataFrame] = field(default=None, repr=False)

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "TAK":
        """Subclasses: parse a single XML file and return an instance."""
        raise NotImplementedError

    def validate(self) -> None:
        """Subclasses: validate TAK settings; raise ValueError on problems."""
        raise NotImplementedError

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subclasses: apply TAK logic to a per-patient, pre-filtered df
        (with columns PatientId, ConceptName, StartDateTime, EndDateTime, Value).
        DF can be originated from InputPatientData (if TAK references a raw-concept) or OutputPatientData.
        Return df with the schema: 
        PatientId, 
        ConceptName = self.name, 
        StartDateTime, EndDateTime, 
        Value
        AbstractionType = self.family
        """
        raise NotImplementedError


@dataclass
class TAKRepository:
    """
    Registry of all parsed TAK objects (raw-concepts, states, trends, etc.).
    Does not cache .df (computed on-demand to save memory).
    """
    taks: Dict[str, TAK] = field(default_factory=dict)  # {tak_name: TAK_instance}

    def register(self, tak: TAK) -> None:
        """Register a TAK by name."""
        if tak.name in self.taks:
            raise ValueError(f"Duplicate TAK name: {tak.name}")
        self.taks[tak.name] = tak

    def get(self, name: str) -> Optional[TAK]:
        """Retrieve a TAK by name."""
        return self.taks.get(name)

    def validate_all(self) -> None:
        """Run business-logic validation on all registered TAKs."""
        for tak in self.taks.values():
            tak.validate()


class TAKRule(ABC):
    """
    Abstract base for all rule types (discretization, abstraction, trend, context, pattern).
    Subclasses: StateDiscretizationRule, StateAbstractionRule, EventAbstractionRule.
    """
    @abstractmethod
    def matches(self, *args, **kwargs) -> bool:
        """Check if rule matches given input."""
        pass


class StateDiscretizationRule(TAKRule):
    """
    Single discretization threshold for one attribute index.
    Example: attribute idx=0, value="Low", min=10, max=20
    """
    def __init__(self, attr_idx: int, value: str, min_val: Optional[float], max_val: Optional[float]):
        self.attr_idx = attr_idx
        self.value = value
        self.min = min_val if min_val is not None else -float('inf')
        self.max = max_val if max_val is not None else float('inf')

    def matches(self, val: Any) -> bool:
        """Check if a numeric value falls within [min, max)."""
        try:
            x = float(val)
            return self.min <= x < self.max
        except (ValueError, TypeError):
            return False


class StateAbstractionRule(TAKRule):
    """
    Logical rule to combine discrete attribute values → final state label.
    Example: value="SubCutaneous Low", operator="and", constraints={(0, ['Very Low','Low']), (1, ['SubCutaneous'])}
    
    Rule matches if ALL specified constraints are satisfied (tuple can have additional unreferenced attributes).
    """
    def __init__(self, value: str, operator: str, constraints: Dict[int, List[str]]):
        self.value = value
        self.operator = operator.lower() # One of "and", "or"
        self.constraints = constraints  # {attr_idx: [allowed_discrete_values]}

    def matches(self, discrete_tuple: Tuple[Any, ...]) -> bool:
        """
        Check if a discrete tuple satisfies this rule.
        Rule matches if all constraint indices are satisfied (tuple can have MORE attributes than rule references).
        """
        results = []
        for idx, allowed in self.constraints.items():
            if idx >= len(discrete_tuple):
                # Rule references an index not present in tuple → fail
                results.append(False)
            else:
                # Convert tuple element to string for comparison (handles bool/nominal uniformly)
                val_str = str(discrete_tuple[idx])
                results.append(val_str in allowed)
        
        if self.operator == "and":
            return all(results)
        elif self.operator == "or":
            return any(results)
        return False


class EventAbstractionRule(TAKRule):
    """
    Custom abstraction rule for Event and Context TAKs.
    constraints: {attr_name: [{type: 'equal'|'min'|'max'|'range', value/min/max: ...}]}
    
    Unlike AbstractionRule (which matches tuple indices), EventAbstractionRule matches
    by raw-concept name and supports flexible numeric constraints (min/max/range/equal).
    """
    def __init__(self, value: str, operator: str, constraints: Dict[str, List[Dict[str, Any]]]):
        self.value = value
        self.operator = operator.lower() # One of "and", "or"
        self.constraints = constraints  # {attr_name: [{constraint_spec}]}

    def matches(self, row: pd.Series, derived_from: List[Dict[str, Any]]) -> bool:
        """
        Check if a row satisfies this rule.
        operator="or": any attribute matches
        operator="and": all attributes match
        
        Args:
            row: DataFrame row with columns [PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType]
            derived_from: List of {name, tak_type, idx} dicts from Event.derived_from
        """
        results = []
        for attr_name, constraint_list in self.constraints.items():
            # Find derived-from entry for this attribute
            df_entry = next((df for df in derived_from if df["name"] == attr_name), None)
            if df_entry is None:
                results.append(False)
                continue

            # Check if row's ConceptName matches this attribute
            if row["ConceptName"] != attr_name:
                results.append(False)
                continue

            # Extract value from tuple (using idx)
            idx = df_entry["idx"]
            row_value = row["Value"]
            if isinstance(row_value, tuple):
                if idx >= len(row_value):
                    results.append(False)
                    continue
                val = row_value[idx]
            else:
                val = row_value

            # Check constraints
            attr_matches = False
            for c in constraint_list:
                if c["type"] == "equal":
                    if str(val) == c["value"]:
                        attr_matches = True
                        break
                elif c["type"] == "min":
                    try:
                        if float(val) >= c["value"]:
                            attr_matches = True
                            break
                    except (ValueError, TypeError):
                        pass
                elif c["type"] == "max":
                    try:
                        if float(val) <= c["value"]:
                            attr_matches = True
                            break
                    except (ValueError, TypeError):
                        pass
                elif c["type"] == "range":
                    try:
                        if c["min"] <= float(val) <= c["max"]:
                            attr_matches = True
                            break
                    except (ValueError, TypeError):
                        pass
            
            results.append(attr_matches)

        if self.operator == "and":
            return all(results)
        elif self.operator == "or":
            return any(results)
        return False


class TemporalRelationRule(TAKRule):
    """Locate anchor/event pairs that satisfy temporal constraints, with an optional context."""
    def __init__(
        self,
        derived_map: Dict[str, Dict[str, Any]],
        relation_spec: Dict[str, Any],
        context_spec: Optional[Dict[str, Any]] = None,
    ):
        self.derived_map = derived_map
        self.relation_spec = relation_spec
        self.context_spec = context_spec or {}
        self.max_delta: Optional[timedelta] = None
        if relation_spec.get("max_distance"):
            self.max_delta = parse_duration(relation_spec["max_distance"])

    def find_matches(
        self,
        df: pd.DataFrame,
        used_anchor_ids: Optional[Set[int]] = None,
        used_event_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        used_anchor_ids = used_anchor_ids or set()
        used_event_ids = used_event_ids or set()

        # Separate input df to components, with only the relevant rows to use as anchor/ event.
        anchors = self._extract_candidates(df, self.relation_spec.get("anchor"))
        events = self._extract_candidates(df, self.relation_spec.get("event"))
        contexts = self._extract_candidates(df, self.context_spec) if self.context_spec else None

        results: List[Dict[str, Any]] = []
        if anchors.empty or events.empty:
            return results

        # Sort anchor options and event options per TAK 'select' specification
        anchor_order = self._order_indices(anchors, self.relation_spec.get("anchor", {}))
        event_order = self._order_indices(events, self.relation_spec.get("event", {}))

        # Iterate over anchors using 'used' buckets to enforce one-to-one matching
        for anchor_idx in anchor_order:
            if anchor_idx in used_anchor_ids:
                continue
            anchor_row = anchors.loc[anchor_idx]
            anchor_start = anchor_row.StartDateTime

            for event_idx in event_order:
                if event_idx in used_event_ids:
                    continue
                event_row = events.loc[event_idx]
                if not self._temporal_match(anchor_row, event_row):
                    continue
                if not self._context_satisfied(anchor_row, event_row, contexts):
                    continue
                results.append(
                    {
                        "anchor_idx": anchor_idx,
                        "event_idx": event_idx,
                        "anchor_row": anchor_row,
                        "event_row": event_row,
                        "start": anchor_start,
                        "end": event_row.EndDateTime,
                    }
                )
                used_anchor_ids.add(anchor_idx)
                used_event_ids.add(event_idx)
                break  # one-to-one pairing
        return results

    def _extract_candidates(self, df: pd.DataFrame, spec: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract candidate rows for anchor/event/context attributes.
        Supports multiple attributes (OR semantics): any attribute that satisfies the constraints is eligible.
        """
        if not spec:
            return pd.DataFrame(columns=df.columns)

        attrs = spec.get("attributes", {})
        if not attrs:
            return pd.DataFrame(columns=df.columns)

        masked_parts = []
        # Take the filtered subset derived from every attribute
        for attr_name, constraints in attrs.items():
            idx = constraints.get("idx", 0)
            rows = df[df["ConceptName"] == attr_name].copy()
            if rows.empty:
                continue

            # Get relevant value for the attribute given idx (if relevant) - A 1D operation
            rows["__value__"] = rows.apply(lambda r: self._extract_value(r["Value"], idx), axis=1)

            allowed = constraints.get("allowed_values") or set()
            if allowed:
                rows = rows[rows["__value__"].astype(str).isin(allowed)]
            if constraints.get("min") is not None:
                rows = rows[pd.to_numeric(rows["__value__"], errors="coerce") >= constraints["min"]]
            if constraints.get("max") is not None:
                rows = rows[pd.to_numeric(rows["__value__"], errors="coerce") <= constraints["max"]]

            if not rows.empty:
                masked_parts.append(rows)

        if not masked_parts:
            return pd.DataFrame(columns=df.columns)

        combined = pd.concat(masked_parts).sort_values("StartDateTime")
        return combined.set_index(combined.index)

    def _order_indices(self, df: pd.DataFrame, spec: Dict[str, Any]) -> List[int]:
        """
        TAK select operation decides if we are to prefer longer or shorter patterns.
        This function assists this selection, sorting ascending for 'first' and descending for 'last'.
        """
        if df.empty:
            return []
        select = (spec or {}).get("select", "first")
        if select == "last":
            return list(df.sort_values("StartDateTime", ascending=False).index)
        return list(df.sort_values("StartDateTime", ascending=True).index)

    def _temporal_match(self, anchor_row: pd.Series, event_row: pd.Series) -> bool:
        """
        Check if the temporal relation between anchor and event rows holds.
        Respects "overlap" and "before" relationships (with <max_distance> restrictions).
        """
        if self.relation_spec["how"] == "overlap":
            return not (
                anchor_row.EndDateTime < event_row.StartDateTime
                or event_row.EndDateTime < anchor_row.StartDateTime
            )
        # how == before
        if anchor_row.EndDateTime > event_row.StartDateTime:
            return False
        if self.max_delta is None:
            return True
        return (event_row.StartDateTime - anchor_row.EndDateTime) <= self.max_delta

    @staticmethod
    def _extract_value(value: Any, idx: int) -> Any:
        """
            1. Extract value from tuple (raw-concept) or string representation.
            2. If not a tuple or parsable string, return value as-is (for all other TAK cases).
        """
        if isinstance(value, tuple):
            return value[idx]
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, tuple):
                    return parsed[idx]
            except (ValueError, SyntaxError):
                return value
        return value

    def _context_satisfied(self, anchor_row: pd.Series, event_row: pd.Series, contexts: Optional[pd.DataFrame]) -> bool:
        """
        Check if any context row overlaps the interval between anchor and event start times.
        Overlap: context.StartDateTime <= max_start AND context.EndDateTime >= min_start
        """
        # Context spec is not defined for this rule
        if not self.context_spec or not self.context_spec.get("attributes"):
            return True
        # No context relevant data was extracted for the patient
        if contexts is None or contexts.empty:
            return False

        min_start = min(anchor_row.StartDateTime, event_row.StartDateTime)
        max_start = max(anchor_row.StartDateTime, event_row.StartDateTime)

        mask = (contexts["StartDateTime"] <= max_start) & (contexts["EndDateTime"] >= min_start)
        return mask.any()


class QATPRule(TAKRule):
    """Evaluate compliance score (placeholder: returns 'True' unless spec provided)."""
    def __init__(self, compliance_spec: Dict[str, Any]):
        self.compliance_spec = compliance_spec

    def evaluate(self, pattern_instance: Dict[str, Any]) -> str:
        if not self.compliance_spec:
            return "True"
        # TODO: implement trapezoid-based compliance scoring
        return "True"


def validate_xml_against_schema(xml_path: Path, schema_path: Optional[Path] = None) -> None:
    """
    Validate XML file against XSD schema.
    
    Args:
        xml_path: Path to XML file
        schema_path: Path to XSD schema file (if None, auto-detects from knowledge-base folder)
    
    Raises:
        ValueError: If validation fails
    """
    if schema_path is None:
        schema_path = Path(TAK_SCHEMA_PATH)
    
    if not schema_path.exists():
        # Schema file not found → skip validation (warn, don't fail)
        logger.warning(f"XSD schema not found: {schema_path}. Skipping structural validation.")
        return
    
    try:
        # Load schema
        with open(schema_path, 'rb') as f:
            schema_doc = lxml_etree.parse(f)
        schema = lxml_etree.XMLSchema(schema_doc)
        
        # Load XML
        with open(xml_path, 'rb') as f:
            xml_doc = lxml_etree.parse(f)
        
        # Validate
        if not schema.validate(xml_doc):
            error_log = schema.error_log
            raise ValueError(f"XML validation failed:\n{error_log}")
            
    except lxml_etree.XMLSyntaxError as e:
        raise ValueError(f"XML syntax error: {e}")



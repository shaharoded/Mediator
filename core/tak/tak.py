from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, List, Any
import pandas as pd
from datetime import timedelta
from abc import ABC, abstractmethod
from lxml import etree as lxml_etree
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# Import schema path from config
from ..config import TAK_SCHEMA_PATH
from .utils import parse_duration


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


class TAKRule(ABC):
    """
    Abstract base for all rule types (discretization, abstraction, trend, context, pattern).
    Subclasses: StateDiscretizationRule, StateAbstractionRule, EventAbstractionRule.
    """
    @abstractmethod
    def matches(self, *args, **kwargs) -> bool:
        """Check if rule matches given input."""
        pass


class StateAbstractionRule(TAKRule):
    """
    Unified abstraction rule combining discretization + abstraction logic.
    
    For numeric attributes: ranges define discretization bins
    For nominal/boolean: exact equality matches
    
    Structure:
        value: Final abstracted state label (e.g., "SubCutaneous Low")
        operator: "and" (all attrs must match) or "or" (any attr matches)
        constraints: {
            attr_idx: {
                "type": "numeric" | "nominal" | "boolean",
                "rules": [
                    {"min": 0, "max": 10},      # for numeric: ranges
                    {"equal": "SubCutaneous"},  # for nominal/boolean: exact matches
                ]
            }
        }
    """
    def __init__(self, value: str, operator: str, constraints: Dict[int, Dict[str, Any]]):
        self.value = value
        self.operator = operator.lower()  # "and" or "or"
        self.constraints = constraints    # {attr_idx: {"type": ..., "rules": [...]}}

    def matches(self, raw_tuple: Tuple[Any, ...]) -> bool:
        """
        Check if a raw tuple satisfies this rule.
        Internally discretizes numeric attributes using ranges.
        
        Args:
            raw_tuple: Tuple from raw-concept value
        
        Returns:
            True if tuple matches this rule (per operator semantics)
        """
        results = []
        
        for idx, constraint_spec in self.constraints.items():
            # Skip if tuple doesn't have this attribute
            if idx >= len(raw_tuple):
                results.append(False)
                continue
            
            attr_value = raw_tuple[idx]
            attr_type = constraint_spec.get("type", "nominal")
            rules = constraint_spec.get("rules", [])
            
            # Check if value matches any of the rules for this attribute
            matches_any_rule = False
            
            for rule in rules:
                if attr_type == "numeric":
                    # Discretize: check if value falls in [min, max) range
                    try:
                        x = float(attr_value)
                        min_val = rule.get("min", -float('inf'))
                        max_val = rule.get("max", float('inf'))
                        
                        if min_val <= x < max_val:
                            matches_any_rule = True
                            break
                    except (ValueError, TypeError):
                        # Can't convert to numeric; doesn't match
                        pass
                
                else:  # nominal or boolean
                    # Exact equality match
                    if rule.get("equal") == str(attr_value):
                        matches_any_rule = True
                        break
            
            results.append(matches_any_rule)
        
        # Combine per operator
        if self.operator == "and":
            return all(results)
        elif self.operator == "or":
            return any(results)
        
        return False


class EventAbstractionRule(TAKRule):
    """
    Abstraction rule for Event and Context TAKs (now using ref mechanism).
    constraints: {ref: [{type: 'equal'|'min'|'max'|'range', value/min/max: ...}]}
    
    Uses ref (not attr_name) to look up derived-from specs.
    """
    def __init__(self, value: str, operator: str, constraints: Dict[str, List[Dict[str, Any]]]):
        self.value = value
        self.operator = operator.lower()  # "and" or "or"
        self.constraints = constraints  # {ref: [{constraint_spec}]}

    def matches(self, row: pd.Series, derived_from_map: Dict[str, Dict[str, Any]]) -> bool:
        """
        Check if a row satisfies this rule.
        operator="or": any ref matches
        operator="and": all refs match
        
        Args:
            row: DataFrame row with columns [PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType]
            derived_from_map: {ref: {name, tak_type, idx}} from Event/Context.derived_from
        """
        results = []
        
        for ref, constraint_list in self.constraints.items():
            # Look up ref in derived_from_map
            df_entry = derived_from_map.get(ref)
            if df_entry is None:
                results.append(False)
                continue

            # Check if row's ConceptName matches this ref's TAK
            if row["ConceptName"] != df_entry["name"]:
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
    """Temporal relation rule for Pattern TAKs with optional compliance functions."""
    def __init__(
        self,
        derived_map: Dict[str, Dict[str, Any]],
        relation_spec: Dict[str, Any],
        context_spec: Optional[Dict[str, Any]] = None,
        time_constraint_compliance: Optional[Dict[str, Any]] = None,
        value_constraint_compliance: Optional[Dict[str, Any]] = None,
    ):
        self.derived_map = derived_map
        self.relation_spec = relation_spec
        self.context_spec = context_spec or {}
        self.max_delta: Optional[timedelta] = None
        self.min_delta: Optional[timedelta] = None
        if relation_spec.get("max_distance"):
            self.max_delta = parse_duration(relation_spec["max_distance"])
        if relation_spec.get("min_distance"):
            self.min_delta = parse_duration(relation_spec["min_distance"])
        
        # Compliance functions (optional)
        self.time_constraint_compliance = time_constraint_compliance  # {func_name, trapez, parameters}
        self.value_constraint_compliance = value_constraint_compliance  # {func_name, trapez, targets, parameters}

    def matches(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        contexts: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Check if an anchor-event pair satisfies this temporal relation rule.
        
        Args:
            anchor_row: Anchor candidate (DataFrame row)
            event_row: Event candidate (DataFrame row)
            contexts: Optional context intervals (DataFrame)
        
        Returns:
            True if pair matches rule (temporal + context constraints)
        """
        # Check temporal relation
        if not self._temporal_match(anchor_row, event_row):
            return False
        
        # Check context (if defined)
        if not self._context_satisfied(anchor_row, event_row, contexts):
            return False
        
        return True

    def _temporal_match(self, anchor_row: pd.Series, event_row: pd.Series) -> bool:
        """
        Check if the temporal relation between anchor and event rows holds.
        Respects "overlap" and "before" relationships (with <max/min_distance> restrictions).
        """
        if self.relation_spec["how"] == "overlap":
            # Inclusive at the boundry
            return not (
                anchor_row["EndDateTime"] < event_row["StartDateTime"]
                or event_row["EndDateTime"] < anchor_row["StartDateTime"]
            )
        # how == "before"
        if anchor_row["EndDateTime"] > event_row["StartDateTime"]:
            return False
        delta = event_row["StartDateTime"] - anchor_row["EndDateTime"]
        if self.min_delta is not None and delta < self.min_delta:
            return False
        if self.max_delta is None:
            return True
        return delta <= self.max_delta

    def _context_satisfied(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        contexts: Optional[pd.DataFrame]
    ) -> bool:
        """
        Check if context requirements are satisfied.
        
        Context must:
        1. Have values matching the required constraints (equal checks only)
        2. Overlap the pattern timeframe [min_start, max_start]
        
        IMPORTANT: Pattern context blocks can only reference ONE context TAK (enforced in validation).
        
        Args:
            anchor_row: Anchor row
            event_row: Event row
            contexts: Context intervals DataFrame
        
        Returns:
            True if context requirements satisfied
        """
        # No context requirement → always satisfied
        if not self.context_spec or not self.context_spec.get("attributes"):
            return True
        
        # No context data → fail
        if contexts is None or contexts.empty:
            return False
        
        # ASSUMPTION: Only ONE context attribute (enforced in validation)
        # Extract the single context attribute specification
        
        attr_name, attr_spec = next(iter(self.context_spec["attributes"].items()))
        allowed_values = attr_spec.get("allowed_values", set())
        
        if not allowed_values:
            # No value constraint → just check temporal overlap
            attr_mask = (contexts["ConceptName"] == attr_name)
            if not attr_mask.any():
                return False
            matching_contexts = contexts[attr_mask]
        else:
            # Filter by ConceptName AND value (vectorized)
            concept_mask = (contexts["ConceptName"] == attr_name)
            value_mask = contexts["Value"].astype(str).isin(allowed_values)
            combined_mask = concept_mask & value_mask
            
            if not combined_mask.any():
                return False
            
            matching_contexts = contexts[combined_mask]
        
        # Check temporal overlap on filtered contexts
        min_start = min(anchor_row["StartDateTime"], event_row["StartDateTime"])
        max_start = max(anchor_row["StartDateTime"], event_row["StartDateTime"])
        
        overlap_mask = (matching_contexts["StartDateTime"] <= max_start) & (matching_contexts["EndDateTime"] >= min_start)
        
        return overlap_mask.any()


class CyclicRule(TAKRule):
    """
    Cyclic rule for GlobalPattern. Assures the inner-window conditions are met.
    Ensures relevant events (can be any concept) occur min/max times within time windows.
    """
    def __init__(
        self,
        start: str,
        end: str,
        time_window: str,
        min_occurrences: int,
        max_occurrences: int,
        initiator_spec: Dict[str, Any],
        event_spec: Dict[str, Any],
        clipper_spec: Dict[str, Any],
        context_spec: Optional[Dict[str, Any]] = None,
        cyclic_constraint_compliance: Optional[Dict[str, Any]] = None,
        value_constraint_compliance: Optional[Dict[str, Any]] = None,
    ):
        self.start = parse_duration(start)
        self.end = parse_duration(end)
        self.time_window = parse_duration(time_window)
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.event_spec = event_spec
        self.initiator_spec = initiator_spec
        self.clipper_spec = clipper_spec
        self.context_spec = context_spec
        self.cyclic_constraint_compliance = cyclic_constraint_compliance
        self.value_constraint_compliance = value_constraint_compliance

    def matches(self, event_row: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp, contexts: Optional[pd.DataFrame]) -> bool:
        """
        There is no event_row for cyclic rules; only window and context matter.
        For every window we'll check if the context is satisfied, and then filter events directly on the time frame.
        """
        pass

    def context_satisfied(self, window_start: pd.Timestamp, window_end: pd.Timestamp, contexts: Optional[pd.DataFrame]) -> bool:
        if not self.context_spec or not self.context_spec.get("attributes"):
            return True
        if contexts is None or contexts.empty:
            return False
        
        # ASSUMPTION: Only ONE context attribute (enforced in validation)
        attr_name, attr_spec = next(iter(self.context_spec["attributes"].items()))
        allowed_values = attr_spec.get("allowed_values", set())
        
        if not allowed_values:
            attr_mask = (contexts["ConceptName"] == attr_name)
            if not attr_mask.any(): return False
            matching_contexts = contexts[attr_mask]
        else:
            concept_mask = (contexts["ConceptName"] == attr_name)
            value_mask = contexts["Value"].astype(str).isin(allowed_values)
            combined_mask = concept_mask & value_mask
            if not combined_mask.any(): return False
            matching_contexts = contexts[combined_mask]
        
        # Check temporal overlap with window
        overlap_mask = (matching_contexts["StartDateTime"] < window_end) & (matching_contexts["EndDateTime"] > window_start)
        return overlap_mask.any()


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



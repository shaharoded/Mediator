from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union, Dict, List, Any
import pandas as pd
from abc import ABC, abstractmethod
from lxml import etree as lxml_etree
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# Import schema path from config
from ..config import TAK_SCHEMA_PATH

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
    family: Optional[Literal["raw-concept","state","trend","context","event","pattern"]] = None
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
    def __init__(self, value: str, operator: Literal["and","or"], constraints: Dict[int, List[str]]):
        self.value = value
        self.operator = operator.lower()
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
    def __init__(self, value: str, operator: Literal["and","or"], constraints: Dict[str, List[Dict[str, Any]]]):
        self.value = value
        self.operator = operator.lower()
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


# NEW: Schema validation helper
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



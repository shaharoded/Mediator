from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union, Dict, List, Any
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod

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
    Subclasses: DiscretizationRule, AbstractionRule, TrendRule, ContextRule, PatternRule.
    """
    @abstractmethod
    def matches(self, *args, **kwargs) -> bool:
        """Check if rule matches given input."""
        pass


class DiscretizationRule(TAKRule):
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


class AbstractionRule(TAKRule):
    """
    Logical rule to combine discrete attribute values → final state label.
    Example: value="SubCutaneous Low", operator="and", constraints={(0, ['Very Low','Low']), (1, ['SubCutaneous'])}
    """
    def __init__(self, value: str, operator: Literal["and","or"], constraints: Dict[int, List[str]]):
        self.value = value
        self.operator = operator.lower()
        self.constraints = constraints  # {attr_idx: [allowed_discrete_values]}

    def matches(self, discrete_tuple: Tuple[str, ...]) -> bool:
        """Check if a discrete tuple satisfies this rule."""
        results = []
        for idx, allowed in self.constraints.items():
            if idx >= len(discrete_tuple):
                results.append(False)
            else:
                results.append(discrete_tuple[idx] in allowed)
        
        if self.operator == "and":
            return all(results)
        elif self.operator == "or":
            return any(results)
        return False



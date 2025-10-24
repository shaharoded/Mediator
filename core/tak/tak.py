from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union
from pathlib import Path
import pandas as pd


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
        (with columns PatientId, ConceptName, StartTime, EndTime, Value).
        DF can be originated from InputPatientData (if TAK references a raw-concept) or OutputPatientData.
        Return df with the schema: 
        PatientId, 
        ConceptName = self.name, 
        StartTime, EndTime, 
        Value
        AbstractionType = self.family
        """
        raise NotImplementedError


class TAKRule():
    """
    Abstract base class for all TAK families to control execution of abstraction rules.
    Consumers: StateTAK, ContextTAK, TrendTAK, PatternTAK.
    """
    pass



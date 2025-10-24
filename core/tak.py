# core/tak.py
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Iterable, Literal, Union, Callable
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from .utils import parse_duration  # "8h" -> timedelta(hours=8)

# -------------------------------------------------------------------
# Base class
# -------------------------------------------------------------------

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

class RawConceptTAK(TAK):
    """
    Raw Concept TAK — parsed from <raw-concept> XML.
    - 'raw'          : multi-attr; requires <tuple>/<order> and <merge tolerance require-all>
    - 'raw-numeric'  : one or more numeric attrs; all must be numeric; ranges kept per-attr (min/max) in self.attributes
    - 'raw-nominal'  : one or more nominal attrs; all must be nominal; each must have <nominal-allowed-values>
    - 'raw-boolean'  : one or more boolean attrs; all must be boolean
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        concept_type: Literal["raw","raw-numeric","raw-nominal","raw-boolean"],
        attributes: List[Dict[str, Any]],
        tuple_order: Tuple[str, ...],
        merge_tolerance: Optional[timedelta],
        merge_require_all: bool = False,
    ):
        super().__init__(name=name, categories=categories, description=description, family="raw-concept")

        self.concept_type = concept_type
        self.attributes = attributes                    # [{name, type, min, max, allowed, map}]
        self.tuple_order = tuple_order                   # declared attribute order
        self.merge_tolerance = merge_tolerance
        self.merge_require_all = merge_require_all      # for 'raw' concepts: whether to require all attrs in merge

        # Runtime patient-level dataframe (filled in apply())
        self.df: Optional[pd.DataFrame] = None

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "RawConceptTAK":
        """Parse a <raw-concept> XML definition file."""
        root = ET.parse(xml_path).getroot()
        if root.tag != "raw-concept":
            raise ValueError(f"{xml_path} is not a raw-concept file")

        name = root.attrib["name"]
        concept_type = root.attrib.get("concept-type", "raw")
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- Attributes ---
        attributes: List[Dict[str, Any]] = []
        attrs_el = root.find("attributes")
        if attrs_el is not None:
            for a in attrs_el.findall("attribute"):
                spec = {
                    "name": a.attrib["name"],
                    "type": a.attrib["type"],  # numeric | nominal | boolean
                    "min": None,
                    "max": None,
                    "allowed": None,
                    "map": {},
                }

                # nominal values
                if spec["type"] == "nominal":
                    allowed_el = a.find("nominal-allowed-values")
                    if allowed_el is not None:
                        spec["allowed"] = tuple(v.attrib["value"] for v in allowed_el.findall("allowed-value"))
                    else:
                        raise ValueError(f"{xml_path}: nominal attribute '{spec['name']}' missing <nominal-allowed-values>")
                
                if spec["type"] == "numeric":
                    allowed_el = a.find("numeric-allowed-values")
                    if allowed_el is None:
                        raise ValueError(f"{xml_path}: numeric attribute '{spec['name']}' missing <numeric-allowed-values>")
                    av = allowed_el.find("allowed-value")
                    if av is None or ("min" not in av.attrib and "max" not in av.attrib):
                        raise ValueError(f"{xml_path}: <numeric-allowed-values> must contain <allowed-value min=... max=...>")
                    spec["min"] = float(av.attrib["min"]) if "min" in av.attrib else None
                    spec["max"] = float(av.attrib["max"]) if "max" in av.attrib else None

                # optional mapping
                map_el = a.find("map")
                if map_el is not None:
                    spec["map"] = {m.attrib["from"]: m.attrib["to"] for m in map_el.findall("entry")}

                attributes.append(spec)

        # --- Tuple order ---
        tuple_order = ()
        tuple_el = root.find("tuple-order")
        # Collect only if concept_type is "raw"
        if tuple_el is not None and concept_type == "raw":
            tuple_order = tuple(f.attrib["name"] for f in tuple_el.findall("attribute"))

        # --- Merge (raw only) ---
        merge_el = root.find("merge")
        tol: Optional[timedelta] = None
        require_all: bool = False
        # Collect only if concept_type is "raw"
        if merge_el is not None and concept_type == "raw":
            tol_val = merge_el.attrib.get("tolerance")
            if tol_val:
                tol = parse_duration(tol_val)
            require_all = (merge_el.attrib.get("require-all", "false").lower() == "true")
        
        # --- Type-specific structural checks ---
        if concept_type == "raw":
            if len(attributes) < 2:
                raise ValueError(f"{name}: raw concept must define ≥2 attributes")
            if tuple_el is None or not tuple_order:
                raise ValueError(f"{name}: raw concept must define <tuple-order> block")
            if merge_el is None or tol is None or require_all is None:
                raise ValueError(f"{name}: raw concept must define <merge tolerance=... require-all=...> block")
        
        elif concept_type == "raw-boolean":
            if not attributes or any(a["type"] != "boolean" for a in attributes):
                raise ValueError(f"{name}: raw-boolean must define at least one attribute and all must be boolean")

        elif concept_type == "raw-nominal":
            if not attributes or any(a["type"] != "nominal" for a in attributes):
                raise ValueError(f"{name}: raw-nominal must define at least one attribute and all must be nominal")

        elif concept_type == "raw-numeric":
            if not attributes or any(a["type"] != "numeric" for a in attributes):
                raise ValueError(f"{name}: raw-numeric must define at least one attribute and all must be numeric")

        # Construct object
        tak = cls(
            name=name,
            categories=cats,
            description=desc,
            concept_type=concept_type,
            attributes=attributes,
            tuple_order=tuple_order,
            merge_tolerance=tol,
            merge_require_all=require_all
        )

        tak.validate()  # semantic/business validation
        return tak

    def validate(self) -> None:
        """Validate internal consistency and business rules after parsing."""

        # --- Duplicate attribute names ---
        names = [a["name"] for a in self.attributes]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate attribute names in definition")

        # --- Allowed values logic ---
        for a in self.attributes:
            if a["type"] == "nominal":
                if len(a["allowed"]) != len(set(a["allowed"])):
                    raise ValueError(f"{self.name}: duplicate nominal allowed values for {a['name']}")
            elif a["type"] == "boolean":
                if a["allowed"] is not None:
                    raise ValueError(f"{self.name}: boolean attribute {a['name']} should not define allowed values")
            elif a["type"] == "numeric":
                if a["min"] is not None and a["max"] is not None:
                    if a["min"] >= a["max"]:
                        raise ValueError(f"{self.name}: invalid range for {a['name']} (min ≥ max)")
                elif a["min"] is None and a["max"] is None:
                    raise ValueError(f"{self.name}: numeric attribute {a['name']} must define at least min or max")

        # --- Tuple coherence ---
        if self.tuple_order and self.concept_type == "raw":
            declared = [a["name"] for a in self.attributes]
            # tuple must reference exactly the declared attributes (same set, same length)
            if set(self.tuple_order) != set(declared) or len(self.tuple_order) != len(declared):
                raise ValueError(f"{self.name}: <tuple-order> must list exactly all declared attributes")

        # --- All good ---
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Normalize per attribute
        attr_by_name = {a["name"]: a for a in self.attributes}

        def norm(row):
            cn = row["ConceptName"]
            if cn not in attr_by_name:
                return row
            spec = attr_by_name[cn]
            val = row["Value"]
            if spec["type"] == "numeric":
                try:
                    x = float(val)
                    if spec["min"] is not None and x < spec["min"]:
                        x = spec["min"]
                    if spec["max"] is not None and x > spec["max"]:
                        x = spec["max"]
                    row["Value"] = x
                except Exception:
                    pass
            elif spec["type"] == "nominal":
                if isinstance(val, str) and spec["map"]:
                    row["Value"] = spec["map"].get(val, val)
                if spec["allowed"] is not None and row["Value"] not in spec["allowed"]:
                    # drop invalid nominal (leave as-is or set None; choose drop for cleanliness)
                    row["Value"] = None
            return row

        df = df.apply(norm, axis=1).sort_values("StartTime").copy()

        # raw-merged → build tuples within tolerance window
        if self.concept_type == "raw-merged":
            tol = self.merge_tolerance
            order = self.tuple_order
            rows = []
            records = df.to_dict("records")
            i = 0
            while i < len(records):
                r = records[i]
                pid = r["PatientId"]
                t0 = r["StartTime"]
                window_end = t0 + tol
                bucket = {k: None for k in order}
                j = i
                # collect first seen value for each attr within tolerance
                while j < len(records) and records[j]["StartTime"] <= window_end:
                    rr = records[j]
                    nm = rr["ConceptName"]
                    if nm in bucket and bucket[nm] is None:
                        bucket[nm] = rr["Value"]
                    j += 1
                tup = tuple(bucket[k] for k in order)
                rows.append({
                    "PatientId": pid,
                    "ConceptName": self.name,
                    "StartTime": t0,
                    "EndTime": t0 + pd.Timedelta(seconds=1),
                    "Value": tup if len(order) > 1 else tup[0],
                    "Source": "abstracted",
                })
                i = j
            return pd.DataFrame(rows).sort_values("StartTime")

        # non-merged → pass-through with ConceptName = self.name
        out = df.copy()
        out["ConceptName"] = self.name
        out["Source"] = "abstracted"
        return out


class TAKRule(ABC):
    """
    Abstract base class for all TAK families.
    Subclasses: StateTAK, EventTAK, ContextTAK, TrendTAK, PatternTAK.
    """
    def __init__(self, abstraction_name, loinc_code, filters, persistence, rules):
        self.abstraction_name = abstraction_name
        self.loinc_code = loinc_code
        self.filters = filters  # dict: e.g., {'sex': 'Male', 'age_group': 'Adult'}
        self.good_before = parse_duration(persistence['before'])
        self.good_after = parse_duration(persistence['after'])
        self.rules = rules  # family-specific structure (e.g., thresholds for State)

    def applies_to(self, patient_params):
        """Check if rule applies to patient based on filters."""
        for key, value in self.filters.items():
            if key not in patient_params or str(patient_params[key]).lower() != str(value).lower():
                return False
        return True

    @abstractmethod
    def apply(self, df):
        """
        Apply TAK logic to a filtered DataFrame of raw concepts.
        
        Args:
            df (pd.DataFrame): Raw records for this patient/LOINC.
                Columns: [PatientId, ConceptName, StartDateTime, EndDateTime, Value, Unit]
        
        Returns:
            pd.DataFrame: Abstracted intervals.
                Columns: [PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType, Source]
        """
        pass



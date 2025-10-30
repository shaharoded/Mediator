from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, get_tak_repository, validate_xml_against_schema
from .raw_concept import RawConcept
from .event import Event
from .context import Context


class Pattern(TAK):
    """
    Base class for all pattern TAKs (local/global).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        abstraction_rules: List[Dict[str, Any]],
        pattern_type: str = "pattern"
    ):
        super().__init__(name=name, categories=categories, description=description, family=pattern_type)
        self.derived_from = derived_from
        self.abstraction_rules = abstraction_rules

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Pattern":
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

    def validate(self) -> None:
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

class LocalPattern(Pattern):
    """
    LocalPattern: interval-based pattern abstraction.
    Output values: True, Partial, False (False only for QA).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        abstraction_rules: List[Dict[str, Any]]
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            derived_from=derived_from,
            abstraction_rules=abstraction_rules,
            pattern_type="local-pattern"
        )

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "LocalPattern":
        xml_path = Path(xml_path)
        validate_xml_against_schema(xml_path)
        root = ET.parse(xml_path).getroot()
        if root.tag != "pattern":
            raise ValueError(f"{xml_path} is not a pattern file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required, one or more) ---
        df_el = root.find("derived-from")
        if df_el is None:
            raise ValueError(f"{name}: missing <derived-from> block")
        derived_from = []
        for attr_el in df_el.findall("attribute"):
            if "name" not in attr_el.attrib or "tak" not in attr_el.attrib:
                raise ValueError(f"{name}: <derived-from><attribute> must have 'name' and 'tak' attributes")
            spec = {
                "name": attr_el.attrib["name"],
                "tak_type": attr_el.attrib["tak"],
                "idx": int(attr_el.attrib.get("idx", 0))
            }
            derived_from.append(spec)
        if not derived_from:
            raise ValueError(f"{name}: <derived-from> must contain at least one <attribute>")

        # --- abstraction-rules (required) ---
        abs_el = root.find("abstraction-rules")
        if abs_el is None:
            raise ValueError(f"{name}: <pattern> must define <abstraction-rules>")
        abstraction_rules = []
        for rule_el in abs_el.findall("rule"):
            op = rule_el.attrib.get("operator", "and")
            # Each rule must have at least one <temporal-relation>
            temporal_relations = []
            for tr_el in rule_el.findall("temporal-relation"):
                how = tr_el.attrib.get("how")
                max_distance = tr_el.attrib.get("max-distance")
                attrs = []
                for attr_el in tr_el.findall("attribute"):
                    attr_name = attr_el.attrib["name"]
                    idx = int(attr_el.attrib.get("idx", 0))
                    allowed = []
                    for av in attr_el.findall("allowed-value"):
                        constraint = {}
                        if "equal" in av.attrib:
                            constraint["type"] = "equal"
                            constraint["value"] = av.attrib["equal"]
                        if "min" in av.attrib:
                            constraint["type"] = "min"
                            constraint["value"] = float(av.attrib["min"])
                        if "max" in av.attrib:
                            constraint["type"] = "max"
                            constraint["value"] = float(av.attrib["max"])
                        allowed.append(constraint)
                    attrs.append({
                        "name": attr_name,
                        "idx": idx,
                        "allowed": allowed
                    })
                temporal_relations.append({
                    "how": how,
                    "max_distance": max_distance,
                    "attributes": attrs
                })
            # Optional compliance-function
            compliance_func = None
            cf_el = rule_el.find("compliance-function")
            if cf_el is not None:
                tc_el = cf_el.find("time-constraint-compliance")
                if tc_el is not None:
                    compliance_func = {k: tc_el.attrib[k] for k in tc_el.attrib}
            abstraction_rules.append({
                "operator": op,
                "temporal_relations": temporal_relations,
                "compliance_function": compliance_func
            })

        pattern = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            abstraction_rules=abstraction_rules
        )
        pattern.validate()
        return pattern

    def validate(self) -> None:
        repo = get_tak_repository()
        # 1) All derived-from TAKs must exist
        for df in self.derived_from:
            tak = repo.get(df["name"])
            if tak is None:
                raise ValueError(f"{self.name}: derived_from TAK '{df['name']}' not found in repository")
            if df["tak_type"] == "raw-concept":
                # Must mention idx in abstraction rules if referenced
                if not any(
                    any(attr["name"] == df["name"] and "idx" in attr for tr in rule["temporal_relations"] for attr in tr["attributes"])
                    for rule in self.abstraction_rules
                ):
                    raise ValueError(f"{self.name}: raw-concept '{df['name']}' must be referenced with idx in abstraction rules")
        # 2) Must have abstraction rules
        if not self.abstraction_rules:
            raise ValueError(f"{self.name}: LocalPattern must define abstraction rules")
        # 3) Each rule must have at least one temporal-relation
        for rule in self.abstraction_rules:
            if not rule["temporal_relations"]:
                raise ValueError(f"{self.name}: abstraction rule missing temporal-relation")
            for tr in rule["temporal_relations"]:
                if not tr["attributes"]:
                    raise ValueError(f"{self.name}: temporal-relation missing attributes")
                for attr in tr["attributes"]:
                    tak = repo.get(attr["name"])
                    if tak is None:
                        raise ValueError(f"{self.name}: referenced attribute TAK '{attr['name']}' not found")
                    # Only raw-concept numeric can use min/max constraints
                    for constraint in attr["allowed"]:
                        if constraint["type"] in ("min", "max"):
                            if not (isinstance(tak, RawConcept) and any(a["type"] == "numeric" for a in tak.attributes)):
                                raise ValueError(f"{self.name}: min/max constraint only allowed for numeric raw-concept attributes")
        # ...other business logic as needed...

class GlobalPattern(Pattern):
    pass  # Not implemented yet
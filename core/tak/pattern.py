from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, get_tak_repository, validate_xml_against_schema, TemporalRelationRule, QATPRule
from .raw_concept import RawConcept
from .event import Event
from .context import Context
from .utils import apply_external_function, parse_duration
from .external_functions import REPO

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
        family: str = "pattern"
    ):
        super().__init__(name=name, categories=categories, description=description, family=family)
        self.derived_from = derived_from
        self.abstraction_rules = abstraction_rules

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Pattern":
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

    def validate(self) -> None:
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

class LocalPattern(Pattern):
    """
    LocalPattern: interval-based pattern abstraction with ref-based attributes.
    Output values: True, Partial, False (False only for QATPs).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        abstraction_rules: List[Dict[str, Any]]
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            derived_from=derived_from,
            abstraction_rules=abstraction_rules,
            family="local-pattern"
        )
        self.parameters = parameters

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
        declared_refs = set()
        
        for attr_el in df_el.findall("attribute"):
            if "name" not in attr_el.attrib or "tak" not in attr_el.attrib or "ref" not in attr_el.attrib:
                raise ValueError(f"{name}: <derived-from><attribute> must have 'name', 'tak', and 'ref' attributes")
            
            ref = attr_el.attrib["ref"]
            if ref in declared_refs:
                raise ValueError(f"{name}: duplicate ref '{ref}' in derived-from block")
            declared_refs.add(ref)
            
            tak_type = attr_el.attrib["tak"]
            idx = int(attr_el.attrib.get("idx")) if attr_el.attrib.get("idx") is not None else None
            if tak_type == 'raw-concept' and idx is None:
                raise ValueError(f"{name}: 'raw-concept' attributes must declare idx for tuple parse")
            
            spec = {
                "ref": ref,
                "name": attr_el.attrib["name"],
                "tak": tak_type,
                "idx": idx
            }
            derived_from.append(spec)
        
        if not derived_from:
            raise ValueError(f"{name}: <derived-from> must contain at least one <attribute>")

        # --- parameters (optional) ---
        parameters = []
        params_el = root.find("parameters")
        if params_el is not None:
            for param_el in params_el.findall("parameter"):
                if "name" not in param_el.attrib or "ref" not in param_el.attrib or "default" not in param_el.attrib:
                    raise ValueError(f"{name}: <parameter> must have 'name', 'ref', and 'default' attributes")
                
                ref = param_el.attrib["ref"]
                if ref in declared_refs:
                    raise ValueError(f"{name}: parameter ref '{ref}' conflicts with derived-from ref")
                declared_refs.add(ref)

                tak_type = param_el.attrib["tak"]
                idx = int(param_el.attrib.get("idx")) if param_el.attrib.get("idx") is not None else None
                if tak_type == 'raw-concept' and idx is None:
                    raise ValueError(f"{name}: 'raw-concept' parameter must declare idx for tuple parse")
                
                param_spec = {
                    "ref": ref,
                    "name": param_el.attrib["name"],
                    "tak": tak_type,
                    "idx": idx,
                    "default": param_el.attrib.get("default")
                }
                parameters.append(param_spec)

        # --- abstraction-rules (required) ---
        abs_el = root.find("abstraction-rules")
        if abs_el is None:
            raise ValueError(f"{name}: <pattern> must define <abstraction-rules>")
        abstraction_rules = []
        
        for rule_el in abs_el.findall("rule"):
            rule_spec = {}

            # --- context (optional) ---
            context_el = rule_el.find("context")
            if context_el is not None:
                context_spec = {}
                for attr_el in context_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref:
                        raise ValueError(f"{name}: context attribute must have 'ref' attribute")
                    allowed_values = set()
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(av.attrib["equal"])
                        else:
                            raise ValueError(f"{name}: only 'equal' constraints supported in context attributes")
                    context_spec[ref] = allowed_values
                rule_spec["context"] = context_spec

            # --- temporal-relation (required) ---
            tr_el = rule_el.find("temporal-relation")
            if tr_el is None:
                raise ValueError(f"{name}: rule missing <temporal-relation>")
            how = tr_el.attrib.get("how")
            if how not in ("before", "overlap"):
                raise ValueError(f"{name}: temporal-relation how='{how}' must be 'before' or 'overlap'")
            max_distance = tr_el.attrib.get("max-distance")
            if how == "before" and not max_distance:
                raise ValueError(f"{name}: temporal-relation how='before' requires max-distance")
            temporal_relation = {
                "how": how,
                "max_distance": max_distance
            }

            # anchor
            anchor_el = tr_el.find("anchor")
            if anchor_el is not None:
                anchor_spec = {
                    "select": anchor_el.attrib.get("select", "first")
                }
                anchor_attrs = {}
                for attr_el in anchor_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref:
                        raise ValueError(f"{name}: anchor attribute must have 'ref' attribute")
                    allowed_values = set()
                    min_val = None
                    max_val = None
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(str(av.attrib["equal"]))
                        if "min" in av.attrib:
                            min_val = float(av.attrib["min"])
                        if "max" in av.attrib:
                            max_val = float(av.attrib["max"])
                    anchor_attrs[ref] = {
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                anchor_spec["attributes"] = anchor_attrs
                # Check if more than 1 attr, select must be mentioned
                if len(anchor_attrs) > 1 and not anchor_spec.get("select"):
                    raise ValueError(f"{name}: anchor with multiple attributes requires explicit select attribute")
                temporal_relation["anchor"] = anchor_spec

            # event
            event_el = tr_el.find("event")
            if event_el is not None:
                event_spec = {
                    "select": event_el.attrib.get("select", "first")
                }
                event_attrs = {}
                for attr_el in event_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref:
                        raise ValueError(f"{name}: event attribute must have 'ref' attribute")
                    allowed_values = set()
                    min_val = None
                    max_val = None
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(av.attrib["equal"])
                        if "min" in av.attrib:
                            min_val = float(av.attrib["min"])
                        if "max" in av.attrib:
                            max_val = float(av.attrib["max"])
                    event_attrs[ref] = {
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                event_spec["attributes"] = event_attrs
                # Check if more than 1 attr, select must be mentioned
                if len(event_attrs) > 1 and not event_spec.get("select"):
                    raise ValueError(f"{name}: event with multiple attributes requires explicit select attribute")
                temporal_relation["event"] = event_spec

            rule_spec["temporal_relation"] = temporal_relation

            # --- compliance-function (optional) ---
            cf_el = rule_el.find("compliance-function")
            if cf_el is not None:
                compliance_function = []

                # time-constraint-compliance
                tcc_el = cf_el.find("time-constraint-compliance")
                if tcc_el is not None:
                    if how != 'before':
                        raise ValueError(f"{name}: time-constraint-compliance only valid for 'before' temporal relation")
                    func_el = tcc_el.find("function")
                    if func_el is not None:
                        func_name = func_el.attrib["name"]
                        # Check function is in REPO
                        if func_name not in REPO:
                            raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions")
                        
                        # Parse raw trapez
                        trapez_el = func_el.find("trapeze")
                        trapez_raw = (
                            trapez_el.attrib["trapezeA"],
                            trapez_el.attrib["trapezeB"],
                            trapez_el.attrib["trapezeC"],
                            trapez_el.attrib["trapezeD"]
                        )
                        
                        # Extract parameters (if any)
                        param_refs = []
                        parameters_el = func_el.find("parameters")
                        if parameters_el is not None:
                            for p in parameters_el.findall("parameter"):
                                ref = p.attrib.get("ref")
                                if ref not in declared_refs:
                                    raise ValueError(f"{name}: parameter ref '{ref}' not declared")
                                param_refs.append(ref)
                        
                        compliance_function.append({
                                "func_name": func_name,
                                "trapez": trapez_raw,
                                "constraint_type": "time-constraint",
                                "parameters": param_refs
                            })
                    else:
                        raise ValueError(f"{name}: time-constraint-compliance missing <function> element. Use 'id' as default.")

                # value-constraint-compliance
                vcc_el = cf_el.find("value-constraint-compliance")
                if vcc_el is not None:
                    func_el = vcc_el.find("function")
                    if func_el is not None:
                        func_name = func_el.attrib["name"]
                        if func_name not in REPO:
                            raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions")
                        
                        # Parse raw trapez (numeric, not time)
                        trapez_el = func_el.find("trapeze")
                        trapez_raw = (
                            float(trapez_el.attrib["trapezeA"]),
                            float(trapez_el.attrib["trapezeB"]),
                            float(trapez_el.attrib["trapezeC"]),
                            float(trapez_el.attrib["trapezeD"])
                        )
                        
                        # Extract target refs
                        target_refs = []
                        target_el = vcc_el.find("target")
                        if target_el is not None:
                            for attr_el in target_el.findall("attribute"):
                                ref = attr_el.attrib.get("ref")
                                if ref not in declared_refs:
                                    raise ValueError(f"{name}: target ref '{ref}' not declared")
                                target_refs.append(ref)
                        else:
                            raise ValueError(f"{name}: value-constraint-compliance missing <target> block with <attribute ref=.../> elements.")
                        
                        # Extract parameter refs
                        param_refs = []
                        parameters_el = func_el.find("parameters")
                        if parameters_el is not None:
                            for p in parameters_el.findall("parameter"):
                                ref = p.attrib.get("ref")
                                if ref not in declared_refs:
                                    raise ValueError(f"{name}: parameter ref '{ref}' not declared")
                                param_refs.append(ref)
                        
                        compliance_function.append({
                            "func_name": func_name,
                            "trapez": trapez_raw,
                            "targets": target_refs,
                            "constraint_type": "value-constraint",
                            "parameters": param_refs
                        })
                    else:
                        raise ValueError(f"{name}: value-constraint-compliance missing <function> element. Use 'id' as default.")

                rule_spec["compliance_function"] = compliance_function

            abstraction_rules.append(rule_spec)

        pattern = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            parameters=parameters,
            abstraction_rules=abstraction_rules
        )
        pattern.validate()
        return pattern

    def validate(self) -> None:
        """Validate ref-based pattern structure and constraints."""
        repo = get_tak_repository()
        
        # Build lookup maps: ref → derived_from entry
        derived_from_by_ref = {item["ref"]: item for item in self.derived_from}
        parameters_by_ref = {item["ref"]: item for item in self.parameters}
        all_declared_refs = set(derived_from_by_ref.keys()) | set(parameters_by_ref.keys())
        
        # 1) Validate all derived-from TAKs exist and referenced correctly
        for df in self.derived_from + self.parameters:
            tak = repo.get(df["name"])
            if tak is None:
                raise ValueError(f"{self.name}: derived-from TAK '{df['name']}' (ref='{df['ref']}') not found in TAK repository")
            
            # If referencing raw-concept and idx specified, validate idx
            if isinstance(tak, RawConcept):
                if tak.concept_type == "raw":
                    if df["idx"] >= len(tak.tuple_order):
                        raise ValueError(f"{self.name}: ref '{df['ref']}' idx={df['idx']} out of bounds for '{df['name']}' (tuple_size={len(tak.tuple_order)})")
                else:
                    # raw-numeric, raw-nominal, raw-boolean: only idx=0 valid
                    if df["idx"] != 0:
                        raise ValueError(f"{self.name}: ref '{df['ref']}' idx={df['idx']} invalid for '{df['name']}' of type '{tak.concept_type}'; only idx=0 allowed")
        
        # 2) Validate all used refs are declared
        for rule in self.abstraction_rules:
            # Check context refs
            if "context" in rule:
                for ref in rule["context"]:
                    if ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: context ref '{ref}' not declared in derived-from or parameters")
            
            tr = rule["temporal_relation"]
            # Check anchor refs
            for ref in tr.get("anchor", {}).get("attributes", {}):
                if ref not in all_declared_refs:
                    raise ValueError(f"{self.name}: anchor ref '{ref}' not declared")
            # Check event refs
            for ref in tr.get("event", {}).get("attributes", {}):
                if ref not in all_declared_refs:
                    raise ValueError(f"{self.name}: event ref '{ref}' not declared")
            # Check compliance function parameter refs
            cf = rule.get("compliance_function", {})
            for cc_type in ("time_constraint", "value_constraint"):
                cc_spec = cf.get(cc_type, {})
                for param in cc_spec.get("parameters", []):
                    if param["ref"] not in all_declared_refs:
                        raise ValueError(f"{self.name}: compliance function parameter ref '{param['ref']}' not declared")
        
        # 3) Validate numeric range constraints match TAK's attribute types
        def _get_parent_attribute(tak, idx):
            """Extract the parent attribute spec from a TAK at the given idx."""
            if isinstance(tak, RawConcept):
                if tak.concept_type == "raw":
                    attr_name = tak.tuple_order[idx]
                    return next((a for a in tak.attributes if a["name"] == attr_name), None)
                else:
                    # raw-numeric, raw-nominal, raw-boolean: single attribute at idx=0
                    return tak.attributes[0] if tak.attributes else None
            return None
        
        for rule in self.abstraction_rules:
            tr = rule["temporal_relation"]
            
            # Validate anchor attributes
            for ref, attr_spec in tr.get("anchor", {}).get("attributes", {}).items():
                if ref not in derived_from_by_ref:
                    continue
                df = derived_from_by_ref[ref]
                tak = repo.get(df["name"])
                if not isinstance(tak, RawConcept):
                    continue
                
                parent_attr = _get_parent_attribute(tak, df["idx"])
                if parent_attr is None:
                    continue
                
                # Reject min/max on non-numeric; reject equal on numeric
                if parent_attr["type"] != "numeric":
                    if attr_spec.get("min") is not None or attr_spec.get("max") is not None:
                        raise ValueError(
                            f"{self.name}: anchor ref '{ref}' has min/max constraints but attribute '{parent_attr['name']}' "
                            f"is {parent_attr['type']} (not numeric). Use only 'equal' constraints for non-numeric attributes."
                        )
                else:
                    # numeric: reject equal, allow min/max
                    if attr_spec.get("allowed_values"):
                        raise ValueError(
                            f"{self.name}: anchor ref '{ref}' has 'equal' constraint but attribute '{parent_attr['name']}' "
                            f"is numeric. Use only 'min'/'max' constraints for numeric attributes."
                        )
                    # Check range overlap with TAK's range
                    tak_min = parent_attr.get("min")
                    tak_max = parent_attr.get("max")
                    rule_min = attr_spec.get("min")
                    rule_max = attr_spec.get("max")
                    
                    if rule_min is not None and tak_max is not None and rule_min > tak_max:
                        raise ValueError(f"{self.name}: anchor ref '{ref}' min={rule_min} exceeds TAK max={tak_max}")
                    if rule_max is not None and tak_min is not None and rule_max < tak_min:
                        raise ValueError(f"{self.name}: anchor ref '{ref}' max={rule_max} below TAK min={tak_min}")
            
            # Validate event attributes (same logic)
            for ref, attr_spec in tr.get("event", {}).get("attributes", {}).items():
                if ref not in derived_from_by_ref:
                    continue
                df = derived_from_by_ref[ref]
                tak = repo.get(df["name"])
                if not isinstance(tak, RawConcept):
                    continue
                
                parent_attr = _get_parent_attribute(tak, df["idx"])
                if parent_attr is None:
                    continue
                
                if parent_attr["type"] != "numeric":
                    if attr_spec.get("min") is not None or attr_spec.get("max") is not None:
                        raise ValueError(
                            f"{self.name}: event ref '{ref}' has min/max constraints but attribute '{parent_attr['name']}' "
                            f"is {parent_attr['type']} (not numeric). Use only 'equal' constraints for non-numeric attributes."
                        )
                else:
                    if attr_spec.get("allowed_values"):
                        raise ValueError(
                            f"{self.name}: event ref '{ref}' has 'equal' constraint but attribute '{parent_attr['name']}' "
                            f"is numeric. Use only 'min'/'max' constraints for numeric attributes."
                        )
                    tak_min = parent_attr.get("min")
                    tak_max = parent_attr.get("max")
                    rule_min = attr_spec.get("min")
                    rule_max = attr_spec.get("max")
                    
                    if rule_min is not None and tak_max is not None and rule_min > tak_max:
                        raise ValueError(f"{self.name}: event ref '{ref}' min={rule_min} exceeds TAK max={tak_max}")
                    if rule_max is not None and tak_min is not None and rule_max < tak_min:
                        raise ValueError(f"{self.name}: event ref '{ref}' max={rule_max} below TAK min={tak_min}")

            # Validate context attributes (nominal only, so reject min/max)
            context_attrs = rule.get("context", {}).get("attributes", {})
            for ref, attr_spec in context_attrs.items():
                if ref not in derived_from_by_ref:
                    continue
                df = derived_from_by_ref[ref]
                tak = repo.get(df["name"])
                if not isinstance(tak, RawConcept):
                    continue
                
                parent_attr = _get_parent_attribute(tak, df["idx"])
                if parent_attr is None:
                    continue
                
                # Context only supports nominal (no min/max)
                if attr_spec.get("min") is not None or attr_spec.get("max") is not None:
                    raise ValueError(
                        f"{self.name}: context ref '{ref}' has min/max constraints. Context attributes are nominal only; "
                        f"use only 'equal' constraints."
                    )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LocalPattern to find and rate pattern instances.
        Input: Patient data (from derived_from TAKs)
        Output: Intervals with Value=True/Partial/False (False only for QA)
        """
        if df.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
        return df
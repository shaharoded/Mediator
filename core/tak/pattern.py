from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, validate_xml_against_schema, TemporalRelationRule
from .utils import apply_external_function, parse_duration
from .repository import get_tak_repository
from .raw_concept import RawConcept
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
        family: str = "pattern"
    ):
        super().__init__(name=name, categories=categories, description=description, family=family)
        self.derived_from = derived_from

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
        abstraction_rules: List[TemporalRelationRule]
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            derived_from=derived_from,
            family="local-pattern"
        )
        self.parameters = parameters
        self.abstraction_rules = abstraction_rules  # List of TemporalRelationRule objects
        
        # Build ref lookup maps for quick access during apply()
        self.derived_from_map: Dict[str, Dict[str, Any]] = {
            df["ref"]: df for df in derived_from
        }
        self.parameters_map: Dict[str, Dict[str, Any]] = {
            p["ref"]: p for p in parameters
        }

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
            
            # Only require idx for raw-concept (contexts/events emit single values)
            if tak_type == 'raw-concept':
                idx = int(attr_el.attrib.get("idx")) if attr_el.attrib.get("idx") is not None else None
                if idx is None:
                    raise ValueError(f"{name}: 'raw-concept' attributes must declare idx for tuple parse")
            else:
                # Context, Event, State, etc: always single-value (default idx=0)
                idx = 0
            
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
                
                # Only require idx for raw-concept
                if tak_type == 'raw-concept':
                    idx = int(param_el.attrib.get("idx")) if param_el.attrib.get("idx") is not None else None
                    if idx is None:
                        raise ValueError(f"{name}: 'raw-concept' parameter must declare idx for tuple parse")
                else:
                    # Context, Event, State, etc: always single-value
                    idx = 0
                
                param_spec = {
                    "ref": ref,
                    "name": param_el.attrib["name"],
                    "tak": tak_type,
                    "idx": idx,
                    "default": param_el.attrib.get("default")
                }
                parameters.append(param_spec)

        # --- abstraction-rules: Parse as TemporalRelationRule objects ---
        abs_el = root.find("abstraction-rules")
        if abs_el is None:
            raise ValueError(f"{name}: <pattern> must define <abstraction-rules>")
        
        abstraction_rules: List[TemporalRelationRule] = []
        
        for rule_el in abs_el.findall("rule"):
            # --- Parse temporal-relation ---
            tr_el = rule_el.find("temporal-relation")
            if tr_el is None:
                raise ValueError(f"{name}: rule missing <temporal-relation>")
            
            how = tr_el.attrib.get("how")
            if how not in ("before", "overlap"):
                raise ValueError(f"{name}: temporal-relation how='{how}' must be 'before' or 'overlap'")
            max_distance = tr_el.attrib.get("max-distance")
            if how == "before" and not max_distance:
                raise ValueError(f"{name}: temporal-relation how='before' requires max-distance")
            
            relation_spec = {
                "how": how,
                "max_distance": max_distance
            }
            
            # Parse anchor
            anchor_el = tr_el.find("anchor")
            if anchor_el is not None:
                anchor_spec = {
                    "select": anchor_el.attrib.get("select", "first"),
                    "attributes": {}
                }
                for attr_el in anchor_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref or ref not in declared_refs:
                        raise ValueError(f"{name}: anchor attribute ref='{ref}' not declared")
                    
                    # Convert ref → name (for TemporalRelationRule compatibility)
                    df_entry = next((d for d in derived_from if d["ref"] == ref), None)
                    attr_name = df_entry["name"]
                    
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
                    
                    anchor_spec["attributes"][attr_name] = {
                        "idx": df_entry["idx"],
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                
                if len(anchor_spec["attributes"]) > 1 and not anchor_spec.get("select"):
                    raise ValueError(f"{name}: anchor with multiple attributes requires explicit select attribute")
                relation_spec["anchor"] = anchor_spec
            
            # Parse event (same logic as anchor)
            event_el = tr_el.find("event")
            if event_el is not None:
                event_spec = {
                    "select": event_el.attrib.get("select", "first"),
                    "attributes": {}
                }
                for attr_el in event_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref or ref not in declared_refs:
                        raise ValueError(f"{name}: event attribute ref='{ref}' not declared")
                    
                    df_entry = next((d for d in derived_from if d["ref"] == ref), None)
                    attr_name = df_entry["name"]
                    
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
                    
                    event_spec["attributes"][attr_name] = {
                        "idx": df_entry["idx"],
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                
                if len(event_spec["attributes"]) > 1 and not event_spec.get("select"):
                    raise ValueError(f"{name}: event with multiple attributes requires explicit select attribute")
                relation_spec["event"] = event_spec
            
            # --- Parse context (optional) ---
            context_spec = None
            context_el = rule_el.find("context")
            if context_el is not None:
                context_spec = {"attributes": {}}
                for attr_el in context_el.findall("attribute"):
                    ref = attr_el.attrib.get("ref")
                    if not ref or ref not in declared_refs:
                        raise ValueError(f"{name}: context attribute ref='{ref}' not declared")
                    
                    df_entry = next((d for d in derived_from if d["ref"] == ref), None)
                    attr_name = df_entry["name"]
                    
                    allowed_values = set()
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(av.attrib["equal"])
                        else:
                            raise ValueError(f"{name}: only 'equal' constraints supported in context attributes")
                    
                    context_spec["attributes"][attr_name] = {
                        "idx": df_entry["idx"],
                        "allowed_values": allowed_values
                    }
            
            # Build derived_map (name → df_entry) for TemporalRelationRule
            derived_map = {d["name"]: d for d in derived_from}
            
            # --- Parse compliance-function (optional) ---
            time_constraint_compliance = None
            value_constraint_compliance = None
            
            cf_el = rule_el.find("compliance-function")
            if cf_el is not None:
                # Parse time-constraint-compliance
                tcc_el = cf_el.find("time-constraint-compliance")
                if tcc_el is not None:
                    if how != 'before':
                        raise ValueError(f"{name}: time-constraint-compliance only valid for 'before' temporal relation")
                    
                    func_el = tcc_el.find("function")
                    if func_el is None:
                        raise ValueError(f"{name}: time-constraint-compliance missing <function> element")
                    
                    func_name = func_el.attrib.get("name")
                    if not func_name:
                        raise ValueError(f"{name}: time-constraint-compliance <function> missing 'name' attribute")
                    
                    # Check function exists in REPO
                    if func_name not in REPO:
                        raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions")
                    
                    # Parse trapeze (time-based, stored as strings)
                    trapez_el = func_el.find("trapeze")
                    if trapez_el is None:
                        raise ValueError(f"{name}: time-constraint-compliance <function> missing <trapeze>")
                    
                    trapez_raw = (
                        trapez_el.attrib.get("trapezeA"),
                        trapez_el.attrib.get("trapezeB"),
                        trapez_el.attrib.get("trapezeC"),
                        trapez_el.attrib.get("trapezeD")
                    )
                    
                    if None in trapez_raw:
                        raise ValueError(f"{name}: time-constraint-compliance trapeze must have all 4 attributes (A, B, C, D)")
                    
                    # Validate max_distance >= trapezeD (pattern must capture all valid instances)
                    if parse_duration(trapez_raw[3]) > parse_duration(max_distance):
                        raise ValueError(
                            f"{name}: temporal-relation max-distance '{max_distance}' must be >= "
                            f"time-constraint-compliance trapezeD '{trapez_raw[3]}' "
                            f"(otherwise pattern may miss valid instances)"
                        )
                    
                    # Extract parameter refs (if any)
                    param_refs = []
                    parameters_el = func_el.find("parameters")
                    if parameters_el is not None:
                        for p in parameters_el.findall("parameter"):
                            ref = p.attrib.get("ref")
                            if not ref or ref not in declared_refs:
                                raise ValueError(f"{name}: parameter ref '{ref}' not declared in derived-from or parameters")
                            param_refs.append(ref)
                    
                    time_constraint_compliance = {
                        "func_name": func_name,
                        "trapez": trapez_raw,  # Store as strings (will be processed by apply_external_function)
                        "parameters": param_refs
                    }
                
                # Parse value-constraint-compliance
                vcc_el = cf_el.find("value-constraint-compliance")
                if vcc_el is not None:
                    # Extract target refs (required for value constraints)
                    target_refs = []
                    target_el = vcc_el.find("target")
                    if target_el is None:
                        raise ValueError(f"{name}: value-constraint-compliance missing <target> block")
                    
                    for attr_el in target_el.findall("attribute"):
                        ref = attr_el.attrib.get("ref")
                        if not ref or ref not in declared_refs:
                            raise ValueError(f"{name}: target ref '{ref}' not declared in derived-from or parameters")
                        target_refs.append(ref)
                    
                    if not target_refs:
                        raise ValueError(f"{name}: value-constraint-compliance <target> must contain at least one <attribute ref=.../>")
                    
                    # Parse function element
                    func_el = vcc_el.find("function")
                    if func_el is None:
                        raise ValueError(f"{name}: value-constraint-compliance missing <function> element")
                    
                    func_name = func_el.attrib.get("name")
                    if not func_name:
                        raise ValueError(f"{name}: value-constraint-compliance <function> missing 'name' attribute")
                    
                    if func_name not in REPO:
                        raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions")
                    
                    # Parse trapeze (numeric values)
                    trapez_el = func_el.find("trapeze")
                    if trapez_el is None:
                        raise ValueError(f"{name}: value-constraint-compliance <function> missing <trapeze>")
                    
                    try:
                        trapez_raw = (
                            float(trapez_el.attrib.get("trapezeA")),
                            float(trapez_el.attrib.get("trapezeB")),
                            float(trapez_el.attrib.get("trapezeC")),
                            float(trapez_el.attrib.get("trapezeD"))
                        )
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"{name}: value-constraint-compliance trapeze values must be numeric: {e}")
                    
                    # Extract parameter refs (if any)
                    param_refs = []
                    for p in func_el.findall("parameter"):
                        ref = p.attrib.get("ref")
                        if not ref or ref not in declared_refs:
                            raise ValueError(f"{name}: parameter ref '{ref}' not declared in derived-from or parameters")
                        param_refs.append(ref)
                    
                    value_constraint_compliance = {
                        "func_name": func_name,
                        "trapez": trapez_raw,  # Stored as floats
                        "targets": target_refs,
                        "parameters": param_refs
                    }
            
            # Create TemporalRelationRule with compliance functions
            rule = TemporalRelationRule(
                derived_map=derived_map,
                relation_spec=relation_spec,
                context_spec=context_spec,
                time_constraint_compliance=time_constraint_compliance,
                value_constraint_compliance=value_constraint_compliance
            )
            
            abstraction_rules.append(rule)

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
            if rule.context_spec:
                for attr_name in rule.context_spec.get("attributes", {}):
                    # Find ref from derived_from that matches this attr_name
                    df_entry = next((d for d in self.derived_from if d["name"] == attr_name), None)
                    if df_entry is None:
                        raise ValueError(f"{self.name}: context references unknown attribute '{attr_name}'")
                    ref = df_entry["ref"]
                    if ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: context uses undeclared ref '{ref}'")
            
            tr = rule.relation_spec
            # Check anchor refs
            for attr_name in tr.get("anchor", {}).get("attributes", {}):
                df_entry = next((d for d in self.derived_from if d["name"] == attr_name), None)
                if df_entry is None:
                    raise ValueError(f"{self.name}: anchor references unknown attribute '{attr_name}'")
                ref = df_entry["ref"]
                if ref not in all_declared_refs:
                    raise ValueError(f"{self.name}: anchor uses undeclared ref '{ref}'")
            
            # Check event refs
            for attr_name in tr.get("event", {}).get("attributes", {}):
                df_entry = next((d for d in self.derived_from if d["name"] == attr_name), None)
                if df_entry is None:
                    raise ValueError(f"{self.name}: event references unknown attribute '{attr_name}'")
                ref = df_entry["ref"]
                if ref not in all_declared_refs:
                    raise ValueError(f"{self.name}: event uses undeclared ref '{ref}'")
            
            # Check compliance function parameter refs
            if rule.time_constraint_compliance:
                for param_ref in rule.time_constraint_compliance.get("parameters", []):
                    if param_ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: time-constraint-compliance uses undeclared parameter ref '{param_ref}'")
            
            if rule.value_constraint_compliance:
                # Check target refs
                for target_ref in rule.value_constraint_compliance.get("targets", []):
                    if target_ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: value-constraint-compliance target uses undeclared ref '{target_ref}'")
                # Check parameter refs
                for param_ref in rule.value_constraint_compliance.get("parameters", []):
                    if param_ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: value-constraint-compliance uses undeclared parameter ref '{param_ref}'")
        
        # Validate value-constraint-compliance targets
        for rule in self.abstraction_rules:
            if rule.value_constraint_compliance:
                for target_ref in rule.value_constraint_compliance.get("targets", []):
                    # Ensure target is from anchor or event (not context or parameter)
                    df_entry = derived_from_by_ref.get(target_ref)
                    if df_entry is None:
                        raise ValueError(f"{self.name}: value-constraint target ref '{target_ref}' not found in derived-from")
                    
                    # Check if target is used in anchor or event
                    tr = rule.relation_spec
                    anchor_attrs = set(tr.get("anchor", {}).get("attributes", {}).keys())
                    event_attrs = set(tr.get("event", {}).get("attributes", {}).keys())
                    target_attr_name = df_entry["name"]
                    
                    if target_attr_name not in anchor_attrs and target_attr_name not in event_attrs:
                        raise ValueError(
                            f"{self.name}: value-constraint target ref '{target_ref}' ('{target_attr_name}') "
                            f"must reference an attribute used in anchor or event"
                        )
        
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
            tr = rule.relation_spec
            
            # Validate anchor attributes
            for attr_name, attr_spec in tr.get("anchor", {}).get("attributes", {}).items():
                # Find ref from derived_from that matches this attr_name
                df = next((d for d in self.derived_from if d["name"] == attr_name), None)
                if df is None:
                    continue
                
                ref = df["ref"]
                if ref not in derived_from_by_ref:
                    continue
                
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
                            f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') has min/max constraints but attribute '{parent_attr['name']}' "
                            f"is {parent_attr['type']} (not numeric). Use only 'equal' constraints for non-numeric attributes."
                        )
                else:
                    # numeric: reject equal, allow min/max
                    if attr_spec.get("allowed_values"):
                        raise ValueError(
                            f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') has 'equal' constraint but attribute '{parent_attr['name']}' "
                            f"is numeric. Use only 'min'/'max' constraints for numeric attributes."
                        )
                    # Check range overlap with TAK's range
                    tak_min = parent_attr.get("min")
                    tak_max = parent_attr.get("max")
                    rule_min = attr_spec.get("min")
                    rule_max = attr_spec.get("max")
                    
                    if rule_min is not None and tak_max is not None and rule_min > tak_max:
                        raise ValueError(f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') min={rule_min} exceeds TAK max={tak_max}")
                    if rule_max is not None and tak_min is not None and rule_max < tak_min:
                        raise ValueError(f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') max={rule_max} below TAK min={tak_min}")
            
            # Validate event attributes (same logic)
            for attr_name, attr_spec in tr.get("event", {}).get("attributes", {}).items():
                df = next((d for d in self.derived_from if d["name"] == attr_name), None)
                if df is None:
                    continue
                
                ref = df["ref"]
                if ref not in derived_from_by_ref:
                    continue
                
                tak = repo.get(df["name"])
                if not isinstance(tak, RawConcept):
                    continue
                
                parent_attr = _get_parent_attribute(tak, df["idx"])
                if parent_attr is None:
                    continue
                
                if parent_attr["type"] != "numeric":
                    if attr_spec.get("min") is not None or attr_spec.get("max") is not None:
                        raise ValueError(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has min/max constraints but attribute '{parent_attr['name']}' "
                            f"is {parent_attr['type']} (not numeric). Use only 'equal' constraints for non-numeric attributes."
                        )
                else:
                    if attr_spec.get("allowed_values"):
                        raise ValueError(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has 'equal' constraint but attribute '{parent_attr['name']}' "
                            f"is numeric. Use only 'min'/'max' constraints for numeric attributes."
                        )
                    tak_min = parent_attr.get("min")
                    tak_max = parent_attr.get("max")
                    rule_min = attr_spec.get("min")
                    rule_max = attr_spec.get("max")
                    
                    if rule_min is not None and tak_max is not None and rule_min > tak_max:
                        raise ValueError(f"{self.name}: event attribute '{attr_name}' (ref='{ref}') min={rule_min} exceeds TAK max={tak_max}")
                    if rule_max is not None and tak_min is not None and rule_max < tak_min:
                        raise ValueError(f"{self.name}: event attribute '{attr_name}' (ref='{ref}') max={rule_max} below TAK min={tak_min}")

            # Validate context attributes (nominal only, so reject min/max)
            if rule.context_spec:
                for attr_name, attr_spec in rule.context_spec.get("attributes", {}).items():
                    df = next((d for d in self.derived_from if d["name"] == attr_name), None)
                    if df is None:
                        continue
                    
                    ref = df["ref"]
                    if ref not in derived_from_by_ref:
                        continue
                    
                    tak = repo.get(df["name"])
                    if not isinstance(tak, RawConcept):
                        continue
                    
                    parent_attr = _get_parent_attribute(tak, df["idx"])
                    if parent_attr is None:
                        continue
                    
                    # Context only supports nominal (no min/max)
                    if attr_spec.get("min") is not None or attr_spec.get("max") is not None:
                        raise ValueError(
                            f"{self.name}: context attribute '{attr_name}' (ref='{ref}') has min/max constraints. Context attributes are nominal only; "
                            f"use only 'equal' constraints."
                        )

        # Validate context blocks (only ONE attribute allowed per context block)
        for rule in self.abstraction_rules:
            if rule.context_spec and rule.context_spec.get("attributes"):
                num_context_attrs = len(rule.context_spec["attributes"])
                if num_context_attrs > 1:
                    raise ValueError(
                        f"{self.name}: Pattern context blocks can only reference ONE context TAK. "
                        f"Found {num_context_attrs} context attributes in rule. "
                        f"Reason: A single DataFrame row cannot match multiple ConceptName values simultaneously."
                        f"You can consider a few differnt contexts (OR condition) by splitting to different rules."
                        f"Or you can consider only the intersection of 2 or more contexts as context of it's own by defining this as a seperate <context> file."
                    )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LocalPattern to find and rate pattern instances.
        
        ASSUMPTION: Input df contains ONLY records relevant to this pattern (potential anchors/events/contexts/parameters).
        All records belong to SINGLE patient.
        
        Output columns:
          - PatientId
          - ConceptName (pattern name)
          - StartDateTime (anchor start)
          - EndDateTime (event end)
          - Value ("True" | "Partial" | "False")
          - TimeConstraintScore (0-1, if time-constraint compliance exists, else None)
          - ValueConstraintScore (0-1, if value-constraint compliance exists, else None)
          - AbstractionType ("local-pattern")
        """
        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))
        
        # Assumption: all records belong to same patient (if not empty)
        patient_id = df.iloc[0]["PatientId"] if not df.empty else None
        
        # If no patient data, return empty DataFrame (no PatientId to emit False for)
        if patient_id is None:
            logger.info("[%s] apply() end | no patient data", self.name)
            return pd.DataFrame(columns=[
                "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
                "Value", "TimeConstraintScore", "ValueConstraintScore", "AbstractionType"
            ])
        
        # Find all anchor-event pairs satisfying temporal relations
        instances = self._find_pattern_instances(df)
        
        if not instances:
            # No pattern found: return False with NaT times
            # If compliance functions exist, emit 0.0
            has_time_compliance = any(r.time_constraint_compliance for r in self.abstraction_rules)
            has_value_compliance = any(r.value_constraint_compliance for r in self.abstraction_rules)
            
            logger.info("[%s] apply() end | no pattern found", self.name)
            return pd.DataFrame([{
                "PatientId": patient_id,
                "ConceptName": self.name,
                "StartDateTime": pd.NaT,
                "EndDateTime": pd.NaT,
                "Value": "False",
                "TimeConstraintScore": 0.0 if has_time_compliance else None,
                "ValueConstraintScore": 0.0 if has_value_compliance else None,
                "AbstractionType": self.family
            }])
        
        # Convert to DataFrame and add pattern metadata
        out = pd.DataFrame(instances)
        out["PatientId"] = patient_id
        out["ConceptName"] = self.name
        out["AbstractionType"] = self.family
        
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
        return out[[
            "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
            "Value", "TimeConstraintScore", "ValueConstraintScore", "AbstractionType"
        ]]

    def _find_pattern_instances(self, patient_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find all anchor-event pairs satisfying temporal relations, compute compliance scores.
        
        OPTIMIZED for per-patient processing (vectorized where possible).
        
        Returns:
            List of pattern instance dicts with scores
        """
        instances = []
        
        # Track used indices for one-to-one matching
        used_anchor_ids = set()
        used_event_ids = set()
        
        # PRE-RESOLVE PARAMETERS ONCE (not per-rule)
        # For patterns with compliance functions, compute parameters upfront
        has_compliance = any(
            r.time_constraint_compliance or r.value_constraint_compliance 
            for r in self.abstraction_rules
        )
        parameter_values = {}
        if has_compliance and self.parameters:
            # Use earliest anchor time as reference (will refine per-instance later if needed)
            # For now, just use patient_df's first timestamp as proxy
            if not patient_df.empty:
                ref_time = patient_df["StartDateTime"].min()
                parameter_values = self._resolve_parameters(ref_time, patient_df)
        
        for rule in self.abstraction_rules:
            # Extract anchor/event/context candidates (OPTIMIZED)
            anchors = self._extract_candidates(patient_df, rule.relation_spec.get("anchor"))
            events = self._extract_candidates(patient_df, rule.relation_spec.get("event"))
            contexts = self._extract_candidates(patient_df, rule.context_spec) if rule.context_spec else None
            
            if anchors.empty or events.empty:
                continue  # EARLY EXIT: no candidates → skip rule
            
            # Sort by select preference
            anchor_order = self._order_indices(anchors, rule.relation_spec.get("anchor", {}))
            event_order = self._order_indices(events, rule.relation_spec.get("event", {}))
            
            # OPTIMIZATION: Pre-filter temporal matches (vectorized before nested loop)
            # This reduces O(N²) nested loop to O(N×M) where M is # of valid candidates.
            valid_pairs = self._prefilter_temporal_matches(
                anchors, events, anchor_order, event_order, rule
            )
            
            # Iterate over valid pairs (one-to-one pairing)
            for anchor_idx, event_idx in valid_pairs:
                if anchor_idx in used_anchor_ids or event_idx in used_event_ids:
                    continue  # Already used
                
                anchor_row = anchors.loc[anchor_idx]
                event_row = events.loc[event_idx]
                
                # Check full rule match (context + detailed constraints)
                if not rule.matches(anchor_row, event_row, contexts):
                    continue
                
                # Pattern found! Compute compliance scores (None if no compliance function)
                time_score = None
                value_score = None
                
                # Refine parameters if needed (use anchor start time as reference)
                if has_compliance and self.parameters:
                    # OPTIMIZATION: Only re-resolve if anchor time differs significantly from ref_time
                    # (For most patients with < 100 records, this is overkill; skip for now)
                    pass  # Use pre-computed parameter_values
                
                # Compute time-constraint compliance (only if function exists)
                if rule.time_constraint_compliance:
                    time_score = self._compute_time_compliance(
                        anchor_row=anchor_row,
                        event_row=event_row,
                        tcc_spec=rule.time_constraint_compliance,
                        parameter_values=parameter_values
                    )
                
                # Compute value-constraint compliance (only if function exists)
                if rule.value_constraint_compliance:
                    value_score = self._compute_value_compliance(
                        anchor_row=anchor_row,
                        event_row=event_row,
                        vcc_spec=rule.value_constraint_compliance,
                        parameter_values=parameter_values
                    )
                
                # Classify pattern instance based on scores
                combined_score = self._compute_combined_score(time_score, value_score)
                
                if combined_score == 1.0:
                    value_label = "True"
                elif combined_score > 0.0:
                    value_label = "Partial"
                else:
                    value_label = "False"
                
                instances.append({
                    "StartDateTime": anchor_row["StartDateTime"],
                    "EndDateTime": event_row["EndDateTime"],
                    "Value": value_label,
                    "TimeConstraintScore": time_score,
                    "ValueConstraintScore": value_score
                })
                
                used_anchor_ids.add(anchor_idx)
                used_event_ids.add(event_idx)
        
        return instances

    def _prefilter_temporal_matches(
        self,
        anchors: pd.DataFrame,
        events: pd.DataFrame,
        anchor_order: List[int],
        event_order: List[int],
        rule: TemporalRelationRule
    ) -> List[Tuple[int, int]]:
        """
        Pre-filter anchor-event pairs using vectorized temporal checks.
        Returns list of (anchor_idx, event_idx) pairs that satisfy temporal relation.
        
        OPTIMIZATION: Reduces O(N²) nested loop to O(N×M) where M is # of valid candidates.
        """
        valid_pairs = []
        
        # Extract temporal relation spec
        how = rule.relation_spec.get("how")
        max_delta = rule.max_delta
        
        # Vectorized temporal check (before="event.start must be after anchor.end")
        if how == "before":
            # For each anchor, find events that start after anchor.end (within max_delta)
            for anchor_idx in anchor_order:
                anchor_end = anchors.loc[anchor_idx, "EndDateTime"]
                
                # Vectorized mask: event.start > anchor.end AND (if max_delta) event.start - anchor.end <= max_delta
                mask = (events["StartDateTime"] > anchor_end)
                if max_delta is not None:
                    mask &= ((events["StartDateTime"] - anchor_end) <= max_delta)
                
                # Get matching event indices (in order)
                matching_event_idxs = [idx for idx in event_order if idx in events[mask].index]
                
                # Add pairs (anchor, event) to valid list
                for event_idx in matching_event_idxs:
                    valid_pairs.append((anchor_idx, event_idx))
        
        elif how == "overlap":
            # For each anchor, find events that overlap
            for anchor_idx in anchor_order:
                anchor_start = anchors.loc[anchor_idx, "StartDateTime"]
                anchor_end = anchors.loc[anchor_idx, "EndDateTime"]
                
                # Vectorized mask: overlap = NOT (event.end < anchor.start OR event.start > anchor.end)
                mask = ~(
                    (events["EndDateTime"] < anchor_start) | 
                    (events["StartDateTime"] > anchor_end)
                )
                
                # Get matching event indices (in order)
                matching_event_idxs = [idx for idx in event_order if idx in events[mask].index]
                
                # Add pairs
                for event_idx in matching_event_idxs:
                    valid_pairs.append((anchor_idx, event_idx))
        
        return valid_pairs

    def _resolve_parameters(self, pattern_start: pd.Timestamp, patient_df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        Resolve parameter values: use closest record to pattern start time.
        
        OPTIMIZED: Vectorized groupby + idxmin (no apply() loops).
        
        Args:
            pattern_start: Start time of pattern instance (anchor start)
            patient_df: Patient's full data
        
        Returns:
            Dict {param_ref: resolved_value (float, or original string if not parseable)}
        """
        resolved = {}
        
        if not self.parameters:
            return resolved
        
        # Build dict of param names → param specs for quick lookup
        param_specs_by_name = {p["name"]: p for p in self.parameters}
        
        # Vectorized: compute time distance for all rows at once
        patient_df = patient_df.copy()  # Avoid SettingWithCopyWarning
        patient_df["TimeDist"] = (patient_df["StartDateTime"] - pattern_start).abs()
        
        # Group by ConceptName and find index of row with min TimeDist
        param_names = list(param_specs_by_name.keys())
        param_df = patient_df[patient_df["ConceptName"].isin(param_names)]
        
        if param_df.empty:
            # No parameter data found → use all defaults
            for param_spec in self.parameters:
                resolved[param_spec["ref"]] = self._parse_parameter_value(param_spec["default"])
            return resolved
        
        # Vectorized: find closest row per parameter
        closest_indices = param_df.groupby("ConceptName")["TimeDist"].idxmin()
        
        for param_spec in self.parameters:
            param_ref = param_spec["ref"]
            param_name = param_spec["name"]
            param_idx = param_spec["idx"]
            param_default = param_spec["default"]
            
            # Check if we found data for this parameter
            if param_name not in closest_indices:
                resolved[param_ref] = self._parse_parameter_value(param_default)
                continue
            
            # Extract value from closest row
            closest_idx = closest_indices[param_name]
            closest_row = patient_df.loc[closest_idx]
            val = closest_row["Value"]
            
            # Extract using idx (handle tuples)
            if isinstance(val, tuple):
                val = val[param_idx] if param_idx < len(val) else None
            
            if val is not None:
                resolved[param_ref] = self._parse_parameter_value(val)
            else:
                resolved[param_ref] = self._parse_parameter_value(param_default)
        
        return resolved

    @staticmethod
    def _parse_parameter_value(val: Any) -> Union[float, str]:
        """
        Parse parameter value: try numeric → time-duration → string.
        
        Args:
            val: Raw parameter value (can be float, int, or string)
        
        Returns:
            - float: if numeric or time-duration (converted to seconds)
            - str: if neither (pass to external function as-is)
        """
        # Try numeric conversion first
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
        
        # Try time-duration parsing (e.g., "1h" → 3600.0 seconds)
        if isinstance(val, str):
            try:
                duration = parse_duration(val)
                return duration.total_seconds()  # Convert to float (seconds)
            except (ValueError, AttributeError):
                pass
        
        # Keep as string (external function will handle it)
        return str(val)

    def _compute_time_compliance(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        tcc_spec: Dict[str, Any],
        parameter_values: Dict[str, float]
    ) -> float:
        """
        Compute time-constraint compliance score.
        
        Args:
            anchor_row: Anchor row
            event_row: Event row
            tcc_spec: {func_name, trapez, parameters}
            parameter_values: Resolved parameter values
        
        Returns:
            float: Compliance score in [0, 1]
        """
        # Compute actual time gap (event.start - anchor.end)
        time_gap = event_row["StartDateTime"] - anchor_row["EndDateTime"]
        param_vals = [parameter_values[ref] for ref in tcc_spec["parameters"]]
        
        try:
            trapez_node = apply_external_function(
                tcc_spec["func_name"],  # positional
                tcc_spec["trapez"],     # positional
                "time-constraint",      # positional
                *param_vals
            )
            
            # Compute compliance score
            score = trapez_node.compliance_score(time_gap)
            return score
        except Exception as e:
            logger.error(f"[{self.name}] Time-constraint compliance error: {e}")
            return 0.0

    def _compute_value_compliance(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        vcc_spec: Dict[str, Any],
        parameter_values: Dict[str, float]
    ) -> float:
        """
        Compute value-constraint compliance score.
        
        Args:
            anchor_row: Anchor row
            event_row: Event row
            vcc_spec: {func_name, trapez, targets, parameters}
            parameter_values: Resolved parameter values
        
        Returns:
            float: Compliance score in [0, 1] (minimum of all target scores)
        """
        # Extract target values (from anchor/event rows)
        target_values = []
        for target_ref in vcc_spec["targets"]:
            target_spec = self.derived_from_map.get(target_ref)
            if not target_spec:
                continue
            
            # Check if target is anchor or event (validation ensures this)
            if target_spec["name"] == anchor_row["ConceptName"]:
                val = anchor_row["Value"]
            elif target_spec["name"] == event_row["ConceptName"]:
                val = event_row["Value"]
            else:
                # Shouldn't happen after validation
                continue
            
            # Extract value using idx (handle tuples)
            if isinstance(val, tuple):
                idx = target_spec["idx"]
                val = val[idx] if idx < len(val) else None
            
            if val is not None:
                try:
                    target_values.append(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"[{self.name}] Cannot convert target value to float: {val}")
        
        if not target_values:
            return 0.0
        
        # Extract parameter values for function (ONLY if parameters are defined)
        param_vals = [parameter_values[ref] for ref in vcc_spec["parameters"]]
        
        try:
            trapez_node = apply_external_function(
                vcc_spec["func_name"],  # positional
                vcc_spec["trapez"],     # positional
                "value-constraint",     # positional
                *param_vals
            )
            scores = [trapez_node.compliance_score(tv) for tv in target_values]
            min_score = min(scores) if scores else 0.0
            return min_score
        except Exception as e:
            logger.error(f"[{self.name}] Value-constraint compliance error: {e}")
            return 0.0

    @staticmethod
    def _compute_combined_score(time_score: Optional[float], value_score: Optional[float]) -> float:
        """
        Combine time and value compliance scores.
        
        Strategy:
        - If both exist: average (allows partial compliance on one dimension to count)
        - If only one exists: use that score (other dimension is not constrained)
        - If neither exists: return 1.0 (no compliance constraints = full compliance)
        
        Examples:
        - time=1.0, value=0.0 → avg=0.5 → "Partial" 
        - time=0.5, value=1.0 → avg=0.75 → "Partial"
        - time=1.0, value=None → 1.0 → "True"
        """
        scores = [s for s in [time_score, value_score] if s is not None]
        
        if not scores:
            return 1.0  # No compliance functions → full compliance
        return sum(scores) / len(scores)

    def _extract_candidates(self, df: pd.DataFrame, spec: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract candidate rows for anchor/event/context attributes.
        
        OPTIMIZED: Vectorized boolean masks (no apply() loops).
        Supports multiple attributes (OR semantics): any attribute that satisfies the constraints is eligible.
        """
        attrs = spec.get("attributes", {})
        if not attrs:
            return pd.DataFrame(columns=df.columns)

        masked_parts = []
        
        for attr_name, constraints in attrs.items():
            idx = constraints.get("idx", 0)
            
            # Filter to rows matching this ConceptName
            rows = df[df["ConceptName"] == attr_name].copy()
            if rows.empty:
                continue
            
            # VECTORIZED: Extract value at idx for all rows at once        
            if idx == 0:
                # Common case: idx=0 (optimize for single-attr or first element)
                rows["__value__"] = rows["Value"].apply(
                    lambda v: v[0] if isinstance(v, tuple) and len(v) > 0 else v
                )
            else:
                # General case: extract at idx
                rows["__value__"] = rows["Value"].apply(
                    lambda v: v[idx] if isinstance(v, tuple) and idx < len(v) else None
                )
            
            # VECTORIZED: Apply constraints using boolean masks
            allowed = constraints.get("allowed_values") or set()
            if allowed:
                rows = rows[rows["__value__"].astype(str).isin(allowed)]
            if constraints.get("min") is not None:
                # Convert to numeric (coerce errors to NaN), then filter
                numeric_vals = pd.to_numeric(rows["__value__"], errors="coerce")
                rows = rows[numeric_vals >= constraints["min"]]
            if constraints.get("max") is not None:
                numeric_vals = pd.to_numeric(rows["__value__"], errors="coerce")
                rows = rows[numeric_vals <= constraints["max"]]
            
            if not rows.empty:
                masked_parts.append(rows)

        if not masked_parts:
            return pd.DataFrame(columns=df.columns)
        # Combine and sort (single sort operation)
        combined = pd.concat(masked_parts, ignore_index=False).sort_values("StartDateTime")
        return combined.reset_index(drop=True)

    def _order_indices(self, df: pd.DataFrame, spec: Dict[str, Any]) -> List[int]:
        """
        Sort indices by select preference (first=ascending, last=descending).
        """
        if df.empty:
            return []
        select = (spec or {}).get("select", "first")
        if select == "last":
            return list(df.sort_values("StartDateTime", ascending=False).index)
        return list(df.sort_values("StartDateTime", ascending=True).index)

    @staticmethod
    def _extract_value(value: Any, idx: int) -> Any:
        """Extract value from tuple (raw-concept) or string representation."""
        if isinstance(value, tuple):
            return value[idx] if idx < len(value) else None
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, tuple):
                    return parsed[idx] if idx < len(parsed) else None
            except (ValueError, SyntaxError):
                pass
        return value
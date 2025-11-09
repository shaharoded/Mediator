from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd
import math
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
                    
                    # Extract parameter refs (if any)
                    param_refs = []
                    parameters_el = func_el.find("parameters")
                    if parameters_el is not None:
                        for p in parameters_el.findall("parameter"):
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
        if df.empty:
            return pd.DataFrame(columns=[
                "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
                "Value", "TimeConstraintScore", "ValueConstraintScore", "AbstractionType"
            ])
        
        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))
        
        # Assumption: all records belong to same patient
        patient_id = df.iloc[0]["PatientId"]
        
        # Find all anchor-event pairs satisfying temporal relations
        instances = self._find_pattern_instances(patient_id, df)
        
        if not instances:
            # No pattern found: return False with NaN times
            logger.info("[%s] apply() end | no pattern found", self.name)
            return pd.DataFrame([{
                "PatientId": patient_id,
                "ConceptName": self.name,
                "StartDateTime": pd.NaT,
                "EndDateTime": pd.NaT,
                "Value": "False",
                "TimeConstraintScore": None,
                "ValueConstraintScore": None,
                "AbstractionType": self.family
            }])
        
        # Convert to DataFrame
        out = pd.DataFrame(instances)
        out["ConceptName"] = self.name
        out["AbstractionType"] = self.family
        
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
        return out[[
            "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
            "Value", "TimeConstraintScore", "ValueConstraintScore", "AbstractionType"
        ]]

    def _find_pattern_instances(self, patient_id: int, patient_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find all anchor-event pairs satisfying temporal relations, compute compliance scores.
        
        Returns:
            List of pattern instance dicts with scores
        """
        instances = []
        
        # Track used indices for one-to-one matching
        used_anchor_ids = set()
        used_event_ids = set()
        
        for rule in self.abstraction_rules:
            # Extract anchor/event/context candidates
            anchors = self._extract_candidates(patient_df, rule.relation_spec.get("anchor"))
            events = self._extract_candidates(patient_df, rule.relation_spec.get("event"))
            contexts = self._extract_candidates(patient_df, rule.context_spec) if rule.context_spec else None
            
            if anchors.empty or events.empty:
                continue
            
            # Sort by select preference
            anchor_order = self._order_indices(anchors, rule.relation_spec.get("anchor", {}))
            event_order = self._order_indices(events, rule.relation_spec.get("event", {}))
            
            # Iterate over anchors (one-to-one pairing)
            for anchor_idx in anchor_order:
                if anchor_idx in used_anchor_ids:
                    continue
                anchor_row = anchors.loc[anchor_idx]
                
                for event_idx in event_order:
                    if event_idx in used_event_ids:
                        continue
                    event_row = events.loc[event_idx]
                    
                    # Check if this pair matches the rule
                    if not rule.matches(anchor_row, event_row, contexts):
                        continue
                    
                    # Pattern found! Compute compliance scores
                    time_score = None
                    value_score = None
                    
                    # Resolve parameters ONCE per instance (use closest to pattern start)
                    parameter_values = self._resolve_parameters(anchor_row["StartDateTime"], patient_df)
                    
                    # Compute time-constraint compliance
                    if rule.time_constraint_compliance:
                        time_score = self._compute_time_compliance(
                            anchor_row=anchor_row,
                            event_row=event_row,
                            tcc_spec=rule.time_constraint_compliance,
                            parameter_values=parameter_values
                        )
                    
                    # Compute value-constraint compliance
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
                        "PatientId": patient_id,
                        "StartDateTime": anchor_row["StartDateTime"],
                        "EndDateTime": event_row["EndDateTime"],
                        "Value": value_label,
                        "TimeConstraintScore": time_score,
                        "ValueConstraintScore": value_score
                    })
                    
                    used_anchor_ids.add(anchor_idx)
                    used_event_ids.add(event_idx)
                    break  # one-to-one pairing
        
        return instances

    def _resolve_parameters(self, pattern_start: pd.Timestamp, patient_df: pd.DataFrame) -> Dict[str, float]:
        """
        Resolve parameter values: use closest record to pattern start time.
        
        Args:
            pattern_start: Start time of pattern instance (anchor start)
            patient_df: Patient's full data
        
        Returns:
            Dict {param_ref: resolved_value}
        """
        resolved = {}
        
        for param_spec in self.parameters:
            param_ref = param_spec["ref"]
            param_name = param_spec["name"]
            param_idx = param_spec["idx"]
            
            # Find rows matching this parameter TAK
            param_rows = patient_df[patient_df["ConceptName"] == param_name]
            
            if param_rows.empty:
                # No data: use default
                resolved[param_ref] = float(param_spec["default"])
                continue
            
            # Find closest record to pattern_start (minimize time distance)
            param_rows = param_rows.copy()
            param_rows["TimeDist"] = (param_rows["StartDateTime"] - pattern_start).abs()
            closest_row = param_rows.loc[param_rows["TimeDist"].idxmin()]
            
            # Extract value using idx
            val = closest_row["Value"]
            if isinstance(val, tuple):
                val = val[param_idx] if param_idx < len(val) else None
            
            if val is not None:
                try:
                    resolved[param_ref] = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"[{self.name}] Cannot convert parameter {param_ref} value to float: {val}, using default")
                    resolved[param_ref] = float(param_spec["default"])
            else:
                resolved[param_ref] = float(param_spec["default"])
        
        return resolved

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
        
        # Extract parameter values for function
        param_vals = [parameter_values.get(ref, 0.0) for ref in tcc_spec["parameters"]]
        
        try:
            # Apply external function to build TrapezNode
            trapez_node = apply_external_function(
                func_name=tcc_spec["func_name"],
                trapez=tcc_spec["trapez"],
                constraint_type="time-constraint",
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
            
            # Check if target is anchor or event
            if target_spec["name"] == anchor_row["ConceptName"]:
                val = anchor_row["Value"]
            elif target_spec["name"] == event_row["ConceptName"]:
                val = event_row["Value"]
            else:
                # Target not in anchor/event (shouldn't happen after validation)
                logger.warning(f"[{self.name}] Target {target_ref} not found in anchor or event")
                continue
            
            # Extract value using idx
            if isinstance(val, tuple):
                idx = target_spec["idx"]
                val = val[idx] if idx < len(val) else None
            
            if val is not None:
                try:
                    target_values.append(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"[{self.name}] Cannot convert target value to float: {val}")
        
        if not target_values:
            logger.warning(f"[{self.name}] No target values found for value-constraint compliance")
            return 0.0
        
        # Extract parameter values for function
        param_vals = [parameter_values.get(ref, 0.0) for ref in vcc_spec["parameters"]]
        
        try:
            # Apply external function to build TrapezNode
            trapez_node = apply_external_function(
                func_name=vcc_spec["func_name"],
                trapez=vcc_spec["trapez"],
                constraint_type="value-constraint",
                *param_vals
            )
            
            # Compute compliance score for each target (take minimum = most restrictive)
            scores = [trapez_node.compliance_score(v) for v in target_values]
            return min(scores)
            
        except Exception as e:
            logger.error(f"[{self.name}] Value-constraint compliance error: {e}")
            return 0.0

    @staticmethod
    def _compute_combined_score(time_score: Optional[float], value_score: Optional[float]) -> float:
        """
        Combine time and value compliance scores.
        
        Strategy: Product (both must be satisfied).
        If only one exists, use that score.
        If neither exists, return 1.0 (full compliance).
        """
        scores = [s for s in [time_score, value_score] if s is not None]
        
        if not scores:
            return 1.0  # No compliance functions → full compliance
        
        return math.prod(scores)

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
        for attr_name, constraints in attrs.items():
            idx = constraints.get("idx", 0)
            rows = df[df["ConceptName"] == attr_name].copy()
            if rows.empty:
                continue

            # Extract value for the attribute given idx
            rows["__value__"] = rows.apply(lambda r: self._extract_value(r["Value"], idx), axis=1)

            # Apply constraints
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
                return value
        return value

    def _compute_compliance_score(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        rule: TemporalRelationRule,
        patient_df: pd.DataFrame
    ) -> float:
        """
        Compute compliance score for a pattern instance.
        
        Combines time-constraint and value-constraint compliance scores (product if both exist).
        Uses apply_external_function from utils to process trapezoid scoring.
        
        Args:
            anchor_row: Anchor row (DataFrame row)
            event_row: Event row (DataFrame row)
            rule: TemporalRelationRule with compliance functions
            patient_df: Patient's full data (for extracting target/parameter values)
        
        Returns:
            float: Compliance score in [0, 1]
        """
        
        scores = []
        
        # 1) Time-constraint compliance
        if rule.time_constraint_compliance:
            tcc = rule.time_constraint_compliance
            
            # Compute actual time gap (event.start - anchor.end)
            time_gap = event_row["StartDateTime"] - anchor_row["EndDateTime"]
            
            # Extract parameter values (if any)
            param_values = []
            for param_ref in tcc["parameters"]:
                param_spec = self.parameters_map.get(param_ref)
                if param_spec:
                    # Use default value (parameters are not in patient_df yet)
                    param_values.append(float(param_spec["default"]))
            
            # Apply external function to get TrapezNode
            try:
                trapez_node = apply_external_function(
                    func_name=tcc["func_name"],
                    trapez=tcc["trapez"],
                    constraint_type="time-constraint",
                    *param_values
                )
                
                # Compute compliance score using TrapezNode
                time_score = trapez_node.compliance_score(time_gap)
                scores.append(time_score)
                
            except Exception as e:
                logger.error(f"[{self.name}] Time-constraint compliance error: {e}")
                scores.append(0.0)
        
        # 2) Value-constraint compliance
        if rule.value_constraint_compliance:
            vcc = rule.value_constraint_compliance
            
            # Extract target values (from anchor/event rows)
            target_values = []
            for target_ref in vcc["targets"]:
                target_spec = self.derived_from_map.get(target_ref)
                if target_spec:
                    # Check if target is anchor or event
                    if target_spec["name"] == anchor_row["ConceptName"]:
                        val = anchor_row["Value"]
                    elif target_spec["name"] == event_row["ConceptName"]:
                        val = event_row["Value"]
                    else:
                        continue
                    
                    # Extract value using idx
                    if isinstance(val, tuple):
                        idx = target_spec["idx"]
                        val = val[idx] if idx < len(val) else None
                    
                    if val is not None:
                        try:
                            target_values.append(float(val))
                        except (ValueError, TypeError):
                            logger.warning(f"[{self.name}] Cannot convert target value to float: {val}")
            
            # Extract parameter values
            param_values = []
            for param_ref in vcc["parameters"]:
                param_spec = self.parameters_map.get(param_ref)
                if param_spec:
                    param_values.append(float(param_spec["default"]))
            
            # Apply external function to get TrapezNode
            try:
                trapez_node = apply_external_function(
                    func_name=vcc["func_name"],
                    trapez=vcc["trapez"],
                    constraint_type="value-constraint",
                    *param_values
                )
                
                # Compute compliance score for each target value (aggregate as minimum)
                if target_values:
                    value_scores = [trapez_node.compliance_score(v) for v in target_values]
                    value_score = min(value_scores)  # Most restrictive target
                    scores.append(value_score)
                else:
                    logger.warning(f"[{self.name}] No target values found for value-constraint compliance")
                    scores.append(0.0)
                
            except Exception as e:
                logger.error(f"[{self.name}] Value-constraint compliance error: {e}")
                scores.append(0.0)
        
        # Combine scores (product: both must be satisfied)
        if scores:
            import math
            return math.prod(scores)
        else:
            return 1.0  # No compliance functions → full compliance
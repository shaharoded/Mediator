from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, validate_xml_against_schema, TemporalRelationRule, CyclicRule
from .utils import apply_external_function_on_trapez, parse_duration, FuzzyLogicTrapez
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
        self.parameters = []

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Pattern":
        xml_path = Path(xml_path)
        validate_xml_against_schema(xml_path)
        root = ET.parse(xml_path).getroot()
        concept_type = root.attrib.get("concept-type")
        
        if concept_type == "local-pattern":
            return LocalPattern.parse(xml_path)
        elif concept_type == "global-pattern":
            return GlobalPattern.parse(xml_path)
        else:
            raise ValueError(f"Unknown concept-type '{concept_type}' in {xml_path}")

    def validate(self) -> None:
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

    def _resolve_parameters(self, pattern_start: pd.Timestamp, patient_df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        resolved = {}
        if not self.parameters: return resolved
        param_specs_by_name = {p["name"]: p for p in self.parameters}
        patient_df = patient_df.copy()
        patient_df["TimeDist"] = (patient_df["StartDateTime"] - pattern_start).abs()
        param_names = list(param_specs_by_name.keys())
        param_df = patient_df[patient_df["ConceptName"].isin(param_names)]
        if param_df.empty:
            for param_spec in self.parameters:
                resolved[param_spec["ref"]] = self._parse_parameter_value(param_spec["default"])
            return resolved
        closest_indices = param_df.groupby("ConceptName")["TimeDist"].idxmin()
        for param_spec in self.parameters:
            param_ref = param_spec["ref"]
            param_name = param_spec["name"]
            param_idx = param_spec["idx"]
            param_default = param_spec["default"]
            if param_name not in closest_indices:
                resolved[param_ref] = self._parse_parameter_value(param_default)
                continue
            closest_idx = closest_indices[param_name]
            closest_row = patient_df.loc[closest_idx]
            val = closest_row["Value"]
            if isinstance(val, tuple):
                val = val[param_idx] if param_idx < len(val) else None
            if val is not None:
                resolved[param_ref] = self._parse_parameter_value(val)
            else:
                resolved[param_ref] = self._parse_parameter_value(param_default)
        return resolved

    @staticmethod
    def _parse_parameter_value(val: Any) -> Union[float, str]:
        try: return float(val)
        except (ValueError, TypeError): pass
        if isinstance(val, str):
            try: return parse_duration(val).total_seconds()
            except (ValueError, AttributeError): pass
        return str(val)

    def _extract_candidates(self, df: pd.DataFrame, spec: Optional[Dict[str, Any]]) -> pd.DataFrame:
        attrs = spec.get("attributes", {})
        if not attrs: return pd.DataFrame(columns=df.columns)
        masked_parts = []
        for attr_name, constraints in attrs.items():
            idx = constraints.get("idx", 0)
            rows = df[df["ConceptName"] == attr_name].copy()
            if rows.empty: continue
            if idx == 0:
                rows["__value__"] = rows["Value"].apply(lambda v: v[0] if isinstance(v, tuple) and len(v) > 0 else v)
            else:
                rows["__value__"] = rows["Value"].apply(lambda v: v[idx] if isinstance(v, tuple) and idx < len(v) else None)
            allowed = constraints.get("allowed_values") or set()
            if allowed: rows = rows[rows["__value__"].astype(str).isin(allowed)]
            if constraints.get("min") is not None:
                numeric_vals = pd.to_numeric(rows["__value__"], errors="coerce")
                rows = rows[numeric_vals >= constraints["min"]]
            if constraints.get("max") is not None:
                numeric_vals = pd.to_numeric(rows["__value__"], errors="coerce")
                rows = rows[numeric_vals <= constraints["max"]]
            if not rows.empty: masked_parts.append(rows)
        if not masked_parts: return pd.DataFrame(columns=df.columns)
        return pd.concat(masked_parts, ignore_index=False).sort_values("StartDateTime")
    
    @staticmethod
    def _extract_value(value: Any, idx: Optional[int]) -> Any:
        """
        Extract value from tuple (raw-concept) or string representation.
        Used for value compliance checks (numeric values).
        """
        if isinstance(value, tuple):
            value = value[idx] if idx < len(value) else None
                    
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            try:
                import ast
                value = ast.literal_eval(value)
                if isinstance(value, tuple):
                    if idx < len(value):
                        value = value[idx]
                    else:
                        value = None
            except (ValueError, SyntaxError):
                pass
        try:
            value = float(value)
        except (ValueError, TypeError):
            pass
        return value
    
    @staticmethod
    def _compute_combined_score(scores: Optional[List[Optional[float]]]) -> float:
        """
        Combine compliance scores.
        
        Strategy:
        - If len(scores) > 1: average (allows partial compliance on one dimension to count)
        - If only one exists: use that score (other dimension is not constrained)
        - If neither exists: return 1.0 (no compliance constraints = full compliance)
        
        Examples:
        - time=1.0, value=0.0 → avg=0.5 → "Partial" 
        - time=0.5, value=1.0 → avg=0.75 → "Partial"
        - time=1.0, value=None → 1.0 → "True"
        """
        scores = [s for s in (scores or []) if s is not None]
        
        if not scores:
            return 1.0  # No compliance functions → full compliance
        return sum(scores) / len(scores)


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
        abstraction_rules: List[TemporalRelationRule],
        ignore_unfulfilled_anchors: bool = False
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
        self.ignore_unfulfilled_anchors = ignore_unfulfilled_anchors

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

        # --- parse ignore-unfulfilled-anchors attribute ---
        ignore_unfulfilled_anchors = root.attrib.get("ignore-unfulfilled-anchors", "false").lower() == "true"

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
            min_distance = tr_el.attrib.get("min-distance")
            existence_compliance = (tr_el.attrib.get("existence-compliance") == "true") if "existence-compliance" in tr_el.attrib else None
            if existence_compliance and ignore_unfulfilled_anchors:
                raise ValueError(f"{name}: cannot set both existence-compliance='true' and ignore-unfulfilled-anchors=true, they conflict. One says 'give score 0.0 if event missing from anchor', the other says 'ignore unfulfilled anchors'")

            # Ensure max-distance and min-distance are not None
            if how == "before":
                if not max_distance:
                    raise ValueError(f"{name}: temporal-relation how='before' requires max-distance")
                else:
                    # Validate format
                    d = parse_duration(max_distance)
                    if d == 0:
                        raise ValueError(f"{name}: temporal-relation max-distance must be > 0, otherwise use 'overlap' relation")
                if not min_distance:
                    min_distance = "0s"  # Default to 0 seconds if not provided
                if existence_compliance:
                    raise ValueError(f"{name}: existence-compliance not supported for 'before' temporal relation, only for 'overlap'")
            else:  # overlap
                if max_distance or min_distance:
                    raise ValueError(f"{name}: temporal-relation how='overlap' cannot have max-distance or min-distance")
                if existence_compliance is None:
                    raise ValueError(f"{name}: temporal-relation how='overlap' requires existence-compliance attribute")
            relation_spec = {
                "how": how,
                "max_distance": max_distance,
                "min_distance": min_distance,
                "existence_compliance": existence_compliance
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
                if cf_el.find("cyclic-constraint-compliance") is not None:
                    raise ValueError(f"{name}: LocalPattern cannot have cyclic-constraint-compliance; only time and value compliance allowed.")
        
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
                    
                    # Parse trapez (time-based, stored as strings)
                    trapez_el = func_el.find("trapez")
                    if trapez_el is None:
                        raise ValueError(f"{name}: time-constraint-compliance <function> missing <trapez>")
                    
                    trapez_raw = (
                        trapez_el.attrib.get("trapezA"),
                        trapez_el.attrib.get("trapezB"),
                        trapez_el.attrib.get("trapezC"),
                        trapez_el.attrib.get("trapezD")
                    )
                    
                    if None in trapez_raw:
                        raise ValueError(f"{name}: time-constraint-compliance trapez must have all 4 attributes (A, B, C, D)")
                    
                    # Validate *_distance against trapez nodes (pattern must capture all valid instances)
                    # Still unparsed for logs clearity
                    if parse_duration(trapez_raw[3]) > parse_duration(max_distance):
                        raise ValueError(
                            f"{name}: temporal-relation max-distance '{max_distance}' must be >= "
                            f"time-constraint-compliance trapezD '{trapez_raw[3]}' "
                            f"(otherwise pattern may miss valid instances)"
                        )
                    if parse_duration(trapez_raw[0]) < parse_duration(min_distance):
                        raise ValueError(
                            f"{name}: temporal-relation min-distance '{min_distance}' must be <= "
                            f"time-constraint-compliance trapezA '{trapez_raw[0]}' "
                            f"(otherwise pattern may miss valid instances)"
                        )
                    if parse_duration(trapez_raw[2]) < parse_duration(trapez_raw[3]) and event_spec.get("select") == "last":
                        logger.warning(
                            f"{name}: time-constraint-compliance trapezC '{trapez_raw[2]}' < trapezD '{trapez_raw[3]}' "
                            f"may lead to unexpected scoring when event select='last'"
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
                        "trapez": FuzzyLogicTrapez(*trapez_raw, is_time=True),  # Build as time-based trapez
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
                    
                    # Parse trapez (numeric values)
                    trapez_el = func_el.find("trapez")
                    if trapez_el is None:
                        raise ValueError(f"{name}: value-constraint-compliance <function> missing <trapez>")
                    
                    try:
                        trapez_raw = (
                            float(trapez_el.attrib.get("trapezA")),
                            float(trapez_el.attrib.get("trapezB")),
                            float(trapez_el.attrib.get("trapezC")),
                            float(trapez_el.attrib.get("trapezD"))
                        )
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"{name}: value-constraint-compliance trapez values must be numeric: {e}")
                    
                    # Extract parameter refs (if any)
                    param_refs = []
                    for p in func_el.findall("parameter"):
                        ref = p.attrib.get("ref")
                        if not ref or ref not in declared_refs:
                            raise ValueError(f"{name}: parameter ref '{ref}' not declared in derived-from or parameters")
                        param_refs.append(ref)
                    
                    value_constraint_compliance = {
                        "func_name": func_name,
                        "trapez": FuzzyLogicTrapez(*trapez_raw, is_time=False),  # Build as value-based trapez
                        "targets": target_refs,
                        "parameters": param_refs
                    }
            
            # Create TemporalRelationRule with compliance functions            
            abstraction_rules.append(TemporalRelationRule(
                derived_map=derived_map,
                relation_spec=relation_spec,
                context_spec=context_spec,
                time_constraint_compliance=time_constraint_compliance,
                value_constraint_compliance=value_constraint_compliance
            ))

        pattern = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            parameters=parameters,
            abstraction_rules=abstraction_rules,
            ignore_unfulfilled_anchors=ignore_unfulfilled_anchors
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
                    
                    # Only warn if pattern EXPLICITLY sets bounds that exceed parent bounds
                    # (None means open range, which implicitly inherits parent bounds)
                    if rule_min is not None and tak_max is not None and rule_min > tak_max:
                        raise ValueError(f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') min={rule_min} exceeds TAK max={tak_max}")
                    if rule_max is not None and tak_min is not None and rule_max < tak_min:
                        raise ValueError(f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') max={rule_max} below TAK min={tak_min}")
                    
                    # Info: warn if pattern constraints are tighter than necessary (outside TAK's range)
                    # This catches cases like: TAK has [0, 100], pattern says min=110 (nonsensical)
                    if rule_min is not None and tak_min is not None and rule_min < tak_min:
                        logger.info(
                            f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') has min={rule_min}, "
                            f"but parent TAK min={tak_min}. Effective range will be [{tak_min}, ...)"
                        )
                    if rule_max is not None and tak_max is not None and rule_max > tak_max:
                        logger.info(
                            f"{self.name}: anchor attribute '{attr_name}' (ref='{ref}') has max={rule_max}, "
                            f"but parent TAK max={tak_max}. Effective range will be [..., {tak_max}]"
                        )
            
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
                    
                    if rule_min is not None and tak_min is not None and rule_min < tak_min:
                        logger.info(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has min={rule_min}, "
                            f"but parent TAK min={tak_min}. Effective range will be [{tak_min}, ...)"
                        )
                    if rule_max is not None and tak_max is not None and rule_max > tak_max:
                        logger.info(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has max={rule_max}, "
                            f"but parent TAK max={tak_max}. Effective range will be [..., {tak_max}]"
                        )
            
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
          - CyclicConstraintScore (always None for LocalPattern)
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
                "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"
            ])
        
        # Find all anchor-event pairs satisfying temporal relations
        instances = self._find_pattern_instances(df)
        
        if not instances:
            # Didn't find any pattern instances, no anchors at all.
            if self.ignore_unfulfilled_anchors:
                logger.info("[%s] apply() end | no pattern found, ignore_unfulfilled_anchors=True", self.name)
                return pd.DataFrame(columns=[
                    "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
                    "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"
                ])
            # No pattern found: return False with NaT times
            # If compliance functions exist, emit missing_score, which is usually 0.0 (unless backward trapez)
            has_existence = any(
                (r.relation_spec.get("how") == "overlap") and r.relation_spec.get("existence_compliance")
                for r in self.abstraction_rules
            ) # Pattern output score 0.0 if no anchors/events found
            has_time_compliance = any(r.time_constraint_compliance for r in self.abstraction_rules) or has_existence
            time_missing_score = next((r.time_constraint_compliance['trapez'].missing_score for r in self.abstraction_rules if r.time_constraint_compliance), None)
            time_missing_score = 0.0 if has_existence else time_missing_score  # Existence compliance missing_score is 0.0
            has_value_compliance = any(r.value_constraint_compliance for r in self.abstraction_rules)
            value_missing_score = next((r.value_constraint_compliance['trapez'].missing_score for r in self.abstraction_rules if r.value_constraint_compliance), None)
            
            logger.info("[%s] apply() end | no pattern found", self.name)
            return pd.DataFrame([{
                "PatientId": patient_id,
                "ConceptName": self.name,
                "StartDateTime": pd.NaT,
                "EndDateTime": pd.NaT,
                "Value": str(bool((time_missing_score or value_missing_score))), # Both 0.0 or None → "False", else "True"
                "TimeConstraintScore": time_missing_score if has_time_compliance else None,
                "ValueConstraintScore": value_missing_score if has_value_compliance else None,
                "CyclicConstraintScore": None,
                "AbstractionType": self.family
            }])
        
        # Convert to DataFrame and add pattern metadata
        out = pd.DataFrame(instances)
        out["PatientId"] = patient_id
        out["ConceptName"] = self.name
        out["AbstractionType"] = self.family
        
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
        
        # Add CyclicConstraintScore as NaN for consistency
        out["CyclicConstraintScore"] = None
        
        return out[[
            "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
            "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"
        ]]

    def _find_pattern_instances(self, patient_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find all anchor-event pairs satisfying temporal relations, compute compliance scores.
        
        OPTIMIZED for per-patient processing (vectorized where possible).
        
        Returns:
            List of pattern instance dicts with scores
        """
        # Ensure stable indices for global tracking across rules
        patient_df = patient_df.reset_index(drop=True)
        instances = []
        
        # Track successful intervals to filter "missed" anchors later
        # List of (StartDateTime, EndDateTime)
        successful_intervals = []
        
        # Track potential "False" instances (unmatched anchors)
        potential_falses = []

        # Track used indices (globally across all rules)
        used_anchor_ids = set()
        used_event_ids = set()
        
        # PRE-RESOLVE PARAMETERS ONCE (not per-rule)
        # For patterns with compliance functions, compute parameters upfront
        has_existence = any(
            (r.relation_spec.get("how") == "overlap") and r.relation_spec.get("existence_compliance")
            for r in self.abstraction_rules
        ) # Pattern output score 0.0 if no anchors/events found
        has_time_compliance = any(r.time_constraint_compliance for r in self.abstraction_rules) or has_existence
        time_missing_score = next((r.time_constraint_compliance['trapez'].missing_score for r in self.abstraction_rules if r.time_constraint_compliance), None)
        time_missing_score = 0.0 if has_existence else time_missing_score  # Existence compliance missing_score is 0.0
        has_value_compliance = any(r.value_constraint_compliance for r in self.abstraction_rules)
        value_missing_score = next((r.value_constraint_compliance['trapez'].missing_score for r in self.abstraction_rules if r.value_constraint_compliance), None)
        has_compliance = any([has_time_compliance, has_value_compliance])
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
            
            if anchors.empty:
                continue  # EARLY EXIT: no candidates → skip rule (no events should not exit!)
            
            if not events.empty:
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
                    # Existence compliance for overlap acts as a binary time score: found=1.0
                    if rule.relation_spec.get("how") == "overlap" and rule.relation_spec.get("existence_compliance"):
                        time_score = 1.0
                    elif rule.time_constraint_compliance:
                        time_score = self._compute_time_compliance(
                            anchor_row=anchor_row,
                            event_row=event_row,
                            rule=rule,
                            parameter_values=parameter_values
                        )
                    
                    # Compute value-constraint compliance (only if function exists)
                    if rule.value_constraint_compliance:
                        value_score = self._compute_value_compliance(
                            anchor_row=anchor_row,
                            event_row=event_row,
                            rule=rule,
                            parameter_values=parameter_values
                        )
                    
                    # Classify pattern instance based on scores
                    combined_score = self._compute_combined_score([time_score, value_score])
                    
                    value_label = "True" if combined_score == 1.0 else ("Partial" if combined_score > 0 else "False")

                    
                    instances.append({
                        "StartDateTime": anchor_row["StartDateTime"],
                        "EndDateTime": event_row["EndDateTime"],
                        "Value": value_label,
                        "TimeConstraintScore": time_score,
                        "ValueConstraintScore": value_score
                    })
                    
                    successful_intervals.append((anchor_row["StartDateTime"], event_row["EndDateTime"]))
                    used_anchor_ids.add(anchor_idx)
                    used_event_ids.add(event_idx)
                
            # --- Handle Unmatched Anchors (Potential False) ---
            unused_anchor_idxs = set(anchors.index) - used_anchor_ids
            if not unused_anchor_idxs:
                continue

            # Prepare for Context Check (Condition 1)
            sorted_candidate_times = None
            if rule.context_spec:
                all_starts = []
                if not anchors.empty:
                    all_starts.append(anchors["StartDateTime"])
                if not events.empty:
                    all_starts.append(events["StartDateTime"])
                
                if all_starts:
                    combined = pd.concat(all_starts, ignore_index=True).dropna()
                    sorted_candidate_times = pd.Series(combined.sort_values().unique())
                else:
                    sorted_candidate_times = pd.Series([], dtype='datetime64[ns]')

            for idx in unused_anchor_idxs:
                anchor_row = anchors.loc[idx]
                start_dt = anchor_row["StartDateTime"]
                end_dt = anchor_row["EndDateTime"]
                
                # Condition 1: Context Satisfaction
                # "satisfied between the instance of this anchor and the next instance of an anchor or event"
                if rule.context_spec:
                    if sorted_candidate_times is None or len(sorted_candidate_times) == 0:
                        continue # Should not happen if anchors exist

                    # Find next instance time
                    # searchsorted returns index where start_dt would be inserted to maintain order
                    # side='right' gives index of first element > start_dt
                    pos = sorted_candidate_times.searchsorted(start_dt, side='right')
                    
                    if pos < len(sorted_candidate_times):
                        next_dt = sorted_candidate_times[pos]
                        
                        # Check overlap with [start_dt, next_dt]
                        # Context must overlap this interval
                        if contexts is None or contexts.empty:
                            continue # Context required but none found

                        # Overlap logic: max(start1, start2) < min(end1, end2)
                        # Here: max(ctx.Start, start_dt) < min(ctx.End, next_dt)
                        # Simplified: ctx.End > start_dt AND ctx.Start < next_dt
                        ctx_overlap = contexts[
                            (contexts["EndDateTime"] > start_dt) & 
                            (contexts["StartDateTime"] < next_dt)
                        ]
                        
                        if ctx_overlap.empty:
                            continue # Context not satisfied in the gap
                    else:
                        # No next instance found (this is the last anchor/event)
                        # If no next instance, we cannot verify "between this and next".
                        # We skip adding it as False.
                        continue

                # Only add to potential falses if not ignoring unfulfilled anchors
                # The compliance score of a missed anchor is 0.0 if compliance functions exist
                if not self.ignore_unfulfilled_anchors:
                    potential_falses.append({
                        "StartDateTime": start_dt,
                        "EndDateTime": end_dt,
                        "Value": str(bool((time_missing_score or value_missing_score))), # Both 0.0 or None → "False", else "True",
                        "TimeConstraintScore": time_missing_score if has_time_compliance else None,
                        "ValueConstraintScore": value_missing_score if has_value_compliance else None
                    })

        # --- Condition 2: Filter Potential Falses based on Successful Intervals ---
        # "another anchor that came before the anchor at time T, is not satisfied by an event that came after time T, 
        # meaning the anchor we want to add as "unfullfilled" is not captured in the middle of another instance's interval."
        
        for pf in potential_falses:
            pf_start = pf["StartDateTime"]
            is_covered = False
            
            for (succ_start, succ_end) in successful_intervals:
                # Check if pf_start is strictly inside a successful interval
                # (or start matches but it's covered by the duration)
                # Using succ_start <= pf_start < succ_end covers most cases
                if succ_start <= pf_start < succ_end:
                    is_covered = True
                    break
            
            if not is_covered:
                instances.append(pf)
        
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
        min_delta = rule.min_delta
        
        # Vectorized temporal check (before="event.start must be after anchor.end")
        if how == "before":            
            # For each anchor, find events that start after anchor.end (within max_delta)
            for anchor_idx in anchor_order:
                anchor_end = anchors.loc[anchor_idx, "EndDateTime"]
                
                # Vectorized mask: event.start > anchor.end AND (if max_delta) event.start - anchor.end <= max_delta
                mask = (events["StartDateTime"] > anchor_end)
                if min_delta is not None:
                    mask &= ((events["StartDateTime"] - anchor_end) >= min_delta)
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

    def _compute_time_compliance(
        self,
        anchor_row: pd.Series,
        event_row: pd.Series,
        rule: TemporalRelationRule,
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
        tcc_spec = rule.time_constraint_compliance
        time_gap = event_row["StartDateTime"] - anchor_row["EndDateTime"]
        param_vals = [parameter_values[ref] for ref in tcc_spec["parameters"]]
        
        try:
            trapez_node = apply_external_function_on_trapez(
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
        rule: TemporalRelationRule,
        parameter_values: Dict[str, float]
    ) -> float:
        """
        Compute value-constraint compliance score.
        
        Args:
            anchor_row: Anchor row
            event_row: Event row
            rule: TemporalRelationRule
            parameter_values: Resolved parameter values
        
        Returns:
            float: Compliance score in [0, 1] (minimum of all target scores)
        """
        # Extract target values (from anchor/event rows)
        vcc_spec = rule.value_constraint_compliance
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
            idx = target_spec["idx"] if isinstance(val, tuple) else None
            val = self._extract_value(value=val, idx=idx)
            
            if val is not None:
                try:
                    target_values.append(float(val))
                except (ValueError, TypeError):
                    logger.error(f"[{self.name}] Cannot convert target value to float: {val}. Value-constraint compliance requires numeric values.")
        
        if not target_values:
            return 0.0
        
        # Extract parameter values for function (ONLY if parameters are defined)
        param_vals = [parameter_values[ref] for ref in vcc_spec["parameters"]]
        
        try:
            trapez_node = apply_external_function_on_trapez(
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


class GlobalPattern(Pattern):
    """
    GlobalPattern: cyclic-based pattern abstraction.
    Checks if events occur min/max times within time windows.
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        abstraction_rules: List[CyclicRule]
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            derived_from=derived_from,
            family="global-pattern"
        )
        self.parameters = parameters
        self.abstraction_rules = abstraction_rules
        
        self.derived_from_map: Dict[str, Dict[str, Any]] = {
            df["ref"]: df for df in derived_from
        }

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "GlobalPattern":
        def _parse_filter_block(block_el, derived_from):
            if block_el is None:
                return None
            spec = {"attributes": {}}
            for attr_el in block_el.findall("attribute"):
                ref = attr_el.attrib["ref"]
                df_entry = next((d for d in derived_from if d["ref"] == ref), None)
                attr_name = df_entry["name"]

                allowed_values = set()
                min_val = None
                max_val = None
                for av in attr_el.findall("allowed-value"):
                    if "equal" in av.attrib: allowed_values.add(av.attrib["equal"])
                    if "min" in av.attrib: min_val = float(av.attrib["min"])
                    if "max" in av.attrib: max_val = float(av.attrib["max"])

                spec["attributes"][attr_name] = {
                    "idx": df_entry["idx"],
                    "allowed_values": allowed_values,
                    "min": min_val,
                    "max": max_val
                }
            return spec
        xml_path = Path(xml_path)
        root = ET.parse(xml_path).getroot()
        
        # --- validation: ignore-unfulfilled-anchors must not be true for global-pattern ---
        ignore_unfulfilled_anchors = root.attrib.get("ignore-unfulfilled-anchors", "false").lower() == "true"
        if ignore_unfulfilled_anchors:
            raise ValueError(f"GlobalPattern '{root.attrib.get('name')}' cannot set ignore-unfulfilled-anchors='true'.")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""
        
        # Parse derived-from
        df_el = root.find("derived-from")
        derived_from = []
        declared_refs = set()
        if df_el is not None:
            for attr_el in df_el.findall("attribute"):
                ref = attr_el.attrib["ref"]
                declared_refs.add(ref)
                tak_type = attr_el.attrib["tak"]
                idx = int(attr_el.attrib.get("idx", 0))
                derived_from.append({
                    "ref": ref, "name": attr_el.attrib["name"], "tak": tak_type, "idx": idx
                })

        # Parse parameters
        parameters = []
        params_el = root.find("parameters")
        if params_el is not None:
            for param_el in params_el.findall("parameter"):
                ref = param_el.attrib["ref"]
                declared_refs.add(ref)
                parameters.append({
                    "ref": ref, "name": param_el.attrib["name"], 
                    "tak": param_el.attrib["tak"], "idx": int(param_el.attrib.get("idx", 0)),
                    "default": param_el.attrib.get("default")
                })

        # Parse abstraction-rules (CyclicRule)
        abs_el = root.find("abstraction-rules")
        abstraction_rules = []
        
        for rule_el in abs_el.findall("rule"):
            if rule_el.find("temporal-relation") is not None:
                raise ValueError(f"{name}: GlobalPattern rule cannot have <temporal-relation>; only <cyclic> allowed.")
            
            # Look for context block in rule
            context_spec = None
            context_el = rule_el.find("context")
            if context_el is not None:
                context_spec = _parse_filter_block(context_el, derived_from)
            
            cyclic_el = rule_el.find("cyclic")
            if cyclic_el is None:
                raise ValueError(f"{name}: GlobalPattern rule missing <cyclic>")
            if any(attr not in cyclic_el.attrib for attr in ["start", "end", "time-window", "min-occurrences", "max-occurrences"]):
                raise ValueError(f"{name}: GlobalPattern cyclic rule missing required attributes. Required: start, end, time-window, min-occurrences, max-occurrences")
            
            start = cyclic_el.attrib["start"]
            end = cyclic_el.attrib["end"]
            time_window = cyclic_el.attrib["time-window"]
            min_occ = int(cyclic_el.attrib["min-occurrences"])
            max_occ = int(cyclic_el.attrib["max-occurrences"])

            # Parse initiator/ event/ clipper / context blocks
            initiator_spec = _parse_filter_block(cyclic_el.find("initiator"), derived_from)
            event_spec = _parse_filter_block(cyclic_el.find("event"), derived_from)
            clipper_spec   = _parse_filter_block(cyclic_el.find("clipper"), derived_from)
            
            # Check anchor refs
            if cyclic_el.find("anchor") is not None:
               raise ValueError(f"{name}: GlobalPattern rules cannot have anchor specifications; only initiator, event, clipper and context are allowed.") 

            # Parse compliance
            cyclic_compliance = None
            value_compliance = None
            cf_el = rule_el.find("compliance-function")
            if cf_el is not None:
                if cf_el.find("time-constraint-compliance") is not None:
                    raise ValueError(f"{name}: GlobalPattern cannot have time-constraint-compliance; only cyclic and value compliance allowed.")
                
                # Cyclic compliance
                ccc_el = cf_el.find("cyclic-constraint-compliance")
                if ccc_el is not None:
                    func_el = ccc_el.find("function")
                    trapez_el = func_el.find("trapez")
                    trapez_raw = (
                        float(trapez_el.attrib["trapezA"]), float(trapez_el.attrib["trapezB"]),
                        float(trapez_el.attrib["trapezC"]), float(trapez_el.attrib["trapezD"])
                    )
                    param_refs = [p.attrib["ref"] for p in func_el.findall("parameter")]
                    cyclic_compliance = {
                        "func_name": func_el.attrib["name"], 
                        "trapez": FuzzyLogicTrapez(*trapez_raw, is_time=False), 
                        "parameters": param_refs
                    }
                
                # Value compliance
                vcc_el = cf_el.find("value-constraint-compliance")
                if vcc_el is not None:
                    func_el = vcc_el.find("function")
                    trapez_el = func_el.find("trapez")
                    trapez_raw = (
                        float(trapez_el.attrib["trapezA"]), float(trapez_el.attrib["trapezB"]),
                        float(trapez_el.attrib["trapezC"]), float(trapez_el.attrib["trapezD"])
                    )
                    target_refs = [a.attrib["ref"] for a in vcc_el.find("target").findall("attribute")]
                    param_refs = [p.attrib["ref"] for p in func_el.findall("parameter")]
                    value_compliance = {
                        "func_name": func_el.attrib["name"], 
                        "trapez": FuzzyLogicTrapez(*trapez_raw, is_time=False), 
                        "targets": target_refs, 
                        "parameters": param_refs
                    }

            abstraction_rules.append(CyclicRule(
                start=start,
                end=end,
                time_window=time_window,
                min_occurrences=min_occ,
                max_occurrences=max_occ,
                initiator_spec=initiator_spec,
                event_spec=event_spec,
                clipper_spec=clipper_spec,
                context_spec=context_spec,
                cyclic_constraint_compliance=cyclic_compliance,
                value_constraint_compliance=value_compliance,
            ))

        return cls(name, cats, desc, derived_from, parameters, abstraction_rules)

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
            
            # Check initiator/ event /clipper / context refs
            for spec_name, spec in [("initiator", rule.initiator_spec), ("event", rule.event_spec), ("clipper", rule.clipper_spec), ("context", rule.context_spec)]:
                if not spec:
                    continue
                for attr_name in spec.get("attributes", {}):
                    df_entry = next((d for d in self.derived_from if d["name"] == attr_name), None)
                    if df_entry is None:
                        raise ValueError(f"{self.name}: {spec_name} references unknown attribute '{attr_name}'")
                    if df_entry["ref"] not in all_declared_refs:
                        raise ValueError(f"{self.name}: {spec_name} uses undeclared ref '{df_entry['ref']}'")
            
            # Check compliance function parameter refs
            if rule.cyclic_constraint_compliance:
                for param_ref in rule.cyclic_constraint_compliance.get("parameters", []):
                    if param_ref not in all_declared_refs:
                        raise ValueError(f"{self.name}: cyclic-constraint-compliance uses undeclared parameter ref '{param_ref}'")
            
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
                    # Ensure target is from event (not context or parameter)
                    df_entry = derived_from_by_ref.get(target_ref)
                    if df_entry is None:
                        raise ValueError(f"{self.name}: value-constraint target ref '{target_ref}' not found in derived-from")
                    
                    # Check if target is used in event
                    event_attrs = set(rule.event_spec.get("attributes", {}).keys())
                    target_attr_name = df_entry["name"]
                    
                    if target_attr_name not in event_attrs:
                        raise ValueError(
                            f"{self.name}: value-constraint target ref '{target_ref}' ('{target_attr_name}') "
                            f"must reference an attribute used in event"
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
            # Validate event attributes
            for attr_name, attr_spec in rule.event_spec.get("attributes", {}).items():
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
                    
                    if rule_min is not None and tak_min is not None and rule_min < tak_min:
                        logger.info(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has min={rule_min}, "
                            f"but parent TAK min={tak_min}. Effective range will be [{tak_min}, ...)"
                        )
                    if rule_max is not None and tak_max is not None and rule_max > tak_max:
                        logger.info(
                            f"{self.name}: event attribute '{attr_name}' (ref='{ref}') has max={rule_max}, "
                            f"but parent TAK max={tak_max}. Effective range will be [..., {tak_max}]"
                        )
            
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
        Apply GlobalPattern to find and rate pattern instances.

        ASSUMPTION: Input df contains ONLY records relevant to this pattern (potential events/contexts/parameters).
        All records belong to SINGLE patient.

        Output columns:
          - PatientId
          - ConceptName (pattern name)
          - StartDateTime (window start)
          - EndDateTime (window end)
          - Value ("True" | "Partial" | "False")
          - TimeConstraintScore (None, place holder. Only used in LocalPattern)
          - ValueConstraintScore (0-1, if value-constraint compliance exists, else None)
          - CyclicConstraintScore (0-1, if cyclic-constraint compliance exists, else None)
          - AbstractionType ("global-pattern")
        """
        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))
        
        # Assumption: all records belong to same patient (if not empty)
        patient_id = df.iloc[0]["PatientId"] if not df.empty else None
        
        # If no patient data, return empty DataFrame (no PatientId to emit False for)
        if patient_id is None:
            logger.info("[%s] apply() end | no patient data", self.name)
            return pd.DataFrame(columns=[
                "PatientId", "ConceptName", "StartDateTime", "EndDateTime", 
                "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"
            ])
        
        # Patient timeline clipping to avoid punishing outside observed data
        # The maximum relevant time for a patient is between their earliest StartDateTime and latest EndDateTime, within their *input* data
        patient_start = pd.Timestamp(df["StartDateTime"].min())
        patient_end = pd.Timestamp(df["EndDateTime"].max())
        
        instances_map = {}
        used_event_idxs = set()

        # PRE-RESOLVE PARAMETERS ONCE (not per-rule)
        # For patterns with compliance functions, compute parameters upfront
        has_compliance = any(
            r.cyclic_constraint_compliance or r.value_constraint_compliance 
            for r in self.abstraction_rules
        )
        parameter_values = {}
        if has_compliance and self.parameters:
            # Use earliest event time as reference (will refine per-instance later if needed)
            # For now, just use patient_df's first timestamp as proxy
            if not df.empty:
                parameter_values = self._resolve_parameters(df["StartDateTime"].min(), df)

        for rule in self.abstraction_rules:
            # Extract candidates once per rule (filtered by spec)
            events = self._extract_candidates(df, rule.event_spec)
            contexts = self._extract_candidates(df, rule.context_spec) if rule.context_spec else None

            initiators = self._extract_candidates(df, rule.initiator_spec) if getattr(rule, "initiator_spec", None) else None
            clippers = self._extract_candidates(df, rule.clipper_spec) if getattr(rule, "clipper_spec", None) else None

            # Always use pandas time types for safe arithmetic
            window_duration = pd.Timedelta(rule.time_window)

            # -------------------------
            # Build episodes (episode = time range to scan for windows, between initiator/ absolute start and a clipper / cyclic end)
            # An episode can yield multiple windows
            # A rule can have multiple episodes (one per initiator), as long as the next initiator is after the previous clipper
            # -------------------------
            # Backward compatible default: single episode from patient_start if no initiator spec or no initiator rows
            if initiators is None or initiators.empty:
                initiator_times = [patient_start]
            else:
                initiator_times = (
                    pd.to_datetime(initiators["StartDateTime"], errors="coerce")
                    .dropna()
                    .sort_values()
                    .drop_duplicates()
                    .map(pd.Timestamp)
                    .tolist()
                )

            for init_time in initiator_times:
                # Episode start/end relative to initiator (or patient_start)
                episode_start = pd.Timestamp(init_time) + pd.Timedelta(rule.start)
                episode_end   = pd.Timestamp(init_time) + pd.Timedelta(rule.end)

                # Clip to observed patient timeline (prevents "punishing" after discharge / end-of-data)
                if episode_start < patient_start:
                    episode_start = patient_start
                if episode_end > patient_end:
                    episode_end = patient_end

                # Apply clipper: first clipper at/after initiator time (you can change to episode_start if preferred)
                if clippers is not None and not clippers.empty:
                    clip_candidates = clippers[clippers["StartDateTime"] >= init_time]
                    if not clip_candidates.empty:
                        clip_time = clip_candidates["StartDateTime"].min()
                        if clip_time < episode_end:
                            episode_end = clip_time

                # If episode is empty or too short to fit even one window, skip
                if not (episode_end > episode_start) or (episode_end - episode_start) < window_duration:
                    continue

                current_start = episode_start
                while current_start < episode_end:
                    # Run over windows within episode
                    current_end = current_start + window_duration

                    # IMPORTANT: do not "continue" here or you'll infinite-loop
                    # Skip incomplete trailing window by breaking
                    if current_end > episode_end:
                        break

                    # Check if window is relevant (context satisfied)
                    if not rule.context_satisfied(current_start, current_end, contexts):
                        current_start = current_end
                        continue

                    # Filter events in window (start-inclusive, end-exclusive)
                    events_in_window = events[
                        (events["StartDateTime"] >= current_start) &
                        (events["StartDateTime"] < current_end)
                    ].copy()

                    valid_events_rows = []
                    candidate_indices = []

                    if not events_in_window.empty:
                        for idx, row in events_in_window.iterrows():
                            if idx in used_event_idxs:
                                continue
                            valid_events_rows.append(row)
                            candidate_indices.append(idx)

                    count = len(valid_events_rows)

                    v_score = None
                    c_score = None
                    cyclic_met = None

                    # Cyclic score
                    if rule.cyclic_constraint_compliance:
                        c_score = self._compute_cyclic_compliance(
                            count=count,
                            rule=rule,
                            parameter_values=parameter_values
                        )
                    else:
                        cyclic_met = int(rule.min_occurrences <= count <= rule.max_occurrences)

                    # Value score
                    if rule.value_constraint_compliance:
                        v_score = self._compute_value_compliance(
                            valid_events_rows=valid_events_rows,
                            rule=rule,
                            parameter_values=parameter_values
                        )

                    # Combine (None ignored by your _compute_combined_score; 0.0 enforces failure)
                    w_score = self._compute_combined_score([c_score, v_score, cyclic_met])
                    value_label = "True" if w_score == 1.0 else ("Partial" if w_score > 0 else "False")

                    key = (current_start, current_end)
                    if key not in instances_map:
                        instances_map[key] = {
                            "PatientId": patient_id,
                            "ConceptName": self.name,
                            "StartDateTime": current_start,
                            "EndDateTime": current_end,
                            "Value": value_label,
                            "TimeConstraintScore": None,
                            "ValueConstraintScore": v_score,
                            "CyclicConstraintScore": c_score,
                            "AbstractionType": self.family
                        }
                        used_event_idxs.update(candidate_indices)

                    current_start = current_end

        instances = list(instances_map.values())
        if not instances:
            return pd.DataFrame(columns=[
                "PatientId", "ConceptName", "StartDateTime", "EndDateTime",
                "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"
            ])

        return pd.DataFrame(instances).sort_values("StartDateTime")[
            ["PatientId", "ConceptName", "StartDateTime", "EndDateTime",
            "Value", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore", "AbstractionType"]
        ]
    
    def _compute_cyclic_compliance(
        self,
        count: int,
        rule: CyclicRule,
        parameter_values: Dict[str, float]
    ) -> float:
        """
        Compute cyclic-constraint compliance score.
        If compliance function exists, use it. Otherwise check min/max bounds.
        """
        spec = rule.cyclic_constraint_compliance
        param_vals = [parameter_values[ref] for ref in spec["parameters"]]
        try:
            trapez_node = apply_external_function_on_trapez(
                spec["func_name"], spec["trapez"], "cyclic-constraint", *param_vals
            )
            return trapez_node.compliance_score(count)
        except Exception as e:
            logger.error(f"[{self.name}] Cyclic-constraint compliance error: {e}")
            return 0.0

    def _compute_value_compliance(
        self,
        valid_events_rows: List[pd.Series],
        rule: CyclicRule,
        parameter_values: Dict[str, float]
    ) -> float:
        """
        Compute value-constraint compliance score for a set of events.
        Returns None if no compliance spec or no events to check.
        """
        if not valid_events_rows:
            # No events to check despite compliance is demanded, return 0.0 compliance
            return 0.0
        vcc_spec = rule.value_constraint_compliance
        param_vals = [parameter_values[ref] for ref in vcc_spec["parameters"]]
        
        try:
            trapez_node = apply_external_function_on_trapez(
                vcc_spec["func_name"], vcc_spec["trapez"], "value-constraint", *param_vals
            )
            
            vals = []
            target_refs = set(vcc_spec["targets"])
            
            for row in valid_events_rows:
                c_name = row["ConceptName"]
                # Find matching target spec
                for t_ref in target_refs:
                    t_spec = self.derived_from_map.get(t_ref)
                    if t_spec and t_spec["name"] == c_name:
                        val = row["Value"]
                        idx = t_spec["idx"] if isinstance(val, tuple) else None
                        val = self._extract_value(value=val, idx=idx)
                        try:
                            vals.append(float(val))
                        except (ValueError, TypeError):
                            logger.error(f"[{self.name}] Cannot convert target value to float: {val}. Value-constraint compliance requires numeric values.")
                            pass
                        break
            
            if not vals:
                return 0.0
                
            scores = [trapez_node.compliance_score(v) for v in vals]
            # Score is the avg of all event scores
            return sum(scores) / len(scores)
            
        except Exception as e:
            logger.error(f"[{self.name}] Value-constraint compliance error: {e}")
            return 0.0





from __future__ import annotations
from typing import Tuple, List, Dict, Any, Union
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, EventAbstractionRule, validate_xml_against_schema
from .repository import get_tak_repository
from .raw_concept import RawConcept


class Event(TAK):
    """
    Event abstraction: point-in-time occurrences derived from one or more raw-concepts.
    - Can have multiple derived-from sources (bridging data gaps)
    - Outputs point-in-time records (preserves input StartDateTime/EndDateTime)
    - If no abstraction rules: emit raw values as-is
    - If abstraction rules: apply matching logic with constraints (equal, min, max, range)
    - Event abstraction using ref mechanism (aligned with Event/Pattern).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],  # [{name, tak_type, idx, ref}]
        abstraction_rules: List[EventAbstractionRule],
    ):
        super().__init__(name=name, categories=categories, description=description, family="event")
        self.derived_from = derived_from
        self.abstraction_rules = abstraction_rules
        
        # Build ref lookup map (for abstraction rules)
        self.derived_from_map: Dict[str, Dict[str, Any]] = {
            df["ref"]: df for df in derived_from if "ref" in df
        }

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Event":
        """Parse <event> XML with structural validation."""
        xml_path = Path(xml_path)
        
        # Validate against XSD schema (graceful if lxml not available)
        validate_xml_against_schema(xml_path)
        
        root = ET.parse(xml_path).getroot()
        if root.tag != "event":
            raise ValueError(f"{xml_path} is not an event file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required, one or more) ---
        df_el = root.find("derived-from")
        if df_el is None:
            raise ValueError(f"{name}: missing <derived-from> block")
        
        derived_from = []
        seen_refs = set()
        
        for attr_el in df_el.findall("attribute"):
            if "name" not in attr_el.attrib or "tak" not in attr_el.attrib:
                raise ValueError(f"{name}: <derived-from><attribute> must have 'name' and 'tak' attributes")
            
            tak_type = attr_el.attrib["tak"]
            if tak_type != "raw-concept":
                raise ValueError(f"{name}: <derived-from><attribute tak='{tak_type}'> invalid. Events can only be derived from 'raw-concept'.")
            
            df_spec = {
                "name": attr_el.attrib["name"],
                "tak_type": tak_type,
                "idx": int(attr_el.attrib.get("idx", 0))
            }
            
            # Parse ref (optional, but required if abstraction-rules exist)
            if "ref" in attr_el.attrib:
                ref = attr_el.attrib["ref"]
                if ref in seen_refs:
                    raise ValueError(f"{name}: duplicate ref='{ref}' in <derived-from>")
                seen_refs.add(ref)
                df_spec["ref"] = ref
            
            derived_from.append(df_spec)
        
        if not derived_from:
            raise ValueError(f"{name}: <derived-from> must contain at least one <attribute>")

        # --- abstraction-rules (optional) ---
        abs_rules: List[EventAbstractionRule] = []
        abs_el = root.find("abstraction-rules")
        
        if abs_el is not None:
            # If abstraction-rules exist, all derived-from entries MUST have refs
            missing_refs = [df["name"] for df in derived_from if "ref" not in df]
            if missing_refs:
                raise ValueError(
                    f"{name}: <abstraction-rules> present but some <derived-from> attributes missing 'ref': {missing_refs}"
                )
            
            for rule_el in abs_el.findall("rule"):
                val = rule_el.attrib["value"]
                op = rule_el.attrib.get("operator", "or")
                constraints: Dict[str, List[Dict[str, Any]]] = {}  # {ref: [{constraint_spec}]}
                
                for attr_el in rule_el.findall("attribute"):
                    if "ref" not in attr_el.attrib:
                        raise ValueError(f"{name}: <abstraction-rules><attribute> must have 'ref' attribute")
                    
                    ref = attr_el.attrib["ref"]
                    
                    # Validate ref exists in derived-from
                    if ref not in seen_refs:
                        raise ValueError(f"{name}: rule references unknown ref='{ref}' (not found in <derived-from>)")
                    
                    allowed = []
                    for av in attr_el.findall("allowed-value"):
                        constraint = {}
                        has_equal = "equal" in av.attrib
                        has_min = "min" in av.attrib
                        has_max = "max" in av.attrib
                        
                        # Validation: reject invalid combinations
                        if has_equal and (has_min or has_max):
                            raise ValueError(f"{name}: <allowed-value> cannot have both 'equal' and 'min'/'max' attributes")
                        if not (has_equal or has_min or has_max):
                            raise ValueError(f"{name}: <allowed-value> must have at least one of: 'equal', 'min', 'max'")
                        
                        # Parse constraint type
                        if has_equal:
                            constraint["type"] = "equal"
                            constraint["value"] = av.attrib["equal"]
                        elif has_min and has_max:
                            constraint["type"] = "range"  # Internal representation only
                            constraint["min"] = float(av.attrib["min"])
                            constraint["max"] = float(av.attrib["max"])
                        elif has_min:
                            constraint["type"] = "min"
                            constraint["value"] = float(av.attrib["min"])
                        elif has_max:
                            constraint["type"] = "max"
                            constraint["value"] = float(av.attrib["max"])
                        
                        allowed.append(constraint)
                    
                    constraints[ref] = allowed
                
                abs_rules.append(EventAbstractionRule(val, op, constraints))

        event = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            abstraction_rules=abs_rules,
        )
        event.validate()
        return event

    def validate(self) -> None:
        """Business logic validation using global TAKRepository."""
        repo = get_tak_repository()

        # 1) Check all derived_from TAKs exist and are RawConcepts
        for df in self.derived_from:
            parent_tak = repo.get(df["name"])
            if parent_tak is None:
                raise ValueError(f"{self.name}: derived_from='{df['name']}' not found in TAK repository")
            
            # Only RawConcepts allowed
            if not isinstance(parent_tak, RawConcept):
                raise ValueError(f"{self.name}: derived_from='{df['name']}' is not a RawConcept (found {parent_tak.family})")

        # 2) Validate abstraction rules (ref-based)
        if self.abstraction_rules:
            for rule in self.abstraction_rules:
                # Check operator="and" only for same-source refs
                if rule.operator == "and":
                    ref_sources = set()
                    for ref in rule.constraints.keys():
                        df_entry = self.derived_from_map.get(ref)
                        if df_entry is None:
                            raise ValueError(f"{self.name}: rule references unknown ref='{ref}'")
                        ref_sources.add(df_entry["name"])
                    
                    if len(ref_sources) > 1:
                        raise ValueError(f"{self.name}: operator='and' requires all refs from same raw-concept source (found: {ref_sources})")

                # Check allowed values are valid for the ref's attribute
                for ref, constraints in rule.constraints.items():
                    df_entry = self.derived_from_map.get(ref)
                    if df_entry is None:
                        continue
                    
                    parent_tak = repo.get(df_entry["name"])
                    if isinstance(parent_tak, RawConcept):
                        # Validate constraints against raw-concept's allowed values
                        if parent_tak.concept_type == "raw":
                            attr_idx = df_entry["idx"]
                            if attr_idx >= len(parent_tak.tuple_order):
                                raise ValueError(f"{self.name}: ref='{ref}' idx={attr_idx} out of bounds for '{parent_tak.name}' (tuple size={len(parent_tak.tuple_order)})")
                            attr_name = parent_tak.tuple_order[attr_idx]
                            parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                        else:
                            # raw-numeric/nominal/boolean: single attribute
                            parent_attr = parent_tak.attributes[0] if parent_tak.attributes else None
                        
                        if parent_attr:
                            # Validate constraint values against attribute type
                            for c in constraints:
                                if parent_attr["type"] == "nominal":
                                    if c["type"] == "equal" and c["value"] not in parent_attr.get("allowed", []):
                                        raise ValueError(f"{self.name}: ref='{ref}' constraint value '{c['value']}' not in allowed values")
                                elif parent_attr["type"] == "boolean":
                                    if c["type"] == "equal" and c["value"] != "True":
                                        raise ValueError(f"{self.name}: ref='{ref}' boolean constraint must be 'True'")

        # If multiple attributes, or any attribute is numeric, or any raw concept idx points to numeric, abstraction_rules must be defined
        num_attrs = len(self.derived_from)
        must_have_rules = False
        for df in self.derived_from:
            parent_tak = repo.get(df["name"])
            idx = df.get("idx", 0)
            if isinstance(parent_tak, RawConcept):
                if parent_tak.concept_type == "raw":
                    # Check if idx points to a numeric attribute
                    if idx < len(parent_tak.tuple_order):
                        attr_name = parent_tak.tuple_order[idx]
                        parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                        if parent_attr and parent_attr["type"] == "numeric":
                            must_have_rules = True
                else:
                    # raw-numeric/nominal/boolean: single attribute
                    parent_attr = parent_tak.attributes[0] if parent_tak.attributes else None
                    if parent_attr and parent_attr["type"] == "numeric":
                        must_have_rules = True
        if num_attrs > 1 or must_have_rules:
            if not self.abstraction_rules:
                raise ValueError(
                    f"{self.name}: Event derived from multiple attributes or with numeric attribute(s) "
                    "must define abstraction rules."
                )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply event abstraction to input data (from one or more derived-from raw-concepts).
        Input:  PatientId, ConceptName(any derived-from), StartDateTime, EndDateTime, Value(tuple), AbstractionType
        Output: PatientId, ConceptName(self.name), StartDateTime, EndDateTime, Value(event_label), AbstractionType
        """
        if df.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))

        # Filter to only relevant derived-from concepts
        valid_concepts = {df_item["name"] for df_item in self.derived_from}
        df = df[df["ConceptName"].isin(valid_concepts)].copy()
        if df.empty:
            logger.info("[%s] apply() end | post-filter=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # If no abstraction rules: emit raw values as-is, but always extract the value using idx
        if not self.abstraction_rules:
            df["ConceptName"] = self.name
            df["AbstractionType"] = self.family
            # Extract value using idx for each row
            def extract_value(row):
                # Find the derived_from spec for this ConceptName
                df_spec = next((d for d in self.derived_from if d["name"] == row["ConceptName"]), None)
                idx = df_spec.get("idx", 0) if df_spec else 0
                val = row["Value"]
                if isinstance(val, tuple):
                    return val[idx] if idx < len(val) else None
                return val
            df["Value"] = df.apply(extract_value, axis=1)
            logger.info("[%s] apply() end (no rules) | output_rows=%d", self.name, len(df))
            return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

        # Apply abstraction rules
        df = self._abstract(df)
        # Keep input EndDateTime (preserve time range)
        df["ConceptName"] = self.name
        df["AbstractionType"] = self.family

        logger.info("[%s] apply() end | output_rows=%d", self.name, len(df))
        return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _abstract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply abstraction rules to input tuples → event labels (VECTORIZED)."""
        if df.empty:
            return df
        
        # For each rule, compute a boolean mask indicating which rows match
        # Then assign the rule's value to matching rows
        results = []
        
        for rule in self.abstraction_rules:
            # Build mask for this rule
            masks = []
            for ref, constraint_list in rule.constraints.items():
                df_entry = self.derived_from_map.get(ref)
                if df_entry is None:
                    masks.append(pd.Series([False] * len(df), index=df.index))
                    continue
                
                attr_mask = (df["ConceptName"] == df_entry["name"])
                if not attr_mask.any():
                    masks.append(pd.Series([False] * len(df), index=df.index))
                    continue
                
                # Extract values (handle tuples)
                idx = df_entry["idx"]
                values = df.loc[attr_mask, "Value"].apply(
                    lambda v: v[idx] if isinstance(v, tuple) and idx < len(v) else v
                )
                
                # Apply constraints vectorized
                constraint_mask = pd.Series([False] * len(df), index=df.index)
                for c in constraint_list:
                    if c["type"] == "equal":
                        constraint_mask.loc[attr_mask] |= (values.astype(str) == c["value"])
                    elif c["type"] == "min":
                        try:
                            constraint_mask.loc[attr_mask] |= (pd.to_numeric(values, errors='coerce') >= c["value"])
                        except Exception:
                            pass
                    elif c["type"] == "max":
                        try:
                            constraint_mask.loc[attr_mask] |= (pd.to_numeric(values, errors='coerce') <= c["value"])
                        except Exception:
                            pass
                    elif c["type"] == "range":
                        try:
                            numeric_vals = pd.to_numeric(values, errors='coerce')
                            constraint_mask.loc[attr_mask] |= (
                                (numeric_vals >= c["min"]) & (numeric_vals <= c["max"])
                            )
                        except Exception:
                            pass
                
                masks.append(constraint_mask)
            
            # Combine masks based on operator
            if rule.operator == "and":
                final_mask = pd.Series([True] * len(df), index=df.index)
                for m in masks:
                    final_mask &= m
            else:  # "or"
                final_mask = pd.Series([False] * len(df), index=df.index)
                for m in masks:
                    final_mask |= m
            
            # Assign value to matching rows
            matched_df = df[final_mask].copy()
            if not matched_df.empty:
                matched_df["Value"] = rule.value
                results.append(matched_df)
        
        # Combine all matched rows (drop duplicates if multiple rules match same row)
        if results:
            out = pd.concat(results, ignore_index=True).drop_duplicates(
                subset=["PatientId", "ConceptName", "StartDateTime"], keep="first"
            )
            return out
        else:
            return pd.DataFrame(columns=df.columns)

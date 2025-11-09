from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union, Set
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .utils import parse_duration
from .tak import TAK, StateAbstractionRule, validate_xml_against_schema
from .repository import get_tak_repository
from .raw_concept import RawConcept
from .event import Event


class State(TAK):
    """
    State abstraction: abstract multi-attribute concepts, merge adjacent intervals.
    - Handles nominal (no discretization), numeric (single-attr), and complex (multi-attr tuples).
    - Requires derived_from to be a RawConcept or Event TAK.
    - A state can only be derived from one TAK, but that TAK can produce multi-valued tuples.
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: str,
        good_after: timedelta,
        interpolate: bool,
        max_skip: int,
        abstraction_rules: List[StateAbstractionRule],
    ):
        super().__init__(name=name, categories=categories, description=description, family="state")
        self.derived_from = derived_from
        self.good_after = good_after
        self.interpolate = interpolate
        self.max_skip = max_skip
        self.abstraction_rules = abstraction_rules

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "State":
        """Parse <state> XML with structural validation."""
        xml_path = Path(xml_path)
        
        # Validate against XSD schema (graceful if lxml not available)
        validate_xml_against_schema(xml_path)
        
        root = ET.parse(xml_path).getroot()
        if root.tag != "state":
            raise ValueError(f"{xml_path} is not a state file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required, must be single attribute reference) ---
        df_elements = root.findall("derived-from")
        if len(df_elements) == 0:
            raise ValueError(f"{name}: missing <derived-from name='...' tak='...'> block")
        if len(df_elements) > 1:
            raise ValueError(f"{name}: State can only have ONE <derived-from> block (found {len(df_elements)})")
        
        df_el = df_elements[0]
        
        # Check if <derived-from> contains <attribute> children (Event/Context pattern)
        attr_children = df_el.findall("attribute")
        if attr_children:
            raise ValueError(
                f"{name}: State's <derived-from> must reference a SINGLE TAK (format: <derived-from name='...' tak='...'/>). "
                f"Found {len(attr_children)} <attribute> children, which is the Event/Context pattern. "
                f"States can only derive from one RawConcept or Event TAK."
            )
        
        # Validate required attributes (name, tak)
        if "name" not in df_el.attrib or "tak" not in df_el.attrib:
            raise ValueError(
                f"{name}: <derived-from> must have 'name' and 'tak' attributes. "
                f"Correct format: <derived-from name='PARENT_TAK_NAME' tak='raw-concept|event'/>"
            )
        
        derived_from = df_el.attrib["name"]

        # --- persistence (required) ---
        pers_el = root.find("persistence")
        if pers_el is None:
            raise ValueError(f"{name}: missing <persistence>")
        ga = parse_duration(pers_el.attrib.get("good-after", "0h"))
        if ga <= timedelta(0):
            raise ValueError(f"{name}: good-after must be positive")
        interpolate = pers_el.attrib.get("interpolate", "false").lower() == "true"
        max_skip = int(pers_el.attrib.get("max-skip", "0"))

        # --- abstraction-rules (now unified, handles discretization internally) ---
        abs_rules: List[StateAbstractionRule] = []
        abs_el = root.find("abstraction-rules")
        
        if abs_el is not None:
            for rule_el in abs_el.findall("rule"):
                value = rule_el.attrib.get("value", "")
                if not value:
                    raise ValueError(f"{name}: <rule> missing value attribute")
                
                operator = rule_el.attrib.get("operator", "or").lower()
                if operator not in ("and", "or"):
                    raise ValueError(f"{name}: <rule value='{value}'> operator='{operator}' must be 'and' or 'or'")
                
                # Parse attributes with integrated discretization ranges
                constraints: Dict[int, Dict[str, Any]] = {}
                
                for attr_el in rule_el.findall("attribute"):
                    try:
                        idx = int(attr_el.attrib.get("idx", 0))
                    except ValueError:
                        raise ValueError(f"{name}: <rule value='{value}'> <attribute idx='{attr_el.attrib.get('idx')}'> idx must be integer")
                    
                    # Parse allowed-values
                    rules = []
                    
                    for allowed_val_el in attr_el.findall("allowed-value"):
                        # For numeric: min/max attributes
                        if allowed_val_el.attrib.get("min") is not None or allowed_val_el.attrib.get("max") is not None:
                            try:
                                min_val = float(allowed_val_el.attrib["min"]) if "min" in allowed_val_el.attrib else -float('inf')
                                max_val = float(allowed_val_el.attrib["max"]) if "max" in allowed_val_el.attrib else float('inf')
                                rules.append({"min": min_val, "max": max_val})
                            except ValueError as e:
                                raise ValueError(f"{name}: <allowed-value> min/max must be numeric: {e}")
                        
                        # For nominal/boolean: equal attribute
                        elif allowed_val_el.attrib.get("equal") is not None:
                            rules.append({"equal": allowed_val_el.attrib["equal"]})
                        else:
                            raise ValueError(f"{name}: <allowed-value> must have either 'equal' or 'min'/'max' attributes")
                    
                    if not rules:
                        raise ValueError(f"{name}: <attribute idx={idx}> must have <allowed-value> children with min/max or equal")
                    
                    # Store WITHOUT type (will be inferred in validate())
                    constraints[idx] = {
                        "type": None,  # To be filled in validate()
                        "rules": rules,
                    }
                
                abs_rules.append(StateAbstractionRule(value, operator, constraints))

        state = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            good_after=ga,
            interpolate=interpolate,
            max_skip=max_skip,
            abstraction_rules=abs_rules,
        )
        state.validate()
        return state

    def validate(self) -> None:
        """Business logic validation using global TAKRepository."""
        repo = get_tak_repository()

        # 1) Check derived_from references a valid RawConcept OR Event
        parent_tak = repo.get(self.derived_from)
        if parent_tak is None:
            raise ValueError(f"{self.name}: derived_from='{self.derived_from}' not found in TAK repository")
        # Allow Event TAKs as parents (treat like raw-nominal)
        if not isinstance(parent_tak, (RawConcept, Event)):
            raise ValueError(f"{self.name}: derived_from='{self.derived_from}' is not a RawConcept or Event (found {parent_tak.family})")

        # 2) Determine tuple size
        if isinstance(parent_tak, RawConcept):
            if parent_tak.concept_type == "raw":
                tuple_size = len(parent_tak.tuple_order)
            else:
                tuple_size = 1
        elif isinstance(parent_tak, Event):
            # Events always emit single-value strings (not tuples)
            tuple_size = 1
        else:
            tuple_size = 1

        # 3) Check abstraction-rules requirement based on parent type
        is_numeric = (
            isinstance(parent_tak, RawConcept) and 
            parent_tak.concept_type in ("raw", "raw-numeric")
        )
        is_multi_attr = tuple_size > 1
        
        # Rule: numeric or multi-attr RAW concepts MUST have abstraction-rules
        if (is_numeric or is_multi_attr) and not self.abstraction_rules:
            raise ValueError(
                f"{self.name}: numeric or multi-attribute raw-concept '{self.derived_from}' "
                f"must have <abstraction-rules> block"
            )

        # 4) Infer types and validate abstraction rules structure
        if self.abstraction_rules:
            for rule in self.abstraction_rules:
                # Check all constraint indices are within bounds
                for idx in rule.constraints.keys():
                    if idx >= tuple_size:
                        raise ValueError(
                            f"{self.name}: rule '{rule.value}' references attr idx={idx} "
                            f"but tuple_size={tuple_size}"
                        )
                
                # INFER type from parent TAK and validate each constraint
                for idx, constraint_spec in rule.constraints.items():
                    rules_list = constraint_spec.get("rules", [])
                    
                    if not rules_list:
                        raise ValueError(
                            f"{self.name}: rule '{rule.value}' attr idx={idx} has no rules"
                        )
                    
                    # Get parent attribute info
                    parent_attr = None
                    if isinstance(parent_tak, RawConcept):
                        if parent_tak.concept_type == "raw":
                            # Multi-attribute: get from tuple_order
                            if idx < len(parent_tak.tuple_order):
                                attr_name = parent_tak.tuple_order[idx]
                                parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                        elif idx == 0:
                            # Single-attribute: first attribute
                            if parent_tak.attributes:
                                parent_attr = parent_tak.attributes[0]
                    elif isinstance(parent_tak, Event):
                        # Events emit strings (nominal)
                        parent_attr = {"type": "nominal", "name": self.derived_from}
                    
                    if not parent_attr:
                        raise ValueError(f"{self.name}: cannot determine type for attribute idx={idx}")
                    
                    # INFER and SET type in constraint_spec
                    attr_type = parent_attr["type"]
                    constraint_spec["type"] = attr_type
                    
                    # VALIDATION FOR NUMERIC ATTRIBUTES
                    if attr_type == "numeric":
                        # Validate all rules have min/max (no 'equal' for numeric)
                        for r in rules_list:
                            if "equal" in r:
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} is numeric "
                                    f"but has <allowed-value equal='...'> (must use min/max)"
                                )
                            
                            # Check range validity
                            min_val = r.get("min", -float('inf'))
                            max_val = r.get("max", float('inf'))
                            
                            if min_val >= max_val:
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} "
                                    f"has invalid range: min={min_val} >= max={max_val}"
                                )
                            
                            # Check for infinite ranges (both sides unbounded)
                            if min_val == -float('inf') and max_val == float('inf'):
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} "
                                    f"has unbounded range (both min and max infinite)"
                                )
                        
                        # Check for overlaps and gaps across ALL rules for this attribute
                        sorted_ranges = sorted(rules_list, key=lambda r: r.get("min", -float('inf')))
                        
                        for i in range(len(sorted_ranges) - 1):
                            r1, r2 = sorted_ranges[i], sorted_ranges[i+1]
                            r1_max = r1.get("max", float('inf'))
                            r2_min = r2.get("min", -float('inf'))
                            
                            # Check overlap (r1.max > r2.min means overlap, since ranges are [min, max))
                            if r1_max > r2_min:
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} "
                                    f"has overlapping ranges: [{r1.get('min', '-inf')}, {r1_max}) "
                                    f"overlaps [{r2_min}, {r2.get('max', 'inf')})"
                                )
                            
                            # Check gap (r1.max < r2.min means gap)
                            if r1_max < r2_min:
                                logger.warning(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} "
                                    f"has gap in ranges: [{r1_max}, {r2_min}) is uncovered. "
                                    f"Values in this range will be filtered out."
                                )
                    
                    # VALIDATION FOR NOMINAL/BOOLEAN ATTRIBUTES
                    else:  # nominal or boolean
                        # Validate all rules have 'equal' (no min/max for nominal/boolean)
                        for r in rules_list:
                            if "min" in r or "max" in r:
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} is {attr_type} "
                                    f"but has <allowed-value min='...' max='...'> (must use equal='...')"
                                )
                            
                            if "equal" not in r:
                                raise ValueError(
                                    f"{self.name}: rule '{rule.value}' attr idx={idx} is {attr_type} "
                                    f"but <allowed-value> missing 'equal' attribute"
                                )

        # 5) Check coverage: warn if parent's attributes/values are not fully covered by rules
        if self.abstraction_rules and isinstance(parent_tak, RawConcept):
            # Build map of what each rule covers per attribute
            covered_per_attr: Dict[int, Set[str]] = {}
            numeric_ranges_per_attr: Dict[int, List[Tuple[float, float]]] = {}
            
            for rule in self.abstraction_rules:
                for idx, constraint_spec in rule.constraints.items():
                    attr_type = constraint_spec.get("type", "nominal")
                    rules_list = constraint_spec.get("rules", [])
                    
                    if attr_type == "numeric":
                        if idx not in numeric_ranges_per_attr:
                            numeric_ranges_per_attr[idx] = []
                        for r in rules_list:
                            min_val = r.get("min", -float('inf'))
                            max_val = r.get("max", float('inf'))
                            numeric_ranges_per_attr[idx].append((min_val, max_val))
                    else:  # nominal/boolean
                        if idx not in covered_per_attr:
                            covered_per_attr[idx] = set()
                        for r in rules_list:
                            covered_per_attr[idx].add(r.get("equal", ""))
            
            # Check each parent attribute
            for idx in range(tuple_size):
                # Get parent attribute info
                if parent_tak.concept_type == "raw":
                    attr_name = parent_tak.tuple_order[idx]
                    parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                else:
                    attr_name = parent_tak.name
                    parent_attr = parent_tak.attributes[0] if parent_tak.attributes else None
                
                if not parent_attr:
                    continue
                
                # Check if attribute is referenced at all
                if idx not in covered_per_attr and idx not in numeric_ranges_per_attr:
                    logger.warning(
                        f"{self.name}: attribute idx={idx} ('{attr_name}', type={parent_attr['type']}) "
                        f"from '{self.derived_from}' is not referenced in any abstraction rule. "
                        f"This attribute will be ignored during abstraction."
                    )
                    continue
                
                # For nominal attributes: check if all allowed values are covered
                if parent_attr["type"] == "nominal" and parent_attr.get("allowed"):
                    covered = covered_per_attr.get(idx, set())
                    parent_allowed = set(parent_attr["allowed"])
                    uncovered = parent_allowed - covered
                    
                    if uncovered:
                        logger.warning(
                            f"{self.name}: attribute idx={idx} ('{attr_name}') has allowed values "
                            f"in parent '{self.derived_from}' that are not covered by abstraction rules: {uncovered}. "
                            f"Rows with these values will be filtered out."
                        )
                
                # For boolean attributes: check if "True"/"False" are covered
                elif parent_attr["type"] == "boolean":
                    covered = covered_per_attr.get(idx, set())
                    # Boolean values come as strings "True" or "False"
                    uncovered = {"True", "False"} - covered
                    
                    if uncovered:
                        logger.warning(
                            f"{self.name}: attribute idx={idx} ('{attr_name}', boolean) "
                            f"has values not covered by abstraction rules: {uncovered}. "
                            f"Rows with these values will be filtered out."
                        )
                
                # For numeric attributes: warn about unbounded ranges
                elif parent_attr["type"] == "numeric":
                    ranges = numeric_ranges_per_attr.get(idx, [])
                    if ranges:
                        # Check if lower bound is covered
                        has_lower_bound = any(min_val == -float('inf') for min_val, _ in ranges)
                        # Check if upper bound is covered
                        has_upper_bound = any(max_val == float('inf') for _, max_val in ranges)
                        
                        if not has_lower_bound:
                            logger.warning(
                                f"{self.name}: attribute idx={idx} ('{attr_name}', numeric) "
                                f"has no range covering -infinity. Very low values will be filtered out."
                            )
                        
                        if not has_upper_bound:
                            logger.warning(
                                f"{self.name}: attribute idx={idx} ('{attr_name}', numeric) "
                                f"has no range covering +infinity. Very high values will be filtered out."
                            )

        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply state abstraction to pre-computed raw-concept df.
        Input:  PatientId, ConceptName(=self.derived_from), StartDateTime, EndDateTime, Value(tuple), AbstractionType
        Output: PatientId, ConceptName(=self.name), StartDateTime, EndDateTime, Value(state_label), AbstractionType
        """
        if df.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))

        # 1) Abstract: apply unified abstraction rules (which internally discretize)
        df = self._abstract(df)
        if df.empty:
            logger.info("[%s] apply() end | post-abstract=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # 2) Merge: concatenate adjacent identical states
        merged = self._merge_intervals(df)
        out = pd.DataFrame(merged) if merged else pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
        
        out["ConceptName"] = self.name
        out["AbstractionType"] = self.family
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
        return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _abstract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply abstraction rules to raw tuple values → final state labels.
        Rules internally handle discretization of numeric attributes.
        Returns first matching rule per tuple.
        """
        if not self.abstraction_rules:
            # No abstraction rules: keep raw values as-is (only nominal/boolean TAKs)
            logger.warning(f"{self.name}: no abstraction rules; passing through raw values")
            df["Value"] = df["Value"].apply(lambda t: str(t[0]) if isinstance(t, tuple) else str(t))
            return df

        def apply_rules_to_tuple(tup: Tuple[str, ...]) -> Optional[str]:
            """Apply rules to raw value, return first match."""
            for rule in self.abstraction_rules:
                if rule.matches(tup):
                    return rule.value
            return None  # No rule matched

        df["Value"] = df["Value"].apply(apply_rules_to_tuple)
        return df[df["Value"].notna()]

    def _merge_intervals(self, df: pd.DataFrame) -> List[dict]:
        """
        Merge adjacent identical state values using persistence windows + interpolation.
        Each interval extends to last_merged_sample_time + good_after (or next sample's start, whichever is earlier).
        
        Returns:
            List of merged interval dicts
        """
        if df.empty:
            return []

        # Sort by StartDateTime and prepare for vectorized operations
        df = df.sort_values("StartDateTime").reset_index(drop=True)
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        
        merged: List[dict] = []
        n = len(df)
        i = 0
        
        # Pre-compute numpy arrays for faster access
        start_times = df["StartDateTime"].values
        values = df["Value"].values
        good_after_td = pd.Timedelta(self.good_after)
        
        while i < n:
            interval_start = start_times[i]
            interval_value = values[i]
            
            # Extract row data once
            row = df.iloc[i]
            interval_pid = row["PatientId"]
            interval_cname = row["ConceptName"]
            interval_abstype = row["AbstractionType"]
            
            # Collect all rows that merge into this interval
            merged_indices = [i]
            skip_count = 0
            j = i + 1
            
            # OPTIMIZATION: Compute window boundary once
            window_end = interval_start + good_after_td
            
            # Merge loop
            while j < n:
                next_start = start_times[j]
                next_value = values[j]
                same_value = (next_value == interval_value)
                within_window = (next_start <= window_end)
                
                if same_value and within_window:
                    merged_indices.append(j)
                    skip_count = 0
                    j += 1
                elif not same_value and within_window and self.interpolate and skip_count < self.max_skip:
                    # Interpolation: check if next row is same_value and within_window
                    if j + 1 < n:
                        peek_start = start_times[j + 1]
                        peek_value = values[j + 1]
                        if peek_value == interval_value and peek_start <= window_end:
                            skip_count += 1
                            j += 1
                            continue
                    break
                else:
                    break
            
            # Compute interval EndDateTime
            last_merged_time = start_times[merged_indices[-1]]
            interval_end = last_merged_time + good_after_td
            
            # Cap at next sample's start if it arrives sooner
            if j < n:
                next_sample_time = start_times[j]
                if next_sample_time < interval_end:
                    interval_end = next_sample_time
            
            merged.append({
                "PatientId": interval_pid,
                "ConceptName": interval_cname,
                "StartDateTime": pd.Timestamp(interval_start),  # convert back to Timestamp for output
                "EndDateTime": pd.Timestamp(interval_end),
                "Value": interval_value,
                "AbstractionType": interval_abstype
            })
            
            i = j

        return merged
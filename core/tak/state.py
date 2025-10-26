from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Literal, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

from .utils import parse_duration
from .tak import TAK, get_tak_repository, StateDiscretizationRule, StateAbstractionRule
from .raw_concept import RawConcept
from .event import Event


class State(TAK):
    """
    State abstraction: discretize (for numeric) + abstract (for multi-attr concepts) + merge intervals (based on same rule match).
    - Handles nominal (no discretization), numeric (single-attr), and complex (multi-attr tuples).
    - Requires derived_from raw-concept(s) to be pre-applied (TAK object cached in TAKRepository, df calculated on demand).
    
    A state will only be derived from one raw concept TAK, but that TAK can produce multiple values in it's tuple.
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: str,  # name of RawConcept TAK
        good_after: timedelta,
        interpolate: bool,
        max_skip: int,
        discretization_rules: List[StateDiscretizationRule],
        abstraction_rules: List[StateAbstractionRule],
        abstraction_order: Literal["first","all"] = "first",
    ):
        super().__init__(name=name, categories=categories, description=description, family="state")
        self.derived_from = derived_from
        self.good_after = good_after
        self.interpolate = interpolate
        self.max_skip = max_skip
        self.discretization_rules = discretization_rules
        self.abstraction_rules = abstraction_rules
        self.abstraction_order = abstraction_order

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "State":
        """Parse <state> XML with structural validation."""
        root = ET.parse(xml_path).getroot()
        if root.tag != "state":
            raise ValueError(f"{xml_path} is not a state file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required) ---
        df_el = root.find("derived-from")
        if df_el is None or "name" not in df_el.attrib or "tak" not in df_el.attrib:
            raise ValueError(f"{name}: missing <derived-from name='state_name' tak='raw-concept'>")
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

        # --- discretization-rules (optional) ---
        disc_rules: List[StateDiscretizationRule] = []
        disc_el = root.find("discretization-rules")
        if disc_el is not None:
            for attr_el in disc_el.findall("attribute"):
                idx = int(attr_el.attrib["idx"])
                for rule_el in attr_el.findall("rule"):
                    val = rule_el.attrib["value"]
                    min_v = float(rule_el.attrib["min"]) if "min" in rule_el.attrib else None
                    max_v = float(rule_el.attrib["max"]) if "max" in rule_el.attrib else None
                    disc_rules.append(StateDiscretizationRule(idx, val, min_v, max_v))

        # --- abstraction-rules (optional, for 'raw') ---
        abs_rules: List[StateAbstractionRule] = []
        abs_el = root.find("abstraction-rules")
        abs_order: Literal["first","all"] = "first"
        if abs_el is not None:
            abs_order = abs_el.attrib.get("order", "first")
            for rule_el in abs_el.findall("rule"):
                val = rule_el.attrib["value"]
                op = rule_el.attrib.get("operator", "and")
                constraints: Dict[int, List[str]] = {}
                for attr_el in rule_el.findall("attribute"):
                    idx = int(attr_el.attrib["idx"])
                    allowed = [av.attrib["value"] for av in attr_el.findall("allowed-value")]
                    constraints[idx] = allowed
                abs_rules.append(StateAbstractionRule(val, op, constraints))

        state = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            good_after=ga,
            interpolate=interpolate,
            max_skip=max_skip,
            discretization_rules=disc_rules,
            abstraction_rules=abs_rules,
            abstraction_order=abs_order,
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

        # 3) Validate discretization rules coherence
        if self.discretization_rules:
            # Check all rule indices are within bounds
            for rule in self.discretization_rules:
                if rule.attr_idx >= tuple_size:
                    raise ValueError(f"{self.name}: discretization rule attr_idx={rule.attr_idx} out of bounds (tuple size={tuple_size})")

            # Check for overlaps AND gaps per attribute
            rules_by_idx = defaultdict(list)
            for rule in self.discretization_rules:
                rules_by_idx[rule.attr_idx].append(rule)

            for idx, rules in rules_by_idx.items():
                # Sort by min to detect overlaps/gaps
                sorted_rules = sorted(rules, key=lambda r: r.min)
                
                # Check overlaps
                for i in range(len(sorted_rules) - 1):
                    r1, r2 = sorted_rules[i], sorted_rules[i+1]
                    if r1.max > r2.min and r1.max != r2.min:
                        raise ValueError(f"{self.name}: overlapping discretization ranges at idx={idx}: "
                                       f"[{r1.min}, {r1.max}) overlaps [{r2.min}, {r2.max})")
                
                # Check gaps (warn only, not fatal)
                for i in range(len(sorted_rules) - 1):
                    r1, r2 = sorted_rules[i], sorted_rules[i+1]
                    if r1.max < r2.min:
                        logger.warning(f"{self.name}: gap in discretization ranges at idx={idx}: "
                                     f"[{r1.max}, {r2.min}) is uncovered")

        # 4) Validate abstraction rules reference valid indices
        if self.abstraction_rules:
            for rule in self.abstraction_rules:
                for idx in rule.constraints.keys():
                    if idx >= tuple_size:
                        raise ValueError(f"{self.name}: abstraction rule constraint idx={idx} out of bounds (tuple size={tuple_size})")

        # 5) NEW: Check numeric attributes are discretized if referenced in abstraction rules
        if self.abstraction_rules and parent_tak.concept_type == "raw":
            referenced_indices = set()
            for rule in self.abstraction_rules:
                referenced_indices.update(rule.constraints.keys())

            for idx in referenced_indices:
                attr_name = parent_tak.tuple_order[idx]
                parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                if parent_attr is None:
                    continue

                has_discretization = any(r.attr_idx == idx for r in self.discretization_rules)

                # ERROR: numeric attribute referenced in abstraction without discretization
                if parent_attr["type"] == "numeric" and not has_discretization:
                    raise ValueError(f"{self.name}: numeric attribute at idx={idx} ('{attr_name}') is referenced in abstraction rules "
                                   f"but has no discretization rules. Add discretization.")

        # 6) WARN: discretized or nominal/boolean attributes NOT referenced in abstraction rules
        if self.abstraction_rules and parent_tak.concept_type == "raw":
            referenced_indices = set()
            for rule in self.abstraction_rules:
                referenced_indices.update(rule.constraints.keys())

            for idx, attr_name in enumerate(parent_tak.tuple_order):
                parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                if parent_attr is None:
                    continue

                has_discretization = any(r.attr_idx == idx for r in self.discretization_rules)
                is_nominal_or_boolean = parent_attr["type"] in ("nominal", "boolean")

                if (has_discretization or is_nominal_or_boolean) and idx not in referenced_indices:
                    logger.warning(f"{self.name}: attribute at idx={idx} ('{attr_name}', type={parent_attr['type']}) "
                                 f"is discretized/nominal/boolean but never referenced in abstraction rules. "
                                 f"This may be intentional (silent abstraction) or a mistake.")

        # 7) Check abstraction rules cover all possible discrete values (existing code)
        if self.abstraction_rules:
            # Build set of all possible discrete values per attribute index
            possible_values_per_idx: Dict[int, set] = defaultdict(set)

            if parent_tak.concept_type == "raw":
                # Collect from discretization rules or parent attributes
                for idx, attr_name in enumerate(parent_tak.tuple_order):
                    parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
                    if parent_attr is None:
                        continue

                    # Check if discretization rules exist for this idx
                    disc_for_idx = [r for r in self.discretization_rules if r.attr_idx == idx]
                    if disc_for_idx:
                        # Use discretization output values
                        possible_values_per_idx[idx] = {r.value for r in disc_for_idx}
                    else:
                        # Use parent's allowed values (for nominal/boolean)
                        if parent_attr["type"] == "nominal" and parent_attr["allowed"]:
                            possible_values_per_idx[idx] = set(parent_attr["allowed"])
                        elif parent_attr["type"] == "boolean":
                            possible_values_per_idx[idx] = {"True"}  # boolean → string "True"
            else:
                # Non-raw: single attribute at idx=0
                parent_attr = parent_tak.attributes[0]
                if self.discretization_rules:
                    # Use discretization outputs
                    disc_for_idx = [r for r in self.discretization_rules if r.attr_idx == 0]
                    possible_values_per_idx[0] = {r.value for r in disc_for_idx}
                else:
                    # Use parent's allowed values
                    if parent_attr["type"] == "nominal" and parent_attr["allowed"]:
                        possible_values_per_idx[0] = set(parent_attr["allowed"])
                    elif parent_attr["type"] == "boolean":
                        possible_values_per_idx[0] = {"True"}

            # Check if abstraction rules cover all combinations (warn if not)
            if possible_values_per_idx:
                # Collect all values referenced in abstraction rules per idx
                covered_per_idx: Dict[int, set] = defaultdict(set)
                for rule in self.abstraction_rules:
                    for idx, allowed in rule.constraints.items():
                        covered_per_idx[idx].update(allowed)

                # Compare covered vs possible
                for idx, possible in possible_values_per_idx.items():
                    covered = covered_per_idx.get(idx, set())
                    uncovered = possible - covered
                    if uncovered:
                        logger.warning(f"{self.name}: abstraction rules do not cover all discrete values at idx={idx}. "
                                     f"Uncovered: {uncovered}. Rows with these values will be filtered out.")

        # All checks passed
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

        # 1) Discretize: map raw tuple values → discrete labels
        df = self._discretize(df.copy())
        if df.empty:
            logger.info("[%s] apply() end | post-discretize=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # 2) Abstract: apply abstraction rules to discrete tuples
        df = self._abstract(df)
        if df.empty:
            logger.info("[%s] apply() end | post-abstract=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # 3) Merge: concatenate adjacent identical states using persistence + interpolation
        df = self._merge_intervals(df)

        df["ConceptName"] = self.name
        df["AbstractionType"] = self.family
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(df))
        return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw tuple values to discrete labels per attribute index."""
        if not self.discretization_rules:
            # No discretization needed (nominal or already discrete)
            return df

        def discretize_tuple(raw_tup: Tuple[Any, ...]) -> Optional[Tuple[str, ...]]:
            """Apply discretization rules to each tuple element."""
            discrete = []
            for idx, raw_val in enumerate(raw_tup):
                # Skip None values (no rule can match None)
                if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                    discrete.append(None)
                    continue

                matched = None
                for rule in self.discretization_rules:
                    if rule.attr_idx == idx and rule.matches(raw_val):
                        matched = rule.value
                        break
                
                # If idx has discretization rules but no match → drop row
                has_rules_for_idx = any(r.attr_idx == idx for r in self.discretization_rules)
                if has_rules_for_idx and matched is None:
                    return None  # no match → filter out
                
                discrete.append(matched if matched else raw_val)  # keep raw if no rules for idx
            
            return tuple(discrete)

        df["Value"] = df["Value"].apply(discretize_tuple)
        df = df[df["Value"].notna()]  # drop None (no-match) rows
        return df

    def _abstract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply abstraction rules to discrete tuples → final state labels."""
        if not self.abstraction_rules:
            # No abstraction rules → use discrete tuple as-is (convert to string)
            df["Value"] = df["Value"].apply(lambda t: str(t) if isinstance(t, tuple) else t)
            return df

        def apply_rules_to_tuple(discrete_tup: Tuple[str, ...]) -> Optional[Union[str, List[str]]]:
            """Match abstraction rules and return value(s)."""
            matches = [r.value for r in self.abstraction_rules if r.matches(discrete_tup)]
            if not matches:
                return None  # no match → filter out
            
            if self.abstraction_order == "first":
                return matches[0]  # single string
            else:  # "all"
                return matches  # list of strings

        df["Value"] = df["Value"].apply(apply_rules_to_tuple)
        df = df[df["Value"].notna()]
        return df

    def _merge_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge adjacent identical state values using persistence windows + interpolation.
        Each interval extends to last_merged_sample_time + good_after (or next sample's start, whichever is earlier).
        """
        if df.empty:
            return df

        df = df.sort_values("StartDateTime").reset_index(drop=True)

        # If order="all", explode multi-value rows and group by value
        if self.abstraction_order == "all":
            exploded_rows = []
            for _, row in df.iterrows():
                val = row["Value"]
                if isinstance(val, list):
                    for v in val:
                        exploded_rows.append({
                            "PatientId": row["PatientId"],
                            "ConceptName": row["ConceptName"],
                            "StartDateTime": row["StartDateTime"],
                            "EndDateTime": row["EndDateTime"],
                            "Value": v,
                            "AbstractionType": row["AbstractionType"]
                        })
                else:
                    exploded_rows.append(row.to_dict())
            df = pd.DataFrame(exploded_rows)
            
            # CORRECTED: Process each unique Value independently
            all_merged = []
            for value in df["Value"].unique():
                value_df = df[df["Value"] == value].sort_values("StartDateTime").reset_index(drop=True)
                merged_for_value = self._merge_single_value_group(value_df)
                all_merged.extend(merged_for_value)
            
            out = pd.DataFrame(all_merged) if all_merged else pd.DataFrame(columns=df.columns)
            return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]
        else:
            # order="first": standard merging
            merged = self._merge_single_value_group(df)
            out = pd.DataFrame(merged) if merged else pd.DataFrame(columns=df.columns)
            return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _merge_single_value_group(self, df: pd.DataFrame) -> List[dict]:
        """
        Merge intervals for a single value group (all rows have same Value).
        Returns list of merged interval dicts.
        """
        merged: List[dict] = []
        n = len(df)
        i = 0
        
        # OPTIMIZATION: Pre-convert StartDateTime to numpy array (avoid repeated pd.Timestamp calls)
        start_times = df["StartDateTime"].values  # numpy datetime64 array
        good_after_td = pd.Timedelta(self.good_after)  # convert once
        
        while i < n:
            current_row = df.iloc[i]
            interval_start = start_times[i]  # numpy datetime64
            interval_value = current_row["Value"]
            interval_pid = current_row["PatientId"]
            interval_cname = current_row["ConceptName"]
            interval_abstype = current_row["AbstractionType"]
            
            # Collect all rows that merge into this interval
            merged_rows = [i]
            skip_count = 0
            j = i + 1
            
            # OPTIMIZATION: Compute window boundary once
            window_end = interval_start + good_after_td
            
            while j < n:
                next_start = start_times[j]
                next_value = df.iloc[j]["Value"]
                same_value = (next_value == interval_value)
                within_window = (next_start <= window_end)
                
                if same_value and within_window:
                    merged_rows.append(j)
                    skip_count = 0
                    j += 1
                elif not same_value and within_window and self.interpolate and skip_count < self.max_skip:
                    # Check if next row after this one is same_value and within_window
                    if j + 1 < n:
                        peek_start = start_times[j + 1]
                        peek_value = df.iloc[j + 1]["Value"]
                        peek_same = (peek_value == interval_value)
                        peek_within = (peek_start <= window_end)
                        if peek_same and peek_within:
                            skip_count += 1
                            j += 1
                            continue
                    break
                else:
                    break
            
            # Determine interval EndDateTime
            last_merged_time = start_times[merged_rows[-1]]
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
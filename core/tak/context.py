from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, get_tak_repository, EventAbstractionRule
from .raw_concept import RawConcept
from .utils import parse_duration


class Context(TAK):
    """
    Context abstraction: background facts with interval windowing and clipping.
    - Derived from one or more raw-concepts (like Event)
    - Applies context window (before/after) — can vary by abstraction value
    - Supports clippers (external raw-concepts that trim interval boundaries)
    - No interval merging (unlike State)
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        abstraction_rules: List[EventAbstractionRule],
        context_windows: Dict[Optional[str], Dict[str, timedelta]],  # {value: {before, after}}
        clippers: List[Dict[str, Any]],
    ):
        super().__init__(name=name, categories=categories, description=description, family="context")
        self.derived_from = derived_from
        self.abstraction_rules = abstraction_rules
        self.context_windows = context_windows  # {value: {'before': td, 'after': td}, None: default}
        self.clippers = clippers

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Context":
        """Parse <context> XML with structural validation."""
        root = ET.parse(xml_path).getroot()
        if root.tag != "context":
            raise ValueError(f"{xml_path} is not a context file")

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
            
            tak_type = attr_el.attrib["tak"]
            if tak_type != "raw-concept":
                raise ValueError(f"{name}: <derived-from><attribute tak='{tak_type}'> invalid. Contexts can only be derived from 'raw-concept'.")
            
            derived_from.append({
                "name": attr_el.attrib["name"],
                "tak_type": tak_type,
                "idx": int(attr_el.attrib.get("idx", 0))
            })
        
        if not derived_from:
            raise ValueError(f"{name}: <derived-from> must contain at least one <attribute>")

        # --- context-window(s) (required) ---
        # NEW: Support multiple <persistence> blocks with optional value="..." attribute
        windows_el = root.find("context-windows")
        if windows_el is None:
            raise ValueError(f"{name}: missing <context-windows> block")
        
        context_windows: Dict[Optional[str], Dict[str, timedelta]] = {}
        for pers_el in windows_el.findall("persistence"):
            value_attr = pers_el.attrib.get("value")  # None = default window
            before_str = pers_el.attrib.get("good-before", "0h")
            after_str = pers_el.attrib.get("good-after", "0h")
            
            context_windows[value_attr] = {
                "before": parse_duration(before_str),
                "after": parse_duration(after_str)
            }
        
        if not context_windows:
            raise ValueError(f"{name}: <context-windows> must contain at least one <persistence> block")

        # --- clippers (optional) ---
        clippers = []
        clippers_el = root.find("clippers")
        if clippers_el is not None:
            for clip_el in clippers_el.findall("clipper"):
                if "name" not in clip_el.attrib:
                    raise ValueError(f"{name}: <clipper> must have 'name' attribute")
                
                clip_spec = {"name": clip_el.attrib["name"]}
                
                # Parse clip-before and clip-after (optional)
                if "clip-before" in clip_el.attrib:
                    clip_spec["clip_before"] = parse_duration(clip_el.attrib["clip-before"])
                if "clip-after" in clip_el.attrib:
                    clip_spec["clip_after"] = parse_duration(clip_el.attrib["clip-after"])
                
                # At least one of clip-before or clip-after must be specified
                if "clip_before" not in clip_spec and "clip_after" not in clip_spec:
                    raise ValueError(f"{name}: clipper '{clip_spec['name']}' must have at least one of 'clip-before' or 'clip-after'")
                
                clippers.append(clip_spec)

        # --- abstraction-rules (optional) ---
        abs_rules: List[EventAbstractionRule] = []
        abs_el = root.find("abstraction-rules")
        if abs_el is not None:
            for rule_el in abs_el.findall("rule"):
                val = rule_el.attrib["value"]
                op = rule_el.attrib.get("operator", "or")
                constraints: Dict[str, List[Dict[str, Any]]] = {}
                
                for attr_el in rule_el.findall("attribute"):
                    attr_name = attr_el.attrib["name"]
                    attr_idx = int(attr_el.attrib.get("idx", 0))
                    allowed = []
                    
                    for av in attr_el.findall("allowed-value"):
                        constraint = {"idx": attr_idx}
                        has_equal = "equal" in av.attrib
                        has_min = "min" in av.attrib
                        has_max = "max" in av.attrib
                        
                        if has_equal and (has_min or has_max):
                            raise ValueError(f"{name}: <allowed-value> cannot have both 'equal' and 'min'/'max' attributes")
                        if not (has_equal or has_min or has_max):
                            raise ValueError(f"{name}: <allowed-value> must have at least one of: 'equal', 'min', 'max'")
                        
                        if has_equal:
                            constraint["type"] = "equal"
                            constraint["value"] = av.attrib["equal"]
                        elif has_min and has_max:
                            constraint["type"] = "range"
                            constraint["min"] = float(av.attrib["min"])
                            constraint["max"] = float(av.attrib["max"])
                        elif has_min:
                            constraint["type"] = "min"
                            constraint["value"] = float(av.attrib["min"])
                        elif has_max:
                            constraint["type"] = "max"
                            constraint["value"] = float(av.attrib["max"])
                        
                        allowed.append(constraint)
                    
                    constraints[attr_name] = allowed
                
                abs_rules.append(EventAbstractionRule(val, op, constraints))

        context = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            abstraction_rules=abs_rules,
            context_windows=context_windows,
            clippers=clippers,
        )
        context.validate()
        return context

    def validate(self) -> None:
        """Business logic validation using global TAKRepository."""
        repo = get_tak_repository()

        # 1) Check all derived_from TAKs exist and are RawConcepts
        for df in self.derived_from:
            parent_tak = repo.get(df["name"])
            if parent_tak is None:
                raise ValueError(f"{self.name}: derived_from='{df['name']}' not found in TAK repository")
            if not isinstance(parent_tak, RawConcept):
                raise ValueError(f"{self.name}: derived_from='{df['name']}' is not a RawConcept (found {parent_tak.family})")

        # 2) Check clippers exist and are RawConcepts
        for clipper in self.clippers:
            clipper_tak = repo.get(clipper["name"])
            if clipper_tak is None:
                raise ValueError(f"{self.name}: clipper='{clipper['name']}' not found in TAK repository")
            if not isinstance(clipper_tak, RawConcept):
                raise ValueError(f"{self.name}: clipper='{clipper['name']}' is not a RawConcept (found {clipper_tak.family})")

        # 3) Validate abstraction rules (same as Event)
        if self.abstraction_rules:
            for rule in self.abstraction_rules:
                if rule.operator == "and":
                    attr_sources = set()
                    for attr_name in rule.constraints.keys():
                        matching_df = next((df for df in self.derived_from if df["name"] == attr_name), None)
                        if matching_df is None:
                            raise ValueError(f"{self.name}: rule references unknown attribute '{attr_name}'")
                        attr_sources.add(matching_df["name"])
                    if len(attr_sources) > 1:
                        raise ValueError(f"{self.name}: operator='and' requires all attributes from same source (found: {attr_sources})")

                for attr_name, constraints in rule.constraints.items():
                    matching_df = next((df for df in self.derived_from if df["name"] == attr_name), None)
                    if matching_df is None:
                        continue
                    parent_tak = repo.get(matching_df["name"])
                    if isinstance(parent_tak, RawConcept):
                        if parent_tak.concept_type == "raw":
                            attr_idx = matching_df["idx"]
                            if attr_idx >= len(parent_tak.tuple_order):
                                raise ValueError(f"{self.name}: idx={attr_idx} out of bounds for '{parent_tak.name}'")
                            attr_name_in_parent = parent_tak.tuple_order[attr_idx]
                            parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name_in_parent), None)
                        else:
                            parent_attr = parent_tak.attributes[0] if parent_tak.attributes else None
                        
                        if parent_attr:
                            for c in constraints:
                                if parent_attr["type"] == "nominal":
                                    if c["type"] == "equal" and c["value"] not in parent_attr.get("allowed", []):
                                        raise ValueError(f"{self.name}: constraint value '{c['value']}' not in allowed values for '{attr_name}'")
                                elif parent_attr["type"] == "boolean":
                                    if c["type"] == "equal" and c["value"] not in ("True", "False"):
                                        raise ValueError(f"{self.name}: boolean constraint must be 'True' or 'False'")

        # UPDATED: Validate context windows vs abstraction rules (bidirectional) - RAISE ERRORS
        if self.abstraction_rules:
            rule_values = {rule.value for rule in self.abstraction_rules}
            window_values = {v for v in self.context_windows.keys() if v is not None}  # exclude default (None)
            
            # Check 1: Windows defined for non-existent rule values (typo detection) - RAISE
            for window_val in window_values:
                if window_val not in rule_values:
                    raise ValueError(f"{self.name}: context-window for value='{window_val}' does not match any abstraction rule value (possible typo)")
            
            # Check 2: Rules without value-specific windows AND no default - RAISE
            default_window_exists = None in self.context_windows
            for rule_val in rule_values:
                if rule_val not in window_values and not default_window_exists:
                    raise ValueError(f"{self.name}: abstraction rule value='{rule_val}' has no value-specific window and no default window defined")

    def apply(self, df: pd.DataFrame, clipper_dfs: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply context abstraction to input data (from one or more derived-from raw-concepts).
        Input:  PatientId, ConceptName(any derived-from), StartDateTime, EndDateTime, Value(tuple), AbstractionType
        Output: PatientId, ConceptName(self.name), StartDateTime, EndDateTime(windowed+clipped), Value(context_label), AbstractionType
        
        Args:
            df: Input DataFrame from raw-concepts
            clipper_dfs: Dict of {clipper_name: DataFrame} for clipping boundaries (optional)
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

        # If no abstraction rules: emit raw values as-is (with windowing using default window)
        if not self.abstraction_rules:
            df = self._apply_context_window(df)
            df = self._apply_clippers(df, clipper_dfs)
            df["ConceptName"] = self.name
            df["AbstractionType"] = self.family
            logger.info("[%s] apply() end (no rules) | output_rows=%d", self.name, len(df))
            return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

        # Apply abstraction rules
        df = self._abstract(df)
        df = self._apply_context_window(df)  # Now uses per-value windows
        df = self._apply_clippers(df, clipper_dfs)
        df["ConceptName"] = self.name
        df["AbstractionType"] = self.family

        logger.info("[%s] apply() end | output_rows=%d", self.name, len(df))
        return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _abstract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply abstraction rules to input tuples → context labels."""
        def apply_rules_to_row(row) -> Optional[str]:
            for rule in self.abstraction_rules:
                if rule.matches(row, self.derived_from):
                    return rule.value
            return None

        df["Value"] = df.apply(apply_rules_to_row, axis=1)
        df = df[df["Value"].notna()]
        return df

    def _apply_context_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extend intervals using context window (before/after).
        OPTIMIZED: Simple vectorized implementation (group by Value).
        """
        if df.empty:
            return df
        
        # Convert timestamps once (avoid repeated conversions)
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"])
        
        # For each unique value, apply its window
        for value in df["Value"].unique():
            # Look up window for this value (value-specific first, then default)
            window = self.context_windows.get(value) or self.context_windows.get(None)
            
            if window is None:
                # No window defined → RAISE ERROR (shouldn't happen after validation, but safety check)
                raise ValueError(f"[{self.name}] No context window defined for value='{value}' (validation should have caught this)")
            
            # Apply window to all rows with this value (vectorized)
            mask = (df["Value"] == value)
            df.loc[mask, "StartDateTime"] = df.loc[mask, "StartDateTime"] - pd.Timedelta(window["before"])
            df.loc[mask, "EndDateTime"] = df.loc[mask, "EndDateTime"] + pd.Timedelta(window["after"])
        
        return df

    def _apply_clippers(self, df: pd.DataFrame, clipper_dfs: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clip intervals using clipper boundaries.
        SIMPLIFIED: Readable row-by-row implementation with basic optimizations.
        """
        if df.empty or not self.clippers or clipper_dfs is None:
            return df

        # Convert timestamps once
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"])

        # Track indices to drop
        indices_to_drop = set()

        for clipper_spec in self.clippers:
            clipper_name = clipper_spec["name"]
            clip_before = clipper_spec.get("clip_before")
            clip_after = clipper_spec.get("clip_after")
            
            if clipper_name not in clipper_dfs:
                logger.warning("[%s] Clipper '%s' not found in clipper_dfs (skipping)", self.name, clipper_name)
                continue
            
            clipper_df = clipper_dfs[clipper_name].copy()
            if clipper_df.empty:
                continue

            # Convert clipper timestamps
            clipper_df["StartDateTime"] = pd.to_datetime(clipper_df["StartDateTime"])
            clipper_df["EndDateTime"] = pd.to_datetime(clipper_df["EndDateTime"])

            # For each context row, find ALL overlapping clippers
            for idx, ctx_row in df.iterrows():
                pid = ctx_row["PatientId"]
                ctx_start = ctx_row["StartDateTime"]
                ctx_end = ctx_row["EndDateTime"]
                
                # Find overlapping clippers for this patient
                patient_clippers = clipper_df[clipper_df["PatientId"] == pid]
                if patient_clippers.empty:
                    continue

                overlaps = patient_clippers[
                    (patient_clippers["StartDateTime"] <= ctx_end) &
                    (patient_clippers["EndDateTime"] >= ctx_start)
                ]
                
                if overlaps.empty:
                    continue

                # Apply clipping for EACH overlapping clipper
                for _, clipper_row in overlaps.iterrows():
                    clipper_start = clipper_row["StartDateTime"]
                    clipper_end = clipper_row["EndDateTime"]

                    # Apply clip-before: trim context front if context starts before clipper
                    if clip_before is not None and ctx_start < clipper_start:
                        new_start = clipper_start + pd.Timedelta(clip_before)
                        ctx_start = max(ctx_start, new_start)

                    # Apply clip-after: delay context start if context overlaps clipper
                    if clip_after is not None and ctx_start < clipper_end:
                        delayed_start = clipper_end + pd.Timedelta(clip_after)
                        ctx_start = max(ctx_start, delayed_start)

                # Update df with clipped start time
                df.at[idx, "StartDateTime"] = ctx_start
                
                # Check if interval is still valid
                if ctx_start >= ctx_end:
                    indices_to_drop.add(idx)

        # Remove contexts with invalid intervals
        if indices_to_drop:
            df = df.drop(index=list(indices_to_drop))
            logger.info("[%s] Removed %d context intervals (flipped after clipping)", self.name, len(indices_to_drop))

        return df

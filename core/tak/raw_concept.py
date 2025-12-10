from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, validate_xml_against_schema
from .utils import apply_external_function
from .external_functions import REPO
from .repository import get_tak_repository


class RawConcept(TAK):
    """
    Raw Concept TAK — parsed from <raw-concept> XML.
    - 'raw': multi-attr; requires <tuple-order> and <merge require-all>
    - 'raw-numeric': one or more numeric attrs; ranges kept per-attr (min/max)
    - 'raw-nominal': one or more nominal attrs; each must have allowed values
    - 'raw-boolean': one or more boolean attrs
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        concept_type: str,  # "raw","raw-numeric","raw-nominal","raw-boolean"
        attributes: List[Dict[str, Any]],
        tuple_order: Tuple[str, ...],
        merge_require_all: bool = False,
    ):
        super().__init__(name=name, categories=categories, description=description, family="raw-concept")

        self.concept_type = concept_type
        self.attributes = attributes                    # [{name, type, min, max, allowed}]
        self.tuple_order = tuple_order                   # declared attribute order
        self.merge_require_all = merge_require_all      # for 'raw' concepts: whether to require all attrs in merge

        # Runtime patient-level dataframe (filled in apply())
        self.df: Optional[pd.DataFrame] = None

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "RawConcept":
        """Parse a <raw-concept> XML definition file."""
        xml_path = Path(xml_path)
        
        # Validate against XSD schema (graceful if lxml not available)
        validate_xml_against_schema(xml_path)
        
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
                    "allowed": None
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

                attributes.append(spec)

        # --- Tuple order ---
        tuple_order = ()
        tuple_el = root.find("tuple-order")
        # Collect only if concept_type is "raw"
        if tuple_el is not None and concept_type == "raw":
            tuple_order = tuple(f.attrib["name"] for f in tuple_el.findall("attribute"))

        # --- Merge (raw only) ---
        merge_el = root.find("merge")
        require_all: bool = False
        # Collect only if concept_type is "raw"
        if merge_el is not None and concept_type == "raw":
            require_all = (merge_el.attrib.get("require-all", "false").lower() == "true")
        
        # --- Type-specific structural checks ---
        if concept_type == "raw":
            if len(attributes) < 2:
                raise ValueError(f"{name}: raw concept must define ≥2 attributes")
            if tuple_el is None or not tuple_order:
                raise ValueError(f"{name}: raw concept must define <tuple-order> block")
            if merge_el is None:
                raise ValueError(f"{name}: raw concept must define <merge require-all=...> block")
        
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
        """
        Apply the TAK definition to patient-level raw records (1 patient only).
        Input:  PatientId, ConceptName, StartDateTime, EndDateTime, Value
        Output: PatientId, ConceptName(self.name), StartDateTime, EndDateTime, Value(tuple), AbstractionType
        """
        # Log input size
        total_in = len(df)
        logger.info("[%s] apply() start | input_rows=%d", self.name, total_in)

        # 0) Filter relevant concepts
        valid_concepts = {a["name"] for a in self.attributes}
        df = df[df["ConceptName"].isin(valid_concepts)]
        if df.empty:
            logger.info("[%s] apply() end | post-filter=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # 1) Normalize & filter values per-attribute type
        for a in self.attributes:
            name, typ = a["name"], a["type"]
            mask = (df["ConceptName"].to_numpy() == name)
            if not mask.any():
                continue

            if typ == "numeric":
                vals = pd.to_numeric(df.loc[mask, "Value"], errors="coerce")
                minv = a["min"] if a["min"] is not None else -float("inf")
                maxv = a["max"] if a["max"] is not None else  float("inf")
                keep = vals.between(minv, maxv, inclusive="both")
                idx_keep = df.loc[mask].index[keep]
                idx_drop = df.loc[mask].index.difference(idx_keep)
                if len(idx_drop):
                    df.drop(index=idx_drop, inplace=True)
                df.loc[idx_keep, "Value"] = vals.loc[keep].to_numpy()

            elif typ == "nominal":
                allowed = a.get("allowed") or ()
                if allowed:
                    allowed_set = set(allowed)
                    keep = df.loc[mask, "Value"].isin(allowed_set)
                    idx_keep = df.loc[mask].index[keep]
                    idx_drop = df.loc[mask].index.difference(idx_keep)
                    if len(idx_drop):
                        df.drop(index=idx_drop, inplace=True)

            elif typ == "boolean":
                # No validation; any present row will become ("True",)
                pass

        if df.empty:
            logger.info("[%s] apply() end | post-type-filter=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        pid = int(df["PatientId"].iloc[0])

        # 2) Concept-type-specific emission
        if self.concept_type != "raw":
            out = df.loc[:, ["PatientId","StartDateTime","EndDateTime","ConceptName","Value"]].copy()

            # boolean → ALWAYS stringify to "True" (never Python bool)
            if any(a["type"] == "boolean" for a in self.attributes):
                bool_names = {a["name"] for a in self.attributes if a["type"] == "boolean"}
                mask_bool = out["ConceptName"].isin(bool_names)
                if mask_bool.any():
                    # Ensure Value is ALWAYS string "True", not Python bool
                    out.loc[mask_bool, "Value"] = out.loc[mask_bool, "Value"].apply(
                        lambda v: "True" if str(v).lower() in ("true", "1", "yes") else str(v)
                    )
                    # Wrap in 1-tuple
                    out.loc[mask_bool, "Value"] = [("True",)] * int(mask_bool.sum())

            # wrap scalars into 1-tuples (avoid double-wrap)
            # Ensure all non-tuple values are stringified before wrapping
            out["Value"] = out["Value"].map(lambda v: v if isinstance(v, tuple) else (v,))

            out["ConceptName"] = self.name
            out["AbstractionType"] = self.family

            # SORT OUTPUT BY TIMESTAMP (critical for downstream TAKs)
            out = out.sort_values("StartDateTime").reset_index(drop=True)
            
            logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
            return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

        # 3) concept_type == "raw": merge tuples
        df = df.copy()
        
        # Stringify boolean attributes BEFORE grouping
        for a in self.attributes:
            if a["type"] == "boolean":
                mask = (df["ConceptName"] == a["name"])
                if mask.any():
                    df.loc[mask, "Value"] = df.loc[mask, "Value"].apply(
                        lambda v: "True" if str(v).lower() in ("true", "1", "yes") else str(v)
                    )
        
        # Group by exact timestamp (rounded to nearest microsecond to handle floating-point errors)
        df["timestamp_key"] = df["StartDateTime"].dt.round("us")
        
        merged: List[dict] = []
        
        for ts, group in df.groupby("timestamp_key"):
            # Build tuple from all attributes at this exact timestamp
            tuple_vals = []
            
            for attr_name in self.tuple_order:
                attr_rows = group[group["ConceptName"] == attr_name]
                if attr_rows.empty:
                    tuple_vals.append(None)
                else:
                    # Take last value if multiple rows for same attribute at same timestamp
                    tuple_vals.append(attr_rows.iloc[-1]["Value"])
            
            # Apply validation rules
            if all(v is None for v in tuple_vals):
                logger.debug("[RAW][%s] Dropping all-None tuple at %s", self.name, ts)
                continue
            
            if self.merge_require_all and any(v is None for v in tuple_vals):
                logger.debug("[RAW][%s] Dropping partial tuple (require_all=True) at %s", self.name, ts)
                continue
            
            # Use first non-None attribute's timestamps
            first_non_none_idx = next((i for i, v in enumerate(tuple_vals) if v is not None), None)
            if first_non_none_idx is not None:
                attr_name = self.tuple_order[first_non_none_idx]
                source_row = group[group["ConceptName"] == attr_name].iloc[-1]
                start_dt = source_row["StartDateTime"]
                end_dt = source_row["EndDateTime"]
            else:
                start_dt = pd.Timestamp(ts)
                end_dt = pd.Timestamp(ts)
            
            merged.append({
                "PatientId": pid,
                "ConceptName": self.name,
                "StartDateTime": start_dt,
                "EndDateTime": end_dt,
                "Value": tuple(tuple_vals),
                "AbstractionType": self.family,
            })
        
        out = pd.DataFrame.from_records(merged) if merged else pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
        
        logger.info("[%s][RAW] emitted_tuples=%d | input_rows=%d", self.name, len(out), total_in)
        
        out["ConceptName"] = self.name
        out["AbstractionType"] = self.family
        
        # SORT OUTPUT BY TIMESTAMP
        out = out.sort_values("StartDateTime").reset_index(drop=True)
        
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out))
        return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]


class ParameterizedRawConcept(RawConcept):
    """
    ParameterizedRawConcept: Like RawConcept, but applies a chain of external functions
    to the parent raw-concept's tuple, possibly using parameters.
    Used to create dynamic calculated raw concepts based on existing ones.
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: str,
        parent_idx_map: List[int],
        parameters: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        
        # The following are inherited from parent after parsing:
        concept_type: str,
        attributes: List[Dict[str, Any]],
        tuple_order: Tuple[str, ...],
        merge_require_all: bool,
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            concept_type=concept_type,
            attributes=attributes,
            tuple_order=tuple_order,
            merge_require_all=merge_require_all
        )
        self.derived_from = derived_from
        self.parent_idx_map = parent_idx_map
        self.parameters = parameters
        self.functions = functions

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "ParameterizedRawConcept":
        xml_path = Path(xml_path)
        validate_xml_against_schema(xml_path)
        root = ET.parse(xml_path).getroot()
        if root.tag != "parameterized-raw-concept":
            raise ValueError(f"{xml_path} is not a parameterized-raw-concept file")

        tak_name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # Parse derived-from (must be single raw-concept)
        df_el = root.find("derived-from")
        if df_el is None or "name" not in df_el.attrib or "tak" not in df_el.attrib:
            raise ValueError(f"{tak_name}: <derived-from> must have name and tak")
        parent_name = df_el.attrib["name"]
        if df_el.attrib["tak"] != "raw-concept":
            raise ValueError(f"{tak_name}: <derived-from> must point to a raw-concept")
        parent_idx_map = []
        if "idx" in df_el.attrib:
            parent_idx_map = [int(df_el.attrib["idx"])]
        else:
            parent_idx_map = None  # Will resolve after parent is loaded

        # Parse parameters (optional, idx optional, default required)
        parameters = []
        params_el = root.find("parameters")
        if params_el is not None:
            for param_el in params_el.findall("parameter"):
                param_name = param_el.attrib.get("name")
                how = param_el.attrib.get("how", "all")
                if param_name == parent_name and how == "all":
                    raise ValueError(f"{tak_name}: parameter cannot reference the parent TAK '{parent_name}' with how='all', this is temporal data leakage. Use how='before' instead.")
                param = {
                    "name": param_name,
                    "tak": param_el.attrib["tak"],
                    "idx": int(param_el.attrib["idx"]) if "idx" in param_el.attrib else 0,
                    "how": how,  # 'before' or 'all'
                    "dynamic": param_el.attrib.get("dynamic", "true").lower() == "true",
                    "ref": param_el.attrib["ref"],
                    "default": param_el.attrib.get("default")
                }
                parameters.append(param)

        # Parse functions (must exist, at least one, and value idx is required)
        functions_el = root.find("functions")
        if functions_el is None:
            raise ValueError(f"{tak_name}: <functions> block is required")
        functions = []
        for func_el in functions_el.findall("function"):
            value_el = func_el.find("value")
            if value_el is None or "idx" not in value_el.attrib:
                raise ValueError(f"{name}: each <function> must have a <value idx='...'> child")
            func = {
                "name": func_el.attrib["name"],
                "value_idx": int(value_el.attrib["idx"]),
                "param_refs": [p.attrib["ref"] for p in func_el.findall("parameter")]
            }
            functions.append(func)
        if not functions:
            raise ValueError(f"{tak_name}: must define at least one <function>")

        # Parent attributes/tuple_order/etc. are inherited after repo is loaded
        repo = get_tak_repository()
        parent = repo.get(parent_name)
        if not isinstance(parent, RawConcept):
            raise ValueError(f"{tak_name}: parent '{parent_name}' is not a RawConcept")
        if parent_idx_map is None:
            parent_idx_map = list(range(len(parent.tuple_order)))
        # Inherit all structure from parent
        concept_type = parent.concept_type
        attributes = parent.attributes
        tuple_order = parent.tuple_order
        merge_require_all = parent.merge_require_all

        obj = cls(
            name=tak_name,
            categories=cats,
            description=desc,
            derived_from=parent_name,
            parent_idx_map=parent_idx_map,
            parameters=parameters,
            functions=functions,
            concept_type=concept_type,
            attributes=attributes,
            tuple_order=tuple_order,
            merge_require_all=merge_require_all
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        """
        Validate that:
        - Parent exists and is a RawConcept
        - For each parameter: TAK exists in repo OR default is provided
        - All function names exist in REPO
        - All function value_idx are valid for parent tuple
        - Parameter idx is optional, default is required
        """
        repo = get_tak_repository()
        parent = repo.get(self.derived_from)
        if not isinstance(parent, RawConcept):
            raise ValueError(f"{self.name}: parent '{self.derived_from}' is not a RawConcept")

        # Use parent.attributes for tuple length (works for all raw concept types)
        parent_tuple_len = len(parent.attributes)

        for param in self.parameters:
            tak = repo.get(param["name"])
            if tak is None and param.get("default") is None:
                raise ValueError(
                    f"{self.name}: parameter TAK '{param['name']}' not found in TAK repository and no default provided"
                )
            if "idx" not in param:
                param["idx"] = 0

        for func in self.functions:
            func_name = func["name"]
            value_idx = func["value_idx"]
            if func_name not in REPO:
                raise ValueError(f"{self.name}: function '{func_name}' not found in external functions")
            if not (0 <= value_idx < parent_tuple_len):
                raise ValueError(f"{self.name}: function '{func_name}' value_idx={value_idx} out of bounds for parent tuple size {parent_tuple_len}")

    def _resolve_parameter_value(self, param: Dict[str, Any], ref_time: pd.Timestamp, df: pd.DataFrame) -> Any:
        """
        Helper: Resolve a single parameter value relative to a specific reference time.
        """
        param_rows = df[df["ConceptName"] == param["name"]]
        
        # 1. Filter by 'how' (temporal constraint)
        if not param_rows.empty:
            how = param.get("how", "all")
            if how == "before":
                param_rows = param_rows[param_rows["StartDateTime"] < ref_time]
        
        # 2. Find closest in time
        if not param_rows.empty:
            # Calculate absolute time distance
            # Note: We use .values to avoid index alignment issues during subtraction if indices differ
            dists = (param_rows["StartDateTime"] - ref_time).abs()
            closest_idx = dists.idxmin()
            
            val = param_rows.loc[closest_idx, "Value"]
            idx = param.get("idx", 0)
            if isinstance(val, tuple):
                val = val[idx]
            return val

        # 3. Fallback to default
        default_val = param.get("default")
        if default_val is not None:
            try:
                return float(default_val)
            except (TypeError, ValueError):
                return default_val
        
        return None # Unresolved

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply: expects input df to contain all parent raw-concept output rows (with tuple Value)
        and all parameter rows (if any). Applies function chain in-place on the tuple at the referenced idx.
        """
        repo = get_tak_repository()
        parent = repo.get(self.derived_from)
        parent_rows = df[df["ConceptName"] == parent.name]
        if parent_rows.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # Separate parameters by dynamic flag
        static_params = [p for p in self.parameters if not p["dynamic"]]
        dynamic_params = [p for p in self.parameters if p["dynamic"]]

        # Pre-calculate static parameters ONCE
        # Reference time for static params is the StartDateTime of the FIRST parent row
        # (Baseline logic: what was the state at the beginning of this concept's timeline?)
        static_values = {}
        if static_params:
            baseline_time = parent_rows.iloc[0]["StartDateTime"]
            for param in static_params:
                val = self._resolve_parameter_value(param, baseline_time, df)
                if val is None:
                    # If a static parameter cannot be resolved (and no default), 
                    # we might fail the whole concept or just skip rows. 
                    # Current logic: if any param is None, we skip the row.
                    # So we mark it as None here.
                    static_values[param["ref"]] = None
                else:
                    static_values[param["ref"]] = val

        out_rows = []
        for _, row in parent_rows.iterrows():
            tuple_val = list(row["Value"])  # mutable copy
            
            # Start with static values
            current_param_values = static_values.copy()
            
            # Resolve dynamic parameters for this specific row
            row_time = row["StartDateTime"]
            for param in dynamic_params:
                val = self._resolve_parameter_value(param, row_time, df)
                current_param_values[param["ref"]] = val

            # Check if any parameter failed to resolve
            if any(v is None for v in current_param_values.values()):
                continue

            # Apply functions
            for func in self.functions:
                func_name = func["name"]
                value_idx = func["value_idx"]
                param_refs = func["param_refs"]
                
                args = [tuple_val[value_idx]]
                for ref in param_refs:
                    args.append(current_param_values.get(ref))
                
                # Use apply_external_function instead of REPO directly
                result = apply_external_function(func_name, *args)
                tuple_val[value_idx] = result  # in-place update
            
            out_rows.append({
                "PatientId": row["PatientId"],
                "ConceptName": self.name,
                "StartDateTime": row["StartDateTime"],
                "EndDateTime": row["EndDateTime"],
                "Value": tuple(tuple_val),
                "AbstractionType": self.family
            })
        
        out = pd.DataFrame(out_rows)
        if out.empty:
             return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
        return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]
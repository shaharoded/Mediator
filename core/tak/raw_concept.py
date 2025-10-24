from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Literal, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .utils import parse_duration  # "8h" -> timedelta(hours=8)
from .tak import TAK


class RawConcept(TAK):
    """
    Raw Concept TAK — parsed from <raw-concept> XML.
    - 'raw'          : multi-attr; requires <tuple>/<order> and <merge tolerance require-all>
    - 'raw-numeric'  : one or more numeric attrs; all must be numeric; ranges kept per-attr (min/max) in self.attributes
    - 'raw-nominal'  : one or more nominal attrs; all must be nominal; each must have <nominal-allowed-values>
    - 'raw-boolean'  : one or more boolean attrs; all must be boolean
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        concept_type: Literal["raw","raw-numeric","raw-nominal","raw-boolean"],
        attributes: List[Dict[str, Any]],
        tuple_order: Tuple[str, ...],
        merge_tolerance: Optional[timedelta],
        merge_require_all: bool = False,
    ):
        super().__init__(name=name, categories=categories, description=description, family="raw-concept")

        self.concept_type = concept_type
        self.attributes = attributes                    # [{name, type, min, max, allowed}]
        self.tuple_order = tuple_order                   # declared attribute order
        self.merge_tolerance = merge_tolerance
        self.merge_require_all = merge_require_all      # for 'raw' concepts: whether to require all attrs in merge

        # Runtime patient-level dataframe (filled in apply())
        self.df: Optional[pd.DataFrame] = None

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "RawConcept":
        """Parse a <raw-concept> XML definition file."""
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
        tol: Optional[timedelta] = None
        require_all: bool = False
        # Collect only if concept_type is "raw"
        if merge_el is not None and concept_type == "raw":
            tol_val = merge_el.attrib.get("tolerance")
            if tol_val:
                tol = parse_duration(tol_val)
            require_all = (merge_el.attrib.get("require-all", "false").lower() == "true")
        
        # --- Type-specific structural checks ---
        if concept_type == "raw":
            if len(attributes) < 2:
                raise ValueError(f"{name}: raw concept must define ≥2 attributes")
            if tuple_el is None or not tuple_order:
                raise ValueError(f"{name}: raw concept must define <tuple-order> block")
            if merge_el is None or tol is None or require_all is None:
                raise ValueError(f"{name}: raw concept must define <merge tolerance=... require-all=...> block")
        
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
            merge_tolerance=tol,
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

    def _finalize_tuple(self, pid: int, block: pd.DataFrame) -> Optional[dict]:
        """
        Build a single merged tuple from a time-tolerance block (already filtered).
        - Uses the *last* value per ConceptName within the block
        - Respects require-all
        - Skips tuples that are entirely None
        - Emits warnings when the block composition is suspicious
        """
        # Last observed value per concept in the block
        last_vals = (
            block.sort_values("StartDateTime")
                .groupby("ConceptName", as_index=False)
                .last()
        )
        mapping = dict(zip(last_vals["ConceptName"].to_numpy(),
                        last_vals["Value"].to_numpy()))

        # Raw: sanity log if block composition != tuple order size
        uniq_concepts_in_block = set(last_vals["ConceptName"].to_numpy())
        if self.concept_type == "raw" and len(uniq_concepts_in_block) != len(self.tuple_order):
            logger.warning(
                "[RAW][%s] Window mismatch: uniq_concepts=%d, tuple_order=%d, require_all=%s",
                self.name, len(uniq_concepts_in_block), len(self.tuple_order), self.merge_require_all
            )

        # require-all enforcement (skip partials)
        if self.concept_type == "raw" and self.merge_require_all:
            if not all(name in mapping for name in self.tuple_order):
                logger.warning("[RAW][%s] Dropping partial tuple (require_all=True). Missing=%s",
                            self.name, [n for n in self.tuple_order if n not in mapping])
                return None

        # Build ordered tuple (None for missing if require_all=False)
        ordered = tuple(mapping.get(name) for name in self.tuple_order)

        # Skip tuple if it’s entirely None (no real data)
        if all(v is None for v in ordered):
            logger.warning("[RAW][%s] Dropping all-None tuple (likely misaligned inputs).", self.name)
            return None

        # If we allow partials and some entries are None → log for visibility
        if self.concept_type == "raw" and not self.merge_require_all:
            if any(v is None for v in ordered):
                logger.info("[RAW][%s] Emitting partial tuple (require_all=False). Missing=%s",
                            self.name, [n for n, v in zip(self.tuple_order, ordered) if v is None])

        start = block["StartDateTime"].min()
        end   = block["EndDateTime"].max()

        return {
            "PatientId": pid,
            "ConceptName": self.name,
            "StartDateTime": start,
            "EndDateTime": end,
            "Value": ordered,
            "AbstractionType": self.family,
        }

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
                # No validation; any present row will become (True,) later
                pass

        if df.empty:
            logger.info("[%s] apply() end | post-type-filter=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        pid = int(df["PatientId"].iloc[0])

        # 2) Concept-type-specific emission
        if self.concept_type != "raw":
            out = df.loc[:, ["PatientId","StartDateTime","EndDateTime","ConceptName","Value"]].copy()

            # boolean → set to (True,) only for those concepts
            if any(a["type"] == "boolean" for a in self.attributes):
                bool_names = {a["name"] for a in self.attributes if a["type"] == "boolean"}
                mask_bool = out["ConceptName"].isin(bool_names)
                if mask_bool.any():
                    out.loc[mask_bool, "Value"] = [(True,)] * int(mask_bool.sum())

            # wrap scalars into 1-tuples (avoid double-wrap)
            out["Value"] = out["Value"].map(lambda v: v if isinstance(v, tuple) else (v,))

            out["ConceptName"] = self.name
            out["AbstractionType"] = self.family

            # Log I/O sizes (non-raw)
            out_rows = len(out)
            if out_rows != len(df):
                logger.warning("[%s][NON-RAW] output_rows != input_rows | in=%d out=%d",
                            self.name, len(df), out_rows)
            else:
                logger.info("[%s][NON-RAW] emitted rows: %d", self.name, out_rows)

            return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

        # 3) raw → tolerance-window merge
        df.sort_values(["StartDateTime","ConceptName"], inplace=True)
        tol_s = self.merge_tolerance.total_seconds() if self.merge_tolerance is not None else 0.0

        merged: List[dict] = []
        start_idx = 0
        n = len(df)
        times = df["StartDateTime"].to_numpy()

        windows = 0
        for i in range(1, n + 1):
            if i == n or (times[i] - times[i-1]).total_seconds() > tol_s:
                block = df.iloc[start_idx:i]
                block = block[block["ConceptName"].isin(self.tuple_order)]
                if not block.empty:
                    windows += 1
                    row = self._finalize_tuple(pid, block)
                    if row is not None:
                        merged.append(row)
                start_idx = i

        out = (pd.DataFrame.from_records(merged)
            if merged else
            pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]))

        # Log I/O sizes (raw)
        logger.info("[%s][RAW] windows=%d | emitted_tuples=%d | input_rows=%d",
                    self.name, windows, len(out), n)

        # Additional consistency check: if windows produced where block size != len(tuple_order)
        # is handled inside _finalize_tuple via warnings.

        return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]
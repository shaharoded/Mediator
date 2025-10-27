from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Literal, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .utils import parse_duration
from .tak import TAK, get_tak_repository
from .raw_concept import RawConcept


class Trend(TAK):
    """
    Trend abstraction: compute local slopes over time-steady windows, build intervals using anchor logic.
    - Derived from one raw-concept with exactly 1 numeric attribute
    - Outputs: "Increasing", "Decreasing", "Steady" (hardcoded values)
    - Uses linear regression over time-steady window to compute slope
    - Intervals stretch backward to last anchor (point where previous interval ended)
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: str,  # raw-concept name
        attribute_idx: int,  # which attribute in tuple to track
        significant_variation: float,  # threshold for trend detection
        time_steady: timedelta,  # lookback window for slope calculation
        good_after: timedelta,  # maximum gap for anchor-based stretching
    ):
        super().__init__(name=name, categories=categories, description=description, family="trend")
        self.derived_from = derived_from
        self.attribute_idx = attribute_idx
        self.significant_variation = significant_variation
        self.time_steady = time_steady
        self.good_after = good_after

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Trend":
        """Parse <trend> XML with structural validation."""
        root = ET.parse(xml_path).getroot()
        if root.tag != "trend":
            raise ValueError(f"{xml_path} is not a trend file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required) ---
        df_el = root.find("derived-from")
        if df_el is None or "name" not in df_el.attrib or "tak" not in df_el.attrib:
            raise ValueError(f"{name}: missing <derived-from name='...' tak='raw-concept'>")
        
        if df_el.attrib["tak"] != "raw-concept":
            raise ValueError(f"{name}: Trends can only be derived from 'raw-concept' (not '{df_el.attrib['tak']}')")
        
        derived_from = df_el.attrib["name"]
        attribute_idx = int(df_el.attrib.get("idx", 0))
        significant_variation = float(df_el.attrib.get("significant-variation", 0))

        # --- time-steady (required) ---
        steady_el = root.find("time-steady")
        if steady_el is None or "value" not in steady_el.attrib:
            raise ValueError(f"{name}: missing <time-steady value='...'>")
        time_steady = parse_duration(steady_el.attrib["value"])

        # --- persistence (required) ---
        pers_el = root.find("persistence")
        if pers_el is None:
            raise ValueError(f"{name}: missing <persistence>")
        good_after = parse_duration(pers_el.attrib.get("good-after", "0h"))
        if good_after <= timedelta(0):
            raise ValueError(f"{name}: good-after must be positive")

        trend = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            attribute_idx=attribute_idx,
            significant_variation=significant_variation,
            time_steady=time_steady,
            good_after=good_after,
        )
        trend.validate()
        return trend

    def validate(self) -> None:
        """Business logic validation using global TAKRepository."""
        repo = get_tak_repository()

        # 1) Check derived_from exists and is RawConcept
        parent_tak = repo.get(self.derived_from)
        if parent_tak is None:
            raise ValueError(f"{self.name}: derived_from='{self.derived_from}' not found in TAK repository")
        if not isinstance(parent_tak, RawConcept):
            raise ValueError(f"{self.name}: derived_from='{self.derived_from}' is not a RawConcept (found {parent_tak.family})")

        # 2) Check attribute_idx is valid and numeric
        if parent_tak.concept_type == "raw":
            if self.attribute_idx >= len(parent_tak.tuple_order):
                raise ValueError(f"{self.name}: attribute_idx={self.attribute_idx} out of bounds (tuple size={len(parent_tak.tuple_order)})")
            attr_name = parent_tak.tuple_order[self.attribute_idx]
            parent_attr = next((a for a in parent_tak.attributes if a["name"] == attr_name), None)
        else:
            # raw-numeric/nominal/boolean → single attribute at idx=0
            if self.attribute_idx != 0:
                raise ValueError(f"{self.name}: derived_from='{self.derived_from}' is not 'raw', so attribute_idx must be 0")
            parent_attr = parent_tak.attributes[0] if parent_tak.attributes else None

        if parent_attr is None or parent_attr["type"] != "numeric":
            raise ValueError(f"{self.name}: attribute at idx={self.attribute_idx} is not numeric (required for trends)")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trend classification to raw-concept data.
        Input:  PatientId, ConceptName(=derived_from), StartDateTime, EndDateTime, Value(tuple), AbstractionType
        Output: PatientId, ConceptName(self.name), StartDateTime, EndDateTime, Value("Increasing"|"Decreasing"|"Steady"), AbstractionType
        """
        if df.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))

        # 1) Extract numeric values and compute trends
        df = self._compute_trends(df.copy())
        if df.empty:
            logger.info("[%s] apply() end | post-trend=0 rows", self.name)
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        # 2) Build anchor-based intervals
        df = self._build_intervals(df)

        df["ConceptName"] = self.name
        df["AbstractionType"] = self.family
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(df))
        return df[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

    def _compute_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each point using OLS slope over the entire lookback window (time_steady).
        The window includes the current point and all prior points within time_steady.
        """
        if df.empty:
            return df

        # Sort by time
        df = df.sort_values("StartDateTime").reset_index(drop=True)

        # Extract numeric values from tuple at attribute_idx (or pass-through if already scalar)
        def extract_value(val):
            if isinstance(val, tuple):
                return val[self.attribute_idx] if self.attribute_idx < len(val) else None
            return val

        df["numeric_value"] = pd.to_numeric(df["Value"].apply(extract_value), errors="coerce")
        df = df[df["numeric_value"].notna()].copy()
        if df.empty:
            return df

        # Ensure numpy-friendly dtypes
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        t = df["StartDateTime"].to_numpy(dtype="datetime64[ns]")
        y = df["numeric_value"].to_numpy(dtype=float)

        # Use timedelta64[ns] for proper window comparison
        win_delta = np.timedelta64(int(pd.Timedelta(self.time_steady).value), "ns")

        T = len(df)
        labels = []
        # Seconds from first point (monotonic increasing)
        t_sec = (t - t[0]) / np.timedelta64(1, "s")

        # For each i, include all j<=i with t_i - t_j <= time_steady
        for i in range(T):
            # find earliest index j_start within the window
            j = i
            while j >= 0 and (t[i] - t[j]) <= win_delta:
                j -= 1
            j_start = j + 1

            # Need at least two points in window to estimate a slope
            if i - j_start < 1:
                labels.append("Steady")
                continue

            xs = t_sec[j_start:i+1]
            ys = y[j_start:i+1]

            # OLS slope: cov(x,y)/var(x)
            x_mean = xs.mean()
            y_mean = ys.mean()
            dx = xs - x_mean
            dy = ys - y_mean
            denom = float((dx * dx).sum())
            if denom <= 0.0:
                labels.append("Steady")
                continue

            slope = float((dx * dy).sum()) / denom  # value per second
            horizon_sec = float(pd.Timedelta(self.time_steady) / pd.Timedelta(seconds=1))
            total_var = slope * horizon_sec

            if total_var >= self.significant_variation:
                labels.append("Increasing")
            elif total_var <= -self.significant_variation:
                labels.append("Decreasing")
            else:
                labels.append("Steady")

        # Overwrite Value with labels; keep schema
        df_out = df.loc[:, ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "AbstractionType"]].copy()
        df_out["Value"] = labels
        return df_out

    def _build_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build intervals using the anchor rule:
        - First point creates no interval; it sets the anchor.
        - Each next point can stretch back to the current anchor if (t_i - anchor) <= good_after.
        - If the gap exceeds good_after, emit a hole interval [anchor, t_i] with Value=None and reset anchor=t_i.
        - Labeled intervals use the CURRENT point's label and stretch [anchor, t_i].
        - Consecutive identical labels merge by extending EndDateTime.
        """
        if df.empty:
            return df

        # Ensure chronological order and numpy-friendly arrays
        df = df.sort_values("StartDateTime").reset_index(drop=True)
        times = pd.to_datetime(df["StartDateTime"]).to_numpy(dtype="datetime64[ns]")
        labels = df["Value"].to_numpy()
        pid_arr = df["PatientId"].to_numpy()
        cname_arr = df["ConceptName"].to_numpy()
        abst_arr = df["AbstractionType"].to_numpy()
        good_after_td = pd.Timedelta(self.good_after)

        merged: List[dict] = []
        n = len(df)

        # Initialize anchor with the first point (no interval yet)
        anchor_time = times[0]
        last_label = None  # label of the last labeled segment we emitted (None if none)
        # For merging: we extend the last emitted segment if same label & contiguous
        # (We always set anchor = current time after processing a point.)

        for i in range(1, n):
            t_i = times[i]
            lbl_i = labels[i]

            # If too far from current anchor → hole [anchor, t_i], then reset anchor and skip label emission now
            if (t_i - anchor_time) > np.timedelta64(int(good_after_td.value), 'ns'):
                merged.append({
                    "PatientId": int(pid_arr[i]),
                    "ConceptName": str(cname_arr[i]),
                    "StartDateTime": pd.Timestamp(anchor_time),
                    "EndDateTime": pd.Timestamp(t_i),
                    "Value": None,
                    "AbstractionType": str(abst_arr[i]),
                })
                anchor_time = t_i
                last_label = None
                continue

            # Within good_after → create/extend labeled segment with CURRENT label
            if merged and last_label == lbl_i and merged[-1]["Value"] == lbl_i and merged[-1]["EndDateTime"] == pd.Timestamp(anchor_time):
                # Extend last labeled segment to current time
                merged[-1]["EndDateTime"] = pd.Timestamp(t_i)
            else:
                # Start new labeled segment [anchor, t_i] with current label
                merged.append({
                    "PatientId": int(pid_arr[i]),
                    "ConceptName": str(cname_arr[i]),
                    "StartDateTime": pd.Timestamp(anchor_time),
                    "EndDateTime": pd.Timestamp(t_i),
                    "Value": str(lbl_i),
                    "AbstractionType": str(abst_arr[i]),
                })
            last_label = lbl_i
            anchor_time = t_i  # move anchor forward

        # Return in the same schema as apply() expects next
        if not merged:
            return pd.DataFrame(columns=df.columns)

        out = pd.DataFrame.from_records(merged)
        return out[["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"]]

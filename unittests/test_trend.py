"""
Comprehensive unit tests for Trend TAK.

Tests cover:
1. Parsing & validation (XML structure, numeric attribute requirement)
2. Slope calculation (increasing, decreasing, steady)
3. Anchor-based interval building (no holes except when anchor too far)
4. Edge cases (single point, trailing isolated points)
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from core.tak.trend import Trend
from core.tak.raw_concept import RawConcept
from core.tak.repository import set_tak_repository, TAKRepository


def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


def make_ts(hhmm: str, day: int = 0) -> datetime:
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


# XML Fixtures
RAW_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Glucose numeric measure</description>
  <attributes>
    <attribute name="GLUCOSE_LAB_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

TREND_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<trend name="GLUCOSE_MEASURE_TREND">
    <categories>Measurements</categories>
    <description>Trend for the measurements of GLUCOSE (Lab / Capillary)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" significant-variation="40"/>
    <time-steady value="12h"/>
    <persistence good-after="24h"/>
</trend>
"""


@pytest.fixture
def repo_with_glucose_trend(tmp_path: Path) -> TAKRepository:
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    trend_path = write_xml(tmp_path, "GLUCOSE_MEASURE_TREND.xml", TREND_GLUCOSE_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(Trend.parse(trend_path))
    return repo


def test_parse_trend_validates_structure(repo_with_glucose_trend):
    """Validate XML structure and instantiation."""
    trend = repo_with_glucose_trend.get("GLUCOSE_MEASURE_TREND")
    assert trend.name == "GLUCOSE_MEASURE_TREND"
    assert trend.derived_from == "GLUCOSE_MEASURE"
    assert trend.significant_variation == 40.0
    assert trend.time_steady == timedelta(hours=12)
    assert trend.good_after == timedelta(hours=24)


def test_trend_anchor_based_intervals(repo_with_glucose_trend):
    """Test anchor-based interval building with your example."""
    raw_tak = repo_with_glucose_trend.get("GLUCOSE_MEASURE")
    trend_tak = repo_with_glucose_trend.get("GLUCOSE_MEASURE_TREND")
    
    # ===== TEST CASE 1 =====
    print("\n" + "="*80)
    print("TEST CASE 1: T=0h(100), T=2h(150), T=4h(200), T=6h(210), T=31h(120), T=36h(130)")
    print("="*80)
    
    # Your example: T=0h, T=2h, T=4h, T=6h, T=31h, T=40h
    # time_steady=12h, significant_variation=40, good_after=24h
    df_raw = pd.DataFrame([
        (1, "GLUCOSE_LAB_MEASURE", make_ts("00:00"), make_ts("00:00"), 100),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("02:00"), make_ts("02:00"), 150),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("04:00"), make_ts("04:00"), 200),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("06:00"), make_ts("06:00"), 210),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("07:00", day=1), make_ts("07:00", day=1), 120),  # T=31h
        (1, "GLUCOSE_LAB_MEASURE", make_ts("12:00", day=1), make_ts("12:00", day=1), 130),  # T=36h
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_raw_out = raw_tak.apply(df_raw)
    
    # DEBUG: Show classifications
    df_classified = trend_tak._compute_trends(df_raw_out.copy())
    print("\nPoint-by-point classifications:")
    for _, row in df_classified.iterrows():
        print(f"  {row['StartDateTime'].strftime('%Y-%m-%d %H:%M')} → {row['Value']}")
    
    df_trend_out = trend_tak.apply(df_raw_out)
    
    print("\nOutput intervals (ACTUAL):")
    for idx, row in df_trend_out.iterrows():
        start = row['StartDateTime'].strftime('%Y-%m-%d %H:%M')
        end = row['EndDateTime'].strftime('%Y-%m-%d %H:%M')
        print(f"  Interval {idx}: [{start} → {end}] Value={row['Value']}")
    print(f"\nTotal intervals: {len(df_trend_out)}")
    print("="*80)
    
    # Expected output:
    # 1. [0h → 4h, "Increase"] (T=0h anchor, T=2h closes with "Increasing", T=4h extends with "Increasing")
    # 2. [4h → 6h, "Steady"] (T=6h closes with "Steady" the current anchor at T=4h)
    # 3. NO interval for 6h → 31h (anchor at T=6h, but 31h-6h=25h, outside window)
    # 4. [31h → 36h, "Steady"] (T=36h closes with "Steady" the current anchor at T=31h)

    
    assert len(df_trend_out) == 3

    assert df_trend_out.iloc[0]["Value"] == "Increasing"
    assert df_trend_out.iloc[0]["StartDateTime"] == make_ts("00:00")
    assert df_trend_out.iloc[0]["EndDateTime"] == make_ts("06:00")

    assert df_trend_out.iloc[1]["Value"] is None
    assert df_trend_out.iloc[1]["StartDateTime"] == make_ts("06:00")
    assert df_trend_out.iloc[1]["EndDateTime"] == make_ts("07:00", day=1)

    assert df_trend_out.iloc[2]["Value"] == "Steady"
    assert df_trend_out.iloc[2]["StartDateTime"] == make_ts("07:00", day=1)
    assert df_trend_out.iloc[2]["EndDateTime"] == make_ts("12:00", day=1)

    # ===== TEST CASE 2 =====
    print("\n" + "="*80)
    print("TEST CASE 2: T=0h(100), T=2h(150), T=4h(50), T=5h(30), T=26h(100), T=31h(120), T=36h(130)")
    print("="*80)
    
    # Your example: T=0h, T=2h, T=4h, T=6h, T=31h, T=40h
    # time_steady=12h, significant_variation=40, good_after=24h
    df_raw = pd.DataFrame([
        (1, "GLUCOSE_LAB_MEASURE", make_ts("00:00"), make_ts("00:00"), 100),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("02:00"), make_ts("02:00"), 150),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("04:00"), make_ts("04:00"), 50),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("05:00"), make_ts("05:00"), 30),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("02:00", day=1), make_ts("02:00", day=1), 100),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("07:00", day=1), make_ts("07:00", day=1), 120),  # T=31h
        (1, "GLUCOSE_LAB_MEASURE", make_ts("12:00", day=1), make_ts("12:00", day=1), 130),  # T=36h
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_raw_out = raw_tak.apply(df_raw)
    
    # DEBUG: Show classifications
    df_classified = trend_tak._compute_trends(df_raw_out.copy())
    print("\nPoint-by-point classifications:")
    for _, row in df_classified.iterrows():
        print(f"  {row['StartDateTime'].strftime('%Y-%m-%d %H:%M')} → {row['Value']}")
    
    df_trend_out = trend_tak.apply(df_raw_out)
    
    print("\nOutput intervals (ACTUAL):")
    for idx, row in df_trend_out.iterrows():
        start = row['StartDateTime'].strftime('%Y-%m-%d %H:%M')
        end = row['EndDateTime'].strftime('%Y-%m-%d %H:%M')
        print(f"  Interval {idx}: [{start} → {end}] Value={row['Value']}")
    print(f"\nTotal intervals: {len(df_trend_out)}")
    print("="*80)
    
    # Expected output:
    # Classification of anchors: Steady → Increasing → Decreasing → Decreasing → Increasing → Steady → Steady
    # 1. [0h → 2h, "Increasing"]
    # 2. [2h → 5h, "Decreasing"]
    # 3. [5h → 21h, "Increasing"]
    # 4. [26h → 31h, "Steady"]
    
    assert len(df_trend_out) == 5

    # 1
    assert df_trend_out.iloc[0]["Value"] == "Increasing"
    assert df_trend_out.iloc[0]["StartDateTime"] == make_ts("00:00")
    assert df_trend_out.iloc[0]["EndDateTime"] == make_ts("02:00")

    # 2
    assert df_trend_out.iloc[1]["Value"] == "Decreasing"
    assert df_trend_out.iloc[1]["StartDateTime"] == make_ts("02:00")
    assert df_trend_out.iloc[1]["EndDateTime"] == make_ts("05:00")

    # 3
    assert df_trend_out.iloc[2]["Value"] == "Steady"
    assert df_trend_out.iloc[2]["StartDateTime"] == make_ts("05:00")
    assert df_trend_out.iloc[2]["EndDateTime"] == make_ts("02:00", day=1)

    # 4
    assert df_trend_out.iloc[3]["Value"] == "Increasing"
    assert df_trend_out.iloc[3]["StartDateTime"] == make_ts("02:00", day=1)
    assert df_trend_out.iloc[3]["EndDateTime"] == make_ts("07:00", day=1)

    # 5
    assert df_trend_out.iloc[4]["Value"] == "Steady"
    assert df_trend_out.iloc[4]["StartDateTime"] == make_ts("07:00", day=1)
    assert df_trend_out.iloc[4]["EndDateTime"] == make_ts("12:00", day=1)
    # ===== TEST CASE 3 =====
    print("\n" + "="*80)
    print("TEST CASE 3: T=0h(100), T=2h(100), T=4h(150), T=6h(200), T=8h(210), T=33h(120), T=38h(130)")
    print("="*80)
    
    # new example starting from T=2h
    df_raw = pd.DataFrame([
        (1, "GLUCOSE_LAB_MEASURE", make_ts("00:00"), make_ts("00:00"), 100),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("02:00"), make_ts("02:00"), 100),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("04:00"), make_ts("04:00"), 150),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("06:00"), make_ts("06:00"), 200),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("08:00"), make_ts("08:00"), 210),  
        (1, "GLUCOSE_LAB_MEASURE", make_ts("09:00", day=1), make_ts("09:00", day=1), 120),  # T=31h
        (1, "GLUCOSE_LAB_MEASURE", make_ts("14:00", day=1), make_ts("14:00", day=1), 130),  # T=36h
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_raw_out = raw_tak.apply(df_raw)
    
    # DEBUG: Show classifications
    df_classified = trend_tak._compute_trends(df_raw_out.copy())
    print("\nPoint-by-point classifications:")
    for _, row in df_classified.iterrows():
        print(f"  {row['StartDateTime'].strftime('%Y-%m-%d %H:%M')} → {row['Value']}")
    
    df_trend_out = trend_tak.apply(df_raw_out)
    
    print("\nOutput intervals (ACTUAL):")
    for idx, row in df_trend_out.iterrows():
        start = row['StartDateTime'].strftime('%Y-%m-%d %H:%M')
        end = row['EndDateTime'].strftime('%Y-%m-%d %H:%M')
        print(f"  Interval {idx}: [{start} → {end}] Value={row['Value']}")
    print(f"\nTotal intervals: {len(df_trend_out)}")
    print("="*80)
    
    # Expected output:
    # 1. [0h → 2h, "Steady"]
    # 1. [2h → 6h, "Increase"] (T=0h anchor, T=2h closes with "Increasing", T=4h extends with "Increasing")
    # 2. [6h → 8h, "Steady"] (T=6h closes with "Steady" the current anchor at T=4h)
    # 3. NO interval for 8h → 33h (anchor at T=8h, but 33h-8h=25h, outside window)
    # 4. [33h → 39h, "Steady"] (T=39h closes with "Steady" the current anchor at T=33h)

    
    assert len(df_trend_out) == 4

    # 1
    assert df_trend_out.iloc[0]["Value"] == "Steady"
    assert df_trend_out.iloc[0]["StartDateTime"] == make_ts("00:00")
    assert df_trend_out.iloc[0]["EndDateTime"] == make_ts("02:00")

    # 2
    assert df_trend_out.iloc[1]["Value"] == "Increasing"
    assert df_trend_out.iloc[1]["StartDateTime"] == make_ts("02:00")
    assert df_trend_out.iloc[1]["EndDateTime"] == make_ts("08:00")

    # 3 (hole)
    assert df_trend_out.iloc[2]["Value"] is None
    assert df_trend_out.iloc[2]["StartDateTime"] == make_ts("08:00")
    assert df_trend_out.iloc[2]["EndDateTime"] == make_ts("09:00", day=1)

    # 4
    assert df_trend_out.iloc[3]["Value"] == "Steady"
    assert df_trend_out.iloc[3]["StartDateTime"] == make_ts("09:00", day=1)
    assert df_trend_out.iloc[3]["EndDateTime"] == make_ts("14:00", day=1)
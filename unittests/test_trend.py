"""
Comprehensive unit tests for Trend TAK.
"""

import pandas as pd
import pytest
from pathlib import Path
from datetime import timedelta

from core.tak.trend import Trend
from core.tak.raw_concept import RawConcept
from core.tak.repository import set_tak_repository, TAKRepository
from unittests.test_utils import write_xml, make_ts

# -----------------------------
# XML Fixtures (SELF-CONTAINED)
# -----------------------------

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

# FIXED: significant-variation is an ATTRIBUTE on <derived-from>, not a separate element
TREND_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<trend name="GLUCOSE_MEASURE_TREND">
    <categories>Measurements</categories>
    <description>Glucose trend detection</description>
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


@pytest.mark.parametrize("changed_value,label", [(200, "Increasing"), (50, "Decreasing")])
@pytest.mark.parametrize("following_hour", [None, 64, 74, 75])
def test_direction_requires_following_measurement(repo_with_glucose_trend, changed_value, label, following_hour):
    trend = repo_with_glucose_trend.get("GLUCOSE_MEASURE_TREND")
    hours = [44, 46, 50]
    values = [100, 100, changed_value]
    if following_hour is not None:
        hours.append(following_hour)
        values.append(100)
    data = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("00:00") + timedelta(hours=hour),
         make_ts("00:00") + timedelta(hours=hour), value, "raw-concept")
        for hour, value in zip(hours, values)
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])

    classified = trend._compute_trends(data)
    expected_labels = ["Steady", "Steady", label]
    if following_hour is not None:
        expected_labels.append("Steady")
    assert classified["Value"].tolist() == expected_labels

    cutoff = make_ts("00:00", day=2)
    prefix = data[data["StartDateTime"] <= cutoff]
    pd.testing.assert_frame_equal(
        classified.iloc[:len(prefix)].reset_index(drop=True), trend._compute_trends(prefix),
    )
    result = trend.apply(data)
    supported = following_hour is not None and following_hour - 50 <= 24
    assert result["Value"].tolist() == (["Steady", label] if supported else ["Steady"])
    assert result.iloc[0]["StartDateTime"] == make_ts("20:00", day=1)
    assert result.iloc[0]["EndDateTime"] == make_ts("02:00", day=2)
    if supported:
        assert result.iloc[1]["StartDateTime"] == make_ts("02:00", day=2)
        assert result.iloc[1]["EndDateTime"] == make_ts("00:00") + timedelta(hours=following_hour)
    assert (result["EndDateTime"] > result["StartDateTime"]).all()
    assert trend.apply(prefix)["Value"].tolist() == ["Steady"]


@pytest.mark.parametrize("labels", [["Increasing"], ["Decreasing"], ["Increasing", "Decreasing"]])
def test_no_interval_without_later_timestamp(repo_with_glucose_trend, labels):
    trend = repo_with_glucose_trend.get("GLUCOSE_MEASURE_TREND")
    data = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("00:00"), make_ts("00:00"), label, "raw-concept")
        for label in labels
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    assert trend._build_intervals(data).empty


def test_trend_forward_intervals(repo_with_glucose_trend):
    """Test forward classification with the original regression and gap examples."""
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
    
    expected = [[0,2,"Steady"],
                [2,6,"Increasing"],
                [31,36,"Steady"]]
    assert list(df_trend_out[["StartDateTime", "EndDateTime", "Value"]].itertuples(index=False, name=None)) == [
        (make_ts("00:00") + timedelta(hours=start),
         make_ts("00:00") + timedelta(hours=end), label)
        for start, end, label in expected
    ]

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
    
    expected = [[0,2,"Steady"],
                [2,4,"Increasing"],
                [4,26,"Decreasing"],
                [26,31,"Steady"],
                [31,36,"Increasing"]]
    assert list(df_trend_out[["StartDateTime", "EndDateTime", "Value"]].itertuples(index=False, name=None)) == [
        (make_ts("00:00") + timedelta(hours=start),
         make_ts("00:00") + timedelta(hours=end), label)
        for start, end, label in expected
    ]
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
    
    expected = [[0,4,"Steady"],
                [4,8,"Increasing"],
                [33,38,"Steady"]]
    assert list(df_trend_out[["StartDateTime", "EndDateTime", "Value"]].itertuples(index=False, name=None)) == [
        (make_ts("00:00") + timedelta(hours=start),
         make_ts("00:00") + timedelta(hours=end), label)
        for start, end, label in expected
    ]

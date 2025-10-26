"""
Comprehensive unit tests for State TAK.

Tests cover:
1. Parsing & validation (XML structure, business logic)
2. Discretization (numeric ranges, gaps, overlaps)
3. Abstraction (first vs all, tuple matching, coverage)
4. Merging (multi-point concatenation, good_before/after, interpolation, max_skip)
5. Edge cases (single-point states, order="all" overlapping intervals, permissive rules)
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Imports from your codebase
from core.tak.state import State
from core.tak.raw_concept import RawConcept
from core.tak.tak import set_tak_repository, TAKRepository
from core.tak.utils import parse_duration


# -----------------------------
# Helpers: XML writers & timestamp builders
# -----------------------------
def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    """Write XML to disk and return path."""
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


def make_ts(hhmm: str, day: int = 0) -> datetime:
    """Build timestamp: 2024-01-01 + day offset + HH:MM."""
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


# -----------------------------
# XML Fixtures (ACTUAL COMPLETE TAKs FROM KNOWLEDGE BASE)
# -----------------------------

# Parent raw-concept: BASAL_BITZUA (tuple: dosage + route)
RAW_BASAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASAL_BITZUA" concept-type="raw">
  <categories>Medications</categories>
  <description>Raw concept to manage the administration of BASAL insulin</description>
  <attributes>
    <attribute name="BASAL_DOSAGE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
    <attribute name="BASAL_ROUTE" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="SubCutaneous"/>
        <allowed-value value="IntraVenous"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
  <tuple-order>
    <attribute name="BASAL_DOSAGE"/>
    <attribute name="BASAL_ROUTE"/>
  </tuple-order>
  <merge tolerance="15m" require-all="false"/>
</raw-concept>
"""

# State: BASAL_BITZUA_STATE (ACTUAL COMPLETE VERSION)
STATE_BASAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="BASAL_BITZUA_STATE">
    <categories>Medications</categories>
    <description>Abstraction for the dosage and route of administration of BASAL insulin</description>
    <derived-from name="BASAL_BITZUA" tak="raw-concept"/>
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    <discretization-rules>
        <attribute idx="0">
            <rule value="Very Low" min="0" max="10"/>
            <rule value="Low" min="10" max="20"/>
            <rule value="Medium" min="20" max="40"/>
            <rule value="High" min="40" max="60"/>
            <rule value="Very High" min="60"/>
        </attribute>
    </discretization-rules>
    <abstraction-rules order="first">
        <rule value="SubCutaneous Low" operator="and">
            <attribute idx="0">
                <allowed-value value="Very Low"/>
                <allowed-value value="Low"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous Low" operator="and">
            <attribute idx="0">
                <allowed-value value="Very Low"/>
                <allowed-value value="Low"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="SubCutaneous Medium" operator="and">
            <attribute idx="0">
                <allowed-value value="Medium"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous Medium" operator="and">
            <attribute idx="0">
                <allowed-value value="Medium"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="SubCutaneous High" operator="and">
            <attribute idx="0">
                <allowed-value value="High"/>
                <allowed-value value="Very High"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous High" operator="and">
            <attribute idx="0">
                <allowed-value value="High"/>
                <allowed-value value="Very High"/>
            </attribute>
            <attribute idx="1">
                <allowed-value value="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="Low" operator="and">
            <attribute idx="0">
                <allowed-value value="Very Low"/>
                <allowed-value value="Low"/>
            </attribute>
        </rule>
        <rule value="Medium" operator="and">
            <attribute idx="0">
                <allowed-value value="Medium"/>
            </attribute>
        </rule>
        <rule value="High" operator="and">
            <attribute idx="0">
                <allowed-value value="High"/>
                <allowed-value value="Very High"/>
            </attribute>
        </rule>
    </abstraction-rules>
</state>
"""

# State with order="all" for testing multi-rule matching
STATE_BASAL_ALL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="BASAL_ALL_STATE">
  <categories>Medications</categories>
  <description>Order=all test</description>
  <derived-from name="BASAL_BITZUA" tak="raw-concept"/>
  <persistence good-after="6h" interpolate="false" max-skip="0"/>
  <discretization-rules>
    <attribute idx="0">
      <rule value="Low" min="0" max="30"/>
      <rule value="High" min="30" max="100"/>
    </attribute>
  </discretization-rules>
  <abstraction-rules order="all">
    <rule value="SubCutaneous" operator="and">
      <attribute idx="1">
        <allowed-value value="SubCutaneous"/>
      </attribute>
    </rule>
    <rule value="Low Dose" operator="and">
      <attribute idx="0">
        <allowed-value value="Low"/>
      </attribute>
    </rule>
  </abstraction-rules>
</state>
"""

# State with permissive rules (partial tuple matching)
STATE_BASAL_PERMISSIVE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="BASAL_PERMISSIVE_STATE">
  <categories>Medications</categories>
  <description>Order=all with permissive rules</description>
  <derived-from name="BASAL_BITZUA" tak="raw-concept"/>
  <persistence good-after="6h" interpolate="false" max-skip="0"/>
  <discretization-rules>
    <attribute idx="0">
      <rule value="Low" min="0" max="30"/>
      <rule value="High" min="30" max="100"/>
    </attribute>
  </discretization-rules>
  <abstraction-rules order="all">
    <rule value="SubCutaneous Low" operator="and">
      <attribute idx="0">
        <allowed-value value="Low"/>
      </attribute>
      <attribute idx="1">
        <allowed-value value="SubCutaneous"/>
      </attribute>
    </rule>
    <rule value="Low Dose" operator="and">
      <attribute idx="0">
        <allowed-value value="Low"/>
      </attribute>
    </rule>
  </abstraction-rules>
</state>
"""

# Raw-numeric parent (ACTUAL COMPLETE VERSION)
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

# State: GLUCOSE_MEASURE_STATE (ACTUAL COMPLETE VERSION)
STATE_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Abstraction for the measurements of GLUCOSE (Lab / Capillary)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    <discretization-rules>
        <attribute idx="0">
            <rule value="Severe hypoglycemia" min="0" max="54"/>
            <rule value="Hypoglycemia" min="54" max="70"/>
            <rule value="Low glucose" min="70" max="140"/>
            <rule value="Normal glucose" min="140" max="180"/>
            <rule value="High glucose" min="180" max="250"/>
            <rule value="Hyperglycemia" min="250"/>
        </attribute>
    </discretization-rules>
</state>
"""


# -----------------------------
# DF Builders (simulate raw-concept output)
# -----------------------------

def df_basal_multi_point_same_rule() -> pd.DataFrame:
    """3 tuples at T=0h, T=6h, T=12h, all match "SubCutaneous Low"."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (15, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("06:00"), make_ts("06:00"), (18, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), (12, "SubCutaneous"), "raw-concept"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_basal_gap_exceeds_window() -> pd.DataFrame:
    """2 tuples at T=0h and T=26h (gap > 24h)."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (15, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("02:00", day=1), make_ts("02:00", day=1), (16, "SubCutaneous"), "raw-concept"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_basal_different_routes_same_dosage() -> pd.DataFrame:
    """2 tuples: SubCutaneous vs IntraVenous, both Low dosage."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (12, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("04:00"), make_ts("04:00"), (15, "IntraVenous"), "raw-concept"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_basal_same_route_different_dosages() -> pd.DataFrame:
    """2 tuples: same route, different dosage levels (Low vs High)."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (12, "SubCutaneous"), "raw-concept"),  # Low
        (1, "BASAL_BITZUA", make_ts("04:00"), make_ts("04:00"), (50, "SubCutaneous"), "raw-concept"),  # High
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_basal_interpolate_skip() -> pd.DataFrame:
    """4 tuples: Low→Low→High→Low (test max_skip=1)."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (12, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("04:00"), make_ts("04:00"), (15, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("08:00"), make_ts("08:00"), (50, "SubCutaneous"), "raw-concept"),  # outlier
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), (14, "SubCutaneous"), "raw-concept"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_basal_order_all() -> pd.DataFrame:
    """2 tuples for order="all" test."""
    rows = [
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (15, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("04:00"), make_ts("04:00"), (20, "SubCutaneous"), "raw-concept"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


def df_glucose_three_different_levels() -> pd.DataFrame:
    """3 glucose measurements at different discretization levels."""
    rows = [
        (1, "GLUCOSE_MEASURE", make_ts("00:00"), make_ts("00:00"), (80,), "raw-concept"),   # Low glucose
        (1, "GLUCOSE_MEASURE", make_ts("06:00"), make_ts("06:00"), (150,), "raw-concept"),  # Normal glucose
        (1, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), (300,), "raw-concept"),  # Hyperglycemia
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])


# -----------------------------
# Pytest Fixtures
# -----------------------------

@pytest.fixture
def repo_with_basal(tmp_path: Path) -> TAKRepository:
    """BASAL raw-concept + state (COMPLETE ACTUAL TAK)."""
    raw_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_BASAL_XML)
    state_path = write_xml(tmp_path, "BASAL_BITZUA_STATE.xml", STATE_BASAL_XML)
    repo = TAKRepository()
    raw_tak = RawConcept.parse(raw_path)
    repo.register(raw_tak)
    set_tak_repository(repo)
    state_tak = State.parse(state_path)
    repo.register(state_tak)
    return repo


@pytest.fixture
def repo_with_basal_all(tmp_path: Path) -> TAKRepository:
    """BASAL + order="all" state."""
    raw_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_BASAL_XML)
    state_path = write_xml(tmp_path, "BASAL_ALL_STATE.xml", STATE_BASAL_ALL_XML)
    repo = TAKRepository()
    raw_tak = RawConcept.parse(raw_path)
    repo.register(raw_tak)
    set_tak_repository(repo)
    state_tak = State.parse(state_path)
    repo.register(state_tak)
    return repo


@pytest.fixture
def repo_with_basal_permissive(tmp_path: Path) -> TAKRepository:
    """BASAL + permissive order="all" state."""
    raw_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_BASAL_XML)
    state_path = write_xml(tmp_path, "BASAL_PERMISSIVE_STATE.xml", STATE_BASAL_PERMISSIVE_XML)
    repo = TAKRepository()
    raw_tak = RawConcept.parse(raw_path)
    repo.register(raw_tak)
    set_tak_repository(repo)
    state_tak = State.parse(state_path)
    repo.register(state_tak)
    return repo


@pytest.fixture
def repo_with_glucose(tmp_path: Path) -> TAKRepository:
    """GLUCOSE raw-concept + state (COMPLETE ACTUAL TAK)."""
    raw_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    state_path = write_xml(tmp_path, "GLUCOSE_STATE.xml", STATE_GLUCOSE_XML)
    repo = TAKRepository()
    raw_tak = RawConcept.parse(raw_path)
    repo.register(raw_tak)
    set_tak_repository(repo)
    state_tak = State.parse(state_path)
    repo.register(state_tak)
    return repo


# -----------------------------
# Tests: Parsing & Validation
# -----------------------------

def test_parse_state_validates_structure(repo_with_basal):
    """Validate XML structure and instantiation."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    assert state_tak.name == "BASAL_BITZUA_STATE"
    assert state_tak.derived_from == "BASAL_BITZUA"
    assert state_tak.good_after == parse_duration("24h")
    assert state_tak.interpolate is True
    assert state_tak.max_skip == 1
    assert len(state_tak.discretization_rules) == 5  # Very Low, Low, Medium, High, Very High
    assert len(state_tak.abstraction_rules) == 9


def test_validate_raises_if_parent_not_found(tmp_path: Path):
    """Validation fails if derived_from is missing."""
    bad_state_xml = STATE_BASAL_XML.replace('name="BASAL_BITZUA"', 'name="NONEXISTENT"')
    state_path = write_xml(tmp_path, "BAD_STATE.xml", bad_state_xml)
    repo = TAKRepository()
    set_tak_repository(repo)
    with pytest.raises(ValueError, match="not found in TAK repository"):
        State.parse(state_path)


# -----------------------------
# Tests: Discretization
# -----------------------------

def test_discretize_numeric_ranges(repo_with_basal):
    """Discretization maps numeric dosages correctly."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (5, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("01:00"), make_ts("01:00"), (25, "IntraVenous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("02:00"), make_ts("02:00"), (55, "SubCutaneous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_disc = state_tak._discretize(df_in.copy())
    discrete_vals = list(df_disc["Value"])
    assert discrete_vals[0] == ("Very Low", "SubCutaneous")
    assert discrete_vals[1] == ("Medium", "IntraVenous")
    assert discrete_vals[2] == ("High", "SubCutaneous")


def test_discretize_filters_out_of_range(repo_with_glucose):
    """Values outside raw-concept's allowed range are pre-filtered by RawConcept.apply()."""
    # This test should use State.apply() end-to-end, not _discretize() in isolation
    state_tak = repo_with_glucose.get("GLUCOSE_MEASURE_STATE")
    raw_tak = repo_with_glucose.get("GLUCOSE_MEASURE")
    
    # Simulate raw-concept output (RawConcept.apply() already filtered out-of-range)
    df_raw_input = pd.DataFrame([
        (1, "GLUCOSE_LAB_MEASURE", make_ts("00:00"), make_ts("00:00"), -10),   # will be filtered by raw-concept
        (1, "GLUCOSE_LAB_MEASURE", make_ts("01:00"), make_ts("01:00"), 80),    # valid
        (1, "GLUCOSE_LAB_MEASURE", make_ts("02:00"), make_ts("02:00"), 700),   # will be filtered by raw-concept
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    # Apply raw-concept first (this filters out-of-range values)
    df_raw_output = raw_tak.apply(df_raw_input)
    
    # Now apply state (should only see value=80)
    df_state_output = state_tak.apply(df_raw_output)
    assert len(df_state_output) == 1
    assert df_state_output.iloc[0]["Value"] == "('Low glucose',)"


# -----------------------------
# Tests: Abstraction
# -----------------------------

def test_abstract_first_order(repo_with_basal):
    """order="first" returns first matching rule."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_disc = pd.DataFrame([
        (1, "BASAL_BITZUA_STATE", make_ts("00:00"), make_ts("00:00"), ("Low", "SubCutaneous"), "state"),
        (1, "BASAL_BITZUA_STATE", make_ts("01:00"), make_ts("01:00"), ("High", "IntraVenous"), "state"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_abs = state_tak._abstract(df_disc.copy())
    assert len(df_abs) == 2
    assert df_abs.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_abs.iloc[1]["Value"] == "IntraVenous High"


def test_abstract_all_order(repo_with_basal_all):
    """order="all" returns list of all matching rules."""
    state_tak = repo_with_basal_all.get("BASAL_ALL_STATE")
    df_disc = pd.DataFrame([
        (1, "BASAL_ALL_STATE", make_ts("00:00"), make_ts("00:00"), ("Low", "SubCutaneous"), "state"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_abs = state_tak._abstract(df_disc.copy())
    assert len(df_abs) == 1
    val = df_abs.iloc[0]["Value"]
    assert isinstance(val, list)
    assert set(val) == {"SubCutaneous", "Low Dose"}


# -----------------------------
# Tests: Merging
# -----------------------------

def test_merge_multi_point_same_rule(repo_with_basal):
    """3 tuples matching same rule merge into 1 interval."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_multi_point_same_rule()  # T=0h, 6h, 12h
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "SubCutaneous Low"
    assert row["StartDateTime"] == make_ts("00:00")
    # CORRECTED: EndDateTime = last_merged_time (12h) + good_after (24h) = 36h (day=1, 12:00)
    assert row["EndDateTime"] == make_ts("12:00", day=1)


def test_merge_gap_exceeds_window(repo_with_basal):
    """Gap > good_after breaks into 2 intervals."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_gap_exceeds_window()
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    assert all(df_out["Value"] == "SubCutaneous Low")


def test_merge_different_routes_same_dosage_no_merge(repo_with_basal):
    """Different routes don't merge, but intervals extend to next sample or +good_after."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_different_routes_same_dosage()  # T=0h SubCutaneous, T=4h IntraVenous
    df_out = state_tak.apply(df_in)
    
    # Expected: 2 intervals
    # - [T=0h → T=4h, "SubCutaneous Low"] (ends at next sample's start)
    # - [T=4h → T=28h (4h+24h), "IntraVenous Low"] (no next sample, extends by good_after)
    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_out.iloc[0]["EndDateTime"] == make_ts("04:00")
    assert df_out.iloc[1]["Value"] == "IntraVenous Low"
    assert df_out.iloc[1]["EndDateTime"] == make_ts("04:00", day=1)  # +24h


def test_merge_same_route_different_dosages_no_merge(repo_with_basal):
    """Same route but different dosage levels don't merge."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_same_route_different_dosages()
    df_out = state_tak.apply(df_in)
    # With corrected interpolation logic: no skip allowed (no third point to validate)
    # Expected: 2 separate intervals
    assert len(df_out) == 2
    values = set(df_out["Value"])
    assert values == {"SubCutaneous Low", "SubCutaneous High"}


def test_merge_interpolate_skip_outlier(repo_with_basal):
    """max_skip=1 allows skipping 1 outlier."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_interpolate_skip()  # T=0h, 4h, 8h(outlier), 12h
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "SubCutaneous Low"
    assert row["StartDateTime"] == make_ts("00:00")
    # CORRECTED: EndDateTime = last_merged_time (12h) + good_after (24h) = 36h
    assert row["EndDateTime"] == make_ts("12:00", day=1)


def test_merge_order_all_overlapping_intervals(repo_with_basal_all):
    """order="all" emits 2 overlapping intervals (one per rule)."""
    state_tak = repo_with_basal_all.get("BASAL_ALL_STATE")
    df_in = df_basal_order_all()  # T=0h, 4h with good_after=6h
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    values = set(df_out["Value"])
    assert values == {"SubCutaneous", "Low Dose"}
    assert all(df_out["StartDateTime"] == make_ts("00:00"))
    # CORRECTED: Each rule merges 2 samples (0h, 4h) → EndDateTime = 4h + 6h = 10h
    assert all(df_out["EndDateTime"] == make_ts("10:00"))


def test_merge_order_all_permissive_rules(repo_with_basal_permissive):
    """Permissive rules: less-specific rule matches more tuples."""
    state_tak = repo_with_basal_permissive.get("BASAL_PERMISSIVE_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (15, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("04:00"), make_ts("04:00"), (20, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("08:00"), make_ts("08:00"), (12, "SubCutaneous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = state_tak.apply(df_in)
    
    # CORRECTED: With good_after=6h, T=8h is beyond window from T=0h (0h+6h=6h)
    # So we get 2 intervals per rule:
    # - First interval: [0h → 4h], EndDateTime = 4h + 6h = 10h, but T=8h arrives before 10h → EndDateTime = 8h
    # - Second interval: [8h → 8h+6h = 14h]
    assert len(df_out) == 4
    values = set(df_out["Value"])
    assert values == {"SubCutaneous Low", "Low Dose"}
    
    # Check that each rule has 2 intervals
    low_dose_intervals = df_out[df_out["Value"] == "Low Dose"]
    subcut_low_intervals = df_out[df_out["Value"] == "SubCutaneous Low"]
    assert len(low_dose_intervals) == 2
    assert len(subcut_low_intervals) == 2
    
    # First interval for each rule: [0h → 8h] (stops at next sample)
    assert low_dose_intervals.iloc[0]["EndDateTime"] == make_ts("08:00")
    assert subcut_low_intervals.iloc[0]["EndDateTime"] == make_ts("08:00")
    
    # Second interval for each rule: [8h → 14h] (8h + 6h)
    assert low_dose_intervals.iloc[1]["StartDateTime"] == make_ts("08:00")
    assert low_dose_intervals.iloc[1]["EndDateTime"] == make_ts("14:00")
    assert subcut_low_intervals.iloc[1]["StartDateTime"] == make_ts("08:00")
    assert subcut_low_intervals.iloc[1]["EndDateTime"] == make_ts("14:00")


# -----------------------------
# Tests: Edge Cases
# -----------------------------

def test_single_point_dull_state(repo_with_basal):
    """Single tuple emits interval extended by good_after."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (12, "SubCutaneous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "SubCutaneous Low"
    assert row["StartDateTime"] == make_ts("00:00")
    # CORRECTED: EndDateTime = 0h + 24h = 24h (next day 00:00)
    assert row["EndDateTime"] == make_ts("00:00", day=1)


def test_no_abstraction_rules_emits_discrete_string(repo_with_glucose):
    """State without abstraction rules emits discrete tuples as strings."""
    state_tak = repo_with_glucose.get("GLUCOSE_MEASURE_STATE")
    df_in = df_glucose_three_different_levels()
    df_out = state_tak.apply(df_in)
    # With corrected interpolation: T=6h "Normal glucose" cannot be skipped (no third point to validate)
    # Expected: 3 separate intervals (no merging)
    assert len(df_out) == 3
    values = list(df_out["Value"])
    assert values[0] == "('Low glucose',)"
    assert values[1] == "('Normal glucose',)"
    assert values[2] == "('Hyperglycemia',)"


def test_empty_input_returns_empty(repo_with_basal):
    """Empty input returns empty DataFrame."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = state_tak.apply(df_in)
    assert df_out.empty

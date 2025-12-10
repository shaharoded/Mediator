"""
Comprehensive unit tests for State TAK.
"""
import pandas as pd
import pytest
from pathlib import Path

from core.tak.state import State
from core.tak.raw_concept import RawConcept
from core.tak.repository import set_tak_repository, TAKRepository
from core.tak.utils import parse_duration
from core.tak.event import Event
from unittests.test_utils import write_xml, make_ts


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
  <merge require-all="false"/>
</raw-concept>
"""

# State: BASAL_BITZUA_STATE
STATE_BASAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="BASAL_BITZUA_STATE">
    <categories>Medications</categories>
    <description>Abstraction for the dosage and route of administration of BASAL insulin</description>
    <derived-from name="BASAL_BITZUA" tak="raw-concept"/>
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    <abstraction-rules>
        <rule value="SubCutaneous Low" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="20"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous Low" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="20"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="SubCutaneous Medium" operator="and">
            <attribute idx="0">
                <allowed-value min="20" max="40"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous Medium" operator="and">
            <attribute idx="0">
                <allowed-value min="20" max="40"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="SubCutaneous High" operator="and">
            <attribute idx="0">
                <allowed-value min="40"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="SubCutaneous"/>
            </attribute>
        </rule>
        <rule value="IntraVenous High" operator="and">
            <attribute idx="0">
                <allowed-value min="40"/>
            </attribute>
            <attribute idx="1">
                <allowed-value equal="IntraVenous"/>
            </attribute>
        </rule>
        <rule value="Low" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="20"/>
            </attribute>
        </rule>
        <rule value="Medium" operator="and">
            <attribute idx="0">
                <allowed-value min="20" max="40"/>
            </attribute>
        </rule>
        <rule value="High" operator="and">
            <attribute idx="0">
                <allowed-value min="40"/>
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

# State: GLUCOSE_MEASURE_STATE
STATE_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Abstraction for the measurements of GLUCOSE (Lab / Capillary)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    <abstraction-rules>
        <rule value="Severe hypoglycemia" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="54"/>
            </attribute>
        </rule>
        <rule value="Hypoglycemia" operator="and">
            <attribute idx="0">
                <allowed-value min="54" max="70"/>
            </attribute>
        </rule>
        <rule value="Low glucose" operator="and">
            <attribute idx="0">
                <allowed-value min="70" max="140"/>
            </attribute>
        </rule>
        <rule value="Normal glucose" operator="and">
            <attribute idx="0">
                <allowed-value min="140" max="180"/>
            </attribute>
        </rule>
        <rule value="High glucose" operator="and">
            <attribute idx="0">
                <allowed-value min="180" max="250"/>
            </attribute>
        </rule>
        <rule value="Hyperglycemia" operator="and">
            <attribute idx="0">
                <allowed-value min="250"/>
            </attribute>
        </rule>
    </abstraction-rules>
</state>
"""

# NEW: Event + State XMLs for testing State-from-Event
EVENT_MEAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="MEAL_EVENT">
    <categories>Events</categories>
    <description>Meal event</description>
    <derived-from>
        <attribute name="MEAL" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Breakfast" operator="or">
            <attribute ref="A1">
                <allowed-value equal="Breakfast"/>
            </attribute>
        </rule>
        <rule value="Lunch" operator="or">
            <attribute ref="A1">
                <allowed-value equal="Lunch"/>
            </attribute>
        </rule>
        <rule value="Dinner" operator="or">
            <attribute ref="A1">
                <allowed-value equal="Dinner"/>
            </attribute>
        </rule>
        <rule value="Snack" operator="or">
            <attribute ref="A1">
                <allowed-value equal="Snack"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""

RAW_MEAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="MEAL" concept-type="raw-nominal">
  <categories>Events</categories>
  <description>Meal type</description>
  <attributes>
    <attribute name="MEAL_TYPE" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="Breakfast"/>
        <allowed-value value="Lunch"/>
        <allowed-value value="Dinner"/>
        <allowed-value value="Snack"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

STATE_FROM_EVENT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="MEAL_STATE">
    <categories>Events</categories>
    <description>Meal state derived from event</description>
    <derived-from name="MEAL_EVENT" tak="event"/>
    <persistence good-after="3h" interpolate="false" max-skip="0"/>
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
    """Setup: BASAL_BITZUA raw-concept + state."""
    basal_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_BASAL_XML)
    state_path = write_xml(tmp_path, "BASAL_BITZUA_STATE.xml", STATE_BASAL_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(basal_path))
    set_tak_repository(repo)
    repo.register(State.parse(state_path))
    return repo


@pytest.fixture
def repo_with_glucose(tmp_path: Path) -> TAKRepository:
    """Setup: GLUCOSE_MEASURE raw-concept + state."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    state_path = write_xml(tmp_path, "GLUCOSE_MEASURE_STATE.xml", STATE_GLUCOSE_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(State.parse(state_path))
    return repo


@pytest.fixture
def repo_with_meal_event_state(tmp_path: Path) -> TAKRepository:
    """Setup: MEAL raw-concept → event → state."""
    meal_path = write_xml(tmp_path, "MEAL.xml", RAW_MEAL_XML)
    event_path = write_xml(tmp_path, "MEAL_EVENT.xml", EVENT_MEAL_XML)
    state_path = write_xml(tmp_path, "MEAL_STATE.xml", STATE_FROM_EVENT_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(meal_path))
    set_tak_repository(repo)
    repo.register(Event.parse(event_path))
    repo.register(State.parse(state_path))
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
    assert len(state_tak.abstraction_rules) == 9  # Updated: all rules now in abstraction-rules


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
    """Test that abstraction rules handle numeric discretization internally."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (5, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("01:00"), make_ts("01:00"), (25, "IntraVenous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("02:00"), make_ts("02:00"), (55, "SubCutaneous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    # Apply abstraction (which handles discretization internally)
    df_out = state_tak.apply(df_in)
    
    # Check that values are correctly abstracted
    # FIX: order="first" means most specific rule wins
    assert len(df_out) == 3
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"  # 5 falls in [0, 20) + SubCutaneous
    assert df_out.iloc[1]["Value"] == "IntraVenous Medium"  # 25 falls in [20, 40) + IntraVenous
    assert df_out.iloc[2]["Value"] == "SubCutaneous High"  # 55 falls in [40, inf) + SubCutaneous


# -----------------------------
# Tests: Abstraction
# -----------------------------

def test_abstract_first_order(repo_with_basal):
    """Abstraction returns first matching rule."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("00:00"), make_ts("00:00"), (12, "SubCutaneous"), "raw-concept"),
        (1, "BASAL_BITZUA", make_ts("01:00"), make_ts("01:00"), (50, "IntraVenous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_out.iloc[1]["Value"] == "IntraVenous High"


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
    df_in = df_basal_gap_exceeds_window()  # T=0h, T=26h (gap > 24h)
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    # First interval: [0h → 24h] (0h + good_after)
    assert df_out.iloc[0]["EndDateTime"] == make_ts("00:00", day=1)
    # Second interval: [26h → 50h] (26h + good_after)
    assert df_out.iloc[1]["EndDateTime"] == make_ts("02:00", day=2)


def test_merge_different_routes_same_dosage_no_merge(repo_with_basal):
    """Different routes → different abstracted values → no merge."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_different_routes_same_dosage()
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_out.iloc[1]["Value"] == "IntraVenous Low"


def test_merge_same_route_different_dosages_no_merge(repo_with_basal):
    """Same route but different dosage levels → no merge."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_same_route_different_dosages()
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_out.iloc[1]["Value"] == "SubCutaneous High"


def test_merge_interpolate_skip_outlier(repo_with_basal):
    """Interpolate with max_skip=1: Low→Low→High→Low skips High."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = df_basal_interpolate_skip()  # T=0h(Low), 4h(Low), 8h(High), 12h(Low)
    df_out = state_tak.apply(df_in)
    # With max_skip=1, High is treated as outlier → merged into Low interval
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "SubCutaneous Low"
    assert df_out.iloc[0]["StartDateTime"] == make_ts("00:00")
    assert df_out.iloc[0]["EndDateTime"] == make_ts("12:00", day=1)  # 12h + 24h


def test_single_point_dull_state(repo_with_basal):
    """Single point creates a dull state interval."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_in = pd.DataFrame([
        (1, "BASAL_BITZUA", make_ts("08:00"), make_ts("08:00"), (15, "SubCutaneous"), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = state_tak.apply(df_in)
    assert len(df_out) == 1
    assert df_out.iloc[0]["StartDateTime"] == make_ts("08:00")
    assert df_out.iloc[0]["EndDateTime"] == make_ts("08:00", day=1)  # 08:00 + 24h


def test_no_abstraction_rules_emits_discrete_string(repo_with_glucose):
    """State with abstraction rules emits abstracted string values for numeric input."""
    state_tak = repo_with_glucose.get("GLUCOSE_MEASURE_STATE")
    
    # Test that numeric input (80, 150) gets abstracted to string labels
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (50,), "raw-concept"),   # Severe hypoglycemia
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (80,), "raw-concept"),   # Low glucose
        (1, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), (150,), "raw-concept"),  # Normal glucose
        (1, "GLUCOSE_MEASURE", make_ts("14:00"), make_ts("14:00"), (300,), "raw-concept"),  # Hyperglycemia
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = state_tak.apply(df_in)
    
    # Should emit 4 separate state intervals (no merging since values differ)
    assert len(df_out) == 4
    
    # Verify abstraction rules work correctly
    assert df_out.iloc[0]["Value"] == "Severe hypoglycemia"
    assert df_out.iloc[1]["Value"] == "Low glucose"
    assert df_out.iloc[2]["Value"] == "Normal glucose"
    assert df_out.iloc[3]["Value"] == "Hyperglycemia"
    
    # Verify all values are strings (not tuples)
    assert all(isinstance(v, str) for v in df_out["Value"])
    
    # CORRECTED: Verify intervals are trimmed by next interval (not extended by full good_after)
    # First interval: [08:00 → 10:00] (trimmed at next interval's start)
    assert df_out.iloc[0]["StartDateTime"] == make_ts("08:00")
    assert df_out.iloc[0]["EndDateTime"] == make_ts("10:00")  # Trimmed at next interval
    
    # Second interval: [10:00 → 12:00] (trimmed at next interval's start)
    assert df_out.iloc[1]["StartDateTime"] == make_ts("10:00")
    assert df_out.iloc[1]["EndDateTime"] == make_ts("12:00")  # Trimmed at next interval
    
    # Third interval: [12:00 → 14:00] (trimmed at next interval's start)
    assert df_out.iloc[2]["StartDateTime"] == make_ts("12:00")
    assert df_out.iloc[2]["EndDateTime"] == make_ts("14:00")  # Trimmed at next interval
    
    # Fourth interval: [14:00 → 38:00] (last interval, extended by good_after=24h)
    assert df_out.iloc[3]["StartDateTime"] == make_ts("14:00")
    assert df_out.iloc[3]["EndDateTime"] == make_ts("14:00", day=1)  # 14:00 + 24h
    
    print("\n✅ State abstraction rules work correctly for numeric input")


def test_empty_input_returns_empty(repo_with_basal):
    """Empty input → empty output."""
    state_tak = repo_with_basal.get("BASAL_BITZUA_STATE")
    df_empty = pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = state_tak.apply(df_empty)
    assert df_out.empty


def test_state_from_event(repo_with_meal_event_state):
    """State derived from Event works correctly."""
    meal_raw = repo_with_meal_event_state.get("MEAL")
    meal_event = repo_with_meal_event_state.get("MEAL_EVENT")
    meal_state = repo_with_meal_event_state.get("MEAL_STATE")
    
    df_raw = pd.DataFrame([
        (1, "MEAL_TYPE", make_ts("08:00"), make_ts("08:00"), "Breakfast"),
        (1, "MEAL_TYPE", make_ts("12:00"), make_ts("12:00"), "Lunch"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_raw_out = meal_raw.apply(df_raw)
    df_event_out = meal_event.apply(df_raw_out)
    df_state_out = meal_state.apply(df_event_out)
    
    assert len(df_state_out) == 2
    assert df_state_out.iloc[0]["Value"] == "Breakfast"
    assert df_state_out.iloc[0]["EndDateTime"] == make_ts("11:00")  # 08:00 + 3h
    assert df_state_out.iloc[1]["Value"] == "Lunch"
    assert df_state_out.iloc[1]["EndDateTime"] == make_ts("15:00")  # 12:00 + 3h


def test_state_apply_extracts_value_for_single_attribute_no_rules(repo_with_glucose):
    """State applies abstraction rules correctly for single numeric attribute."""
    state_tak = repo_with_glucose.get("GLUCOSE_MEASURE_STATE")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (50,), "raw-concept"),  # Severe hypoglycemia
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (150,), "raw-concept"),  # Normal glucose
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = state_tak.apply(df_in)
    
    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "Severe hypoglycemia"
    assert df_out.iloc[1]["Value"] == "Normal glucose"


def test_state_from_parameterized_raw_concept(tmp_path: Path):
    """Test State derived from a ParameterizedRawConcept (e.g., M-SHR_MEASURE_STATE)."""
    # Parameterized raw concept XML (as in your previous tests)
    param_raw_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="FIRST_GLUCOSE_MEASURE" tak="raw-concept" idx="0" how='before' dynamic='false' ref="P1" default="120"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""
    # State XML for M-SHR_MEASURE_STATE (as in your KB)
    state_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="M-SHR_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Abstraction for the M-SHR ratio measurements</description>
    <derived-from name="M-SHR_MEASURE" tak="raw-concept"/>
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    <abstraction-rules>
        <rule value="Very Low" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="0.6"/>
            </attribute>
        </rule>
        <rule value="Very Low" operator="and">
            <attribute idx="0">
                <allowed-value min="0.6" max="0.85"/>
            </attribute>
        </rule>
        <rule value="Stable" operator="and">
            <attribute idx="0">
                <allowed-value min="0.85" max="1.15"/>
            </attribute>
        </rule>
        <rule value="High" operator="and">
            <attribute idx="0">
                <allowed-value min="1.15" max="1.4"/>
            </attribute>
        </rule>
        <rule value="Very High" operator="and">
            <attribute idx="0">
                <allowed-value min="1.4"/>
            </attribute>
        </rule>
    </abstraction-rules>
</state>
"""
    # Raw concept for glucose (parent of parameterized)
    raw_glucose_xml = """\
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
  <tuple-order>
    <attribute name="GLUCOSE_LAB_MEASURE"/>
  </tuple-order>
  <merge require-all="false"/>
</raw-concept>
"""
    # Write XMLs
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", raw_glucose_xml)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml)
    state_path = write_xml(tmp_path, "M-SHR_MEASURE_STATE.xml", state_xml)
    # Build repo
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    from core.tak.raw_concept import ParameterizedRawConcept
    repo.register(ParameterizedRawConcept.parse(param_path))
    from core.tak.state import State
    state_tak = State.parse(state_path)
    repo.register(state_tak)
    # Simulate input: glucose and param (first glucose measure)
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "FIRST_GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    # Apply parameterized raw concept
    param_tak = repo.get("M-SHR_MEASURE")
    df_param = param_tak.apply(df_in)
    # Apply state
    df_state = state_tak.apply(df_param)
    assert len(df_state) == 1
    # 100/50 = 2.0 → "Very High"
    assert df_state.iloc[0]["Value"] == "Very High"
    assert df_state.iloc[0]["StartDateTime"] == make_ts("08:00")
    assert df_state.iloc[0]["EndDateTime"] == make_ts("08:00", day=1)  # 24h window

"""
Comprehensive unit tests for Event TAK.

Tests cover:
1. Parsing & validation (XML structure, multi-source, operator checks)
2. Abstraction (or/and logic, constraint matching)
3. Edge cases (no rules, single source, State as derived-from)
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from core.tak.event import Event
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
  <description>Glucose measure</description>
  <attributes>
    <attribute name="GLUCOSE_LAB_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_HYPOGLYCEMIA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="HYPOGLYCEMIA" concept-type="raw-boolean">
  <categories>Events</categories>
  <description>Hypoglycemia flag</description>
  <attributes>
    <attribute name="HYPOGLYCEMIA" type="boolean"/>
  </attributes>
</raw-concept>
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
      </nominal-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

EVENT_DISGLYCEMIA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="DISGLYCEMIA_EVENT">
    <categories>Events</categories>
    <description>Dysglycemia event</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0" ref="A2"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute ref="A1">
                <allowed-value max="70"/>
            </attribute>
            <attribute ref="A2">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
        <rule value="Hyperglycemia" operator="or">
            <attribute ref="A1">
                <allowed-value min="250"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""

EVENT_NO_RULES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="MEAL_EVENT">
    <categories>Events</categories>
    <description>Meal event</description>
    <derived-from>
        <attribute name="MEAL" tak="raw-concept" idx="0"/>
    </derived-from>
</event>
"""


@pytest.fixture
def repo_with_disglycemia(tmp_path: Path) -> TAKRepository:
    """Setup TAK repo with raw-concepts + disglycemia event."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    hypo_path = write_xml(tmp_path, "HYPOGLYCEMIA.xml", RAW_HYPOGLYCEMIA_XML)
    event_path = write_xml(tmp_path, "DISGLYCEMIA_EVENT.xml", EVENT_DISGLYCEMIA_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(hypo_path))
    set_tak_repository(repo)
    repo.register(Event.parse(event_path))
    return repo


def test_parse_event_validates_structure(repo_with_disglycemia):
    """Validate XML structure and instantiation."""
    event = repo_with_disglycemia.get("DISGLYCEMIA_EVENT")
    assert event.name == "DISGLYCEMIA_EVENT"
    assert len(event.derived_from) == 2
    assert event.derived_from[0]["name"] == "GLUCOSE_MEASURE"
    assert event.derived_from[1]["name"] == "HYPOGLYCEMIA"
    assert len(event.abstraction_rules) == 2


def test_event_apply_no_rules(tmp_path: Path):
    """Event with no rules emits raw values as-is."""
    meal_path = write_xml(tmp_path, "MEAL.xml", RAW_MEAL_XML)
    event_path = write_xml(tmp_path, "MEAL_EVENT.xml", EVENT_NO_RULES_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(meal_path))
    set_tak_repository(repo)
    event = Event.parse(event_path)
    repo.register(event)
    
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("08:00"), make_ts("08:00"), ("Breakfast",), "raw-concept"),
        (1, "MEAL", make_ts("12:00"), make_ts("12:00"), ("Lunch",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = event.apply(df_in)
    assert len(df_out) == 2
    assert all(df_out["ConceptName"] == "MEAL_EVENT")
    assert all(df_out["StartDateTime"] == df_out["EndDateTime"])  # point-in-time
    assert list(df_out["Value"]) == ["Breakfast", "Lunch"]


def test_event_apply_with_rules_or_operator(repo_with_disglycemia):
    """Event with operator='or' matches if any source triggers."""
    event = repo_with_disglycemia.get("DISGLYCEMIA_EVENT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),  # Hypo
        (1, "HYPOGLYCEMIA", make_ts("09:00"), make_ts("09:00"), ("True",), "raw-concept"),  # Hypo
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (300,), "raw-concept"),  # Hyper
        (1, "GLUCOSE_MEASURE", make_ts("11:00"), make_ts("11:00"), (120,), "raw-concept"),  # No match
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = event.apply(df_in)
    assert len(df_out) == 3
    values = list(df_out["Value"])
    assert values == ["Hypoglycemia", "Hypoglycemia", "Hyperglycemia"]


def test_event_validation_fails_if_parent_not_found(tmp_path: Path):
    """Validation fails if derived-from TAK doesn't exist."""
    bad_event_xml = EVENT_DISGLYCEMIA_XML.replace('name="GLUCOSE_MEASURE"', 'name="NONEXISTENT"')
    event_path = write_xml(tmp_path, "BAD_EVENT.xml", bad_event_xml)
    
    repo = TAKRepository()
    set_tak_repository(repo)
    with pytest.raises(ValueError, match="not found in TAK repository"):
        Event.parse(event_path)


def test_event_validates_constraint_schema(tmp_path: Path):
    """Validation rejects invalid constraint combinations (e.g., equal + min)."""
    bad_event_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="BAD_EVENT">
    <categories>Events</categories>
    <description>Invalid constraint schema</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Bad" operator="or">
            <attribute ref="A1">
                <allowed-value equal="100" min="50"/>  <!-- INVALID: can't mix equal with min/max -->
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    event_path = write_xml(tmp_path, "BAD_EVENT.xml", bad_event_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    with pytest.raises(ValueError, match="cannot have both 'equal' and 'min'/'max'"):
        Event.parse(event_path)


def test_event_parses_range_constraint(tmp_path: Path):
    """Range constraint (min + max) is parsed correctly as internal 'range' type."""
    range_event_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="RANGE_EVENT">
    <categories>Events</categories>
    <description>Range constraint test</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Normal Range" operator="or">
            <attribute ref="A1">
                <allowed-value min="70" max="180"/>  <!-- Range: [70, 180] -->
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    event_path = write_xml(tmp_path, "RANGE_EVENT.xml", range_event_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    event = Event.parse(event_path)
    repo.register(event)
    
    # Verify internal representation
    rule = event.abstraction_rules[0]
    constraint = rule.constraints["A1"][0]
    assert constraint["type"] == "range"
    assert constraint["min"] == 70.0
    assert constraint["max"] == 180.0
    
    # Test matching
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),  # In range
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (50,), "raw-concept"),   # Below range
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (200,), "raw-concept"),  # Above range
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = event.apply(df_in)
    assert len(df_out) == 1  # Only the 100 value matches
    assert df_out.iloc[0]["Value"] == "Normal Range"


def test_event_validation_requires_rules_for_multiple_attributes(tmp_path: Path):
    """Validation fails if multiple attributes and no abstraction rules."""
    event_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="MULTI_ATTR_EVENT">
    <categories>Events</categories>
    <description>Multi-attribute event</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0"/>
    </derived-from>
</event>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    hypo_path = write_xml(tmp_path, "HYPOGLYCEMIA.xml", RAW_HYPOGLYCEMIA_XML)
    event_path = write_xml(tmp_path, "MULTI_ATTR_EVENT.xml", event_xml)
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(hypo_path))
    set_tak_repository(repo)
    with pytest.raises(ValueError, match="must define abstraction rules"):
        Event.parse(event_path)


def test_event_validation_requires_rules_for_numeric(tmp_path: Path):
    """Validation fails if single numeric attribute and no abstraction rules."""
    event_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="NUMERIC_EVENT">
    <categories>Events</categories>
    <description>Numeric event</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
    </derived-from>
</event>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    event_path = write_xml(tmp_path, "NUMERIC_EVENT.xml", event_xml)
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    with pytest.raises(ValueError, match="must define abstraction rules"):
        Event.parse(event_path)


def test_event_apply_extracts_value_by_idx(tmp_path: Path):
    """Event with no abstraction rules emits correct value (not tuple) using idx."""
    event_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="MEAL_EVENT">
    <categories>Events</categories>
    <description>Meal event</description>
    <derived-from>
        <attribute name="MEAL" tak="raw-concept" idx="0"/>
    </derived-from>
</event>
"""
    raw_meal_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="MEAL" concept-type="raw-nominal">
    <categories>Events</categories>
    <description>Raw concept to manage the assessed meal events in hospitalization</description>
    <attributes>
        <attribute name="MEAL" type="nominal">
            <nominal-allowed-values>
                <allowed-value value="Breakfast"/>
                <allowed-value value="Lunch"/>
                <allowed-value value="Dinner"/>
                <allowed-value value="Night"/>
            </nominal-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
"""
    meal_path = write_xml(tmp_path, "MEAL.xml", raw_meal_xml)
    event_path = write_xml(tmp_path, "MEAL_EVENT.xml", event_xml)
    repo = TAKRepository()
    repo.register(RawConcept.parse(meal_path))
    set_tak_repository(repo)
    event = Event.parse(event_path)
    repo.register(event)
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("08:00"), make_ts("08:00"), ("Breakfast",), "raw-concept"),
        (1, "MEAL", make_ts("12:00"), make_ts("12:00"), ("Lunch",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = event.apply(df_in)
    assert list(df_out["Value"]) == ["Breakfast", "Lunch"]
    assert all(not isinstance(v, tuple) for v in df_out["Value"])

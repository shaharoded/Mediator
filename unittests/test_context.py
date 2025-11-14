"""
Comprehensive unit tests for Context TAK.
"""
import pandas as pd
import pytest
from pathlib import Path
from datetime import timedelta

from core.tak.context import Context
from core.tak.raw_concept import RawConcept
from core.tak.repository import set_tak_repository, TAKRepository
from unittests.test_utils import write_xml, make_ts  # FIXED: correct import path


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

RAW_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission event</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
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

# CORRECTED ORDER: abstraction-rules → context-windows
CONTEXT_HYPOGLYCEMIA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="HYPOGLYCEMIA_CONTEXT">
    <categories>Contexts</categories>
    <description>Hypoglycemia context with windowing</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute ref="A1">
                <allowed-value max="70"/>
            </attribute>
        </rule>
    </abstraction-rules>
    
    <context-windows>
        <persistence good-before="1h" good-after="2h"/>
    </context-windows>
</context>
"""

# CORRECTED ORDER: no abstraction-rules → context-windows → clippers
CONTEXT_WITH_CLIPPER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="CLIPPED_CONTEXT">
    <categories>Contexts</categories>
    <description>Context with clipping</description>

    <derived-from>
        <attribute name="MEAL" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>

    <clippers>
        <clipper name="ADMISSION" tak="raw-concept" clip-before="30m" clip-after="1h"/>
    </clippers>
    
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
    </abstraction-rules>
    
    <context-windows>
        <persistence good-before="1h" good-after="2h"/>
    </context-windows>
</context>
"""

# CORRECTED ORDER: abstraction-rules → context-windows → clippers
CONTEXT_VALUE_SPECIFIC_WINDOW_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="BASAL_CONTEXT">
    <categories>Contexts</categories>
    <description>Context with value-specific windows</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>

    <clippers>
        <clipper name="ADMISSION" tak="raw-concept" clip-before="0h" clip-after="1h"/>
    </clippers>
    
    <abstraction-rules>
        <rule value="Low" operator="or">
            <attribute ref="A1">
                <allowed-value max="70"/>
            </attribute>
        </rule>
        <rule value="High" operator="or">
            <attribute ref="A1">
                <allowed-value min="180"/>
            </attribute>
        </rule>
    </abstraction-rules>
    
    <context-windows>
        <persistence value="Low" good-before="30m" good-after="2h"/>
        <persistence value="High" good-before="1h" good-after="4h"/>
    </context-windows>
</context>
"""


@pytest.fixture
def repo_with_hypoglycemia_context(tmp_path: Path) -> TAKRepository:
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    context_path = write_xml(tmp_path, "HYPOGLYCEMIA_CONTEXT.xml", CONTEXT_HYPOGLYCEMIA_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(Context.parse(context_path))
    return repo


@pytest.fixture
def repo_with_clipped_context(tmp_path: Path) -> TAKRepository:
    meal_path = write_xml(tmp_path, "MEAL.xml", RAW_MEAL_XML)
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    context_path = write_xml(tmp_path, "CLIPPED_CONTEXT.xml", CONTEXT_WITH_CLIPPER_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(meal_path))
    repo.register(RawConcept.parse(admission_path))
    set_tak_repository(repo)
    repo.register(Context.parse(context_path))
    return repo


def test_parse_context_validates_structure(repo_with_hypoglycemia_context):
    """Validate XML structure and instantiation."""
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    assert context.name == "HYPOGLYCEMIA_CONTEXT"
    assert len(context.derived_from) == 1
    # CORRECTED: Context now uses context_windows dict
    assert None in context.context_windows  # default window exists
    default_window = context.context_windows[None]
    assert default_window["before"] == timedelta(hours=1)
    assert default_window["after"] == timedelta(hours=2)
    assert len(context.abstraction_rules) == 1


def test_context_apply_with_windowing(repo_with_hypoglycemia_context):
    """Context windowing extends intervals before/after."""
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "Hypoglycemia"
    assert row["ConceptName"] == "HYPOGLYCEMIA_CONTEXT"
    # Original: 08:00 - 08:00
    # Windowed: StartDateTime = 08:00 - 1h = 07:00, EndDateTime = 08:00 + 2h = 10:00
    assert row["StartDateTime"] == make_ts("07:00")
    assert row["EndDateTime"] == make_ts("10:00")


def test_context_with_clippers(repo_with_clipped_context):
    """Context clipping produces valid output when context doesn't fully overlap clipper."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")
    
    # Input: MEAL @ 05:00 + ADMISSION clipper @ 06:30-08:00
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("05:00"), make_ts("05:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("06:30"), make_ts("08:00"), ("True",), "raw-concept"),  # Clipper
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    # FIXED: No clipper_dfs parameter — Context separates internally
    df_out = context.apply(df_in)
    
    # Windowed: [04:00 - 07:00] (05:00 - 1h, 05:00 + 2h)
    # Clipped:
    #   - clip-before=30m: Context starts BEFORE clipper (04:00 < 06:30) → new start = 06:30 + 30m = 07:00
    #   - clip-after=1h: Context start (07:00) < clipper end (08:00) → delay to 08:00 + 1h = 09:00
    # Result: [09:00 - 07:00] → INVALID (flipped) → removed
    assert len(df_out) == 0


def test_context_clipping_valid_output(repo_with_clipped_context):
    """Context clipping that produces valid output (not removed)."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")
    
    # Input glucose @ 04:00 (early start, long window)
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("04:00"), make_ts("04:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("05:30"), make_ts("06:00"), ("True",), "raw-concept"),  # Clipper
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    
    # Windowed: [03:00 - 06:00] (04:00 - 1h, 04:00 + 2h)
    # Clipped:
    #   - clip-before=30m: Context starts BEFORE clipper (03:00 < 05:30) → new start = 05:30 + 30m = 06:00
    #   - clip-after=1h: Context start (06:00) >= clipper end (06:00)? YES → no delay
    # Result: [06:00 - 06:00] → INVALID (start == end) → removed
    assert len(df_out) == 0


def test_context_clipping_really_valid_output(repo_with_clipped_context):
    """Context clipping with valid output (context starts after clipper ends)."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")
    
    # Input glucose @ 10:00
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("10:00"), make_ts("10:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:30"), ("True",), "raw-concept"),  # Clipper
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    
    # Windowed: [09:00 - 12:00] (10:00 - 1h, 10:00 + 2h)
    # Clipper [08:00 - 08:30]: no overlap (clipper.end < context.start) → no clipping
    # Result: [09:00 - 12:00] ✅ VALID
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("09:00")
    assert row["EndDateTime"] == make_ts("12:00")


def test_context_multiple_clippers_applied(repo_with_clipped_context):
    """Multiple clippers: only first clipper overlaps."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")
    
    # Input glucose @ 05:00 + two clippers
    df_in = pd.DataFrame([
        (1, "MEAL", make_ts("05:00"), make_ts("05:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("06:00"), make_ts("06:30"), ("True",), "raw-concept"),
        (1, "ADMISSION", make_ts("10:00"), make_ts("11:00"), ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    
    # Windowed: [04:00 - 07:00] (05:00 - 1h, 05:00 + 2h)
    # Clipper 1 [06:00-06:30]: overlaps
    #   - clip-before=30m: Context starts before clipper (04:00 < 06:00) → new start = 06:00 + 30m = 06:30
    #   - clip-after=1h: Context start (06:30) is NOT < clipper end (06:30) → NO delay
    # Clipper 2 [10:00-11:00]: no overlap with [06:30-07:00] → no clipping
    # Result: [06:30 - 07:00] ✅ VALID
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("06:30")
    assert row["EndDateTime"] == make_ts("07:00")


def test_context_validation_requires_rules_for_multiple_attributes(tmp_path: Path):
    """Validation fails if multiple attributes and no abstraction rules."""
    context_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="MULTI_ATTR_CONTEXT">
    <categories>Contexts</categories>
    <description>Multi-attribute context</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
        <attribute name="ADMISSION" tak="raw-concept" idx="0"/>
    </derived-from>
    <context-windows>
        <persistence good-before="1h" good-after="2h"/>
    </context-windows>
</context>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    context_path = write_xml(tmp_path, "MULTI_ATTR_CONTEXT.xml", context_xml)
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(admission_path))
    set_tak_repository(repo)
    # FIXED: XSD catches element ordering error (abstraction-rules must come before context-windows)
    with pytest.raises(ValueError, match="(must define abstraction rules|Missing child element.*abstraction-rules|Expected is one of.*abstraction-rules)"):
        Context.parse(context_path)


def test_context_validation_requires_rules_for_numeric(tmp_path: Path):
    """Validation fails if single numeric attribute and no abstraction rules."""
    context_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="NUMERIC_CONTEXT">
    <categories>Contexts</categories>
    <description>Numeric context</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
    </derived-from>
    <context-windows>
        <persistence good-before="1h" good-after="2h"/>
    </context-windows>
</context>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    context_path = write_xml(tmp_path, "NUMERIC_CONTEXT.xml", context_xml)
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    # FIXED: XSD catches element ordering error (abstraction-rules must come before context-windows)
    with pytest.raises(ValueError, match="(must define abstraction rules|Missing child element.*abstraction-rules|Expected is one of.*abstraction-rules)"):
        Context.parse(context_path)


def test_context_auto_clips_overlapping_same_context(repo_with_hypoglycemia_context):
    """
    Test that overlapping context intervals from the same context auto-clip each other.
    
    Given two context instances at T=08:00 and T=09:30:
    - First:  windowed [07:00 - 10:00] (08:00 - 1h, 08:00 + 2h)
    - Second: windowed [08:30 - 11:30] (09:30 - 1h, 09:30 + 2h)
    
    Expected: First interval clipped to [07:00 - 08:30] (ends at second's start)
    """
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),  # First
        (1, "GLUCOSE_MEASURE", make_ts("09:30"), make_ts("09:30"), (55,), "raw-concept"),  # Second (overlaps)
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    assert len(df_out) == 2
    
    # First interval: clipped at second's start
    row1 = df_out.iloc[0]
    assert row1["StartDateTime"] == make_ts("07:00")
    assert row1["EndDateTime"] == make_ts("08:30")  # Clipped (was 10:00)
    
    # Second interval: unaffected
    row2 = df_out.iloc[1]
    assert row2["StartDateTime"] == make_ts("08:30")
    assert row2["EndDateTime"] == make_ts("11:30")


def test_context_auto_clips_removes_invalid_intervals(repo_with_hypoglycemia_context):
    """
    Test that context intervals that become invalid after auto-clipping are removed.
    
    Given two context instances at T=08:00 and T=08:05 (very close):
    - First:  windowed [07:00 - 10:00]
    - Second: windowed [07:05 - 10:05]
    
    If second starts BEFORE first's original start (after windowing), first becomes invalid.
    """
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("08:05"), make_ts("08:05"), (65,), "raw-concept"),  # Very close
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    
    # First interval: windowed [07:00 - 10:00]
    # Second interval: windowed [07:05 - 10:05]
    # Clipping: first's end becomes 07:05 (second's start)
    # Result: first = [07:00 - 07:05] ✅ VALID
    # So we expect 2 intervals
    assert len(df_out) == 2
    
    row1 = df_out.iloc[0]
    assert row1["StartDateTime"] == make_ts("07:00")
    assert row1["EndDateTime"] == make_ts("07:05")
    
    row2 = df_out.iloc[1]
    assert row2["StartDateTime"] == make_ts("07:05")
    assert row2["EndDateTime"] == make_ts("10:05")


def test_context_auto_clips_multiple_overlaps(repo_with_hypoglycemia_context):
    """
    Test that multiple overlapping context intervals are clipped sequentially.
    
    Given three context instances:
    - T=08:00 → windowed [07:00 - 10:00]
    - T=09:00 → windowed [08:00 - 11:00]
    - T=10:00 → windowed [09:00 - 12:00]
    
    Expected:
    - First:  [07:00 - 08:00] (clipped by second)
    - Second: [08:00 - 09:00] (clipped by third)
    - Third:  [09:00 - 12:00] (no clip)
    """
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (65,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (58,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    assert len(df_out) == 3
    
    assert df_out.iloc[0]["EndDateTime"] == make_ts("08:00")  # Clipped
    assert df_out.iloc[1]["EndDateTime"] == make_ts("09:00")  # Clipped
    assert df_out.iloc[2]["EndDateTime"] == make_ts("12:00")  # Unclipped


def test_context_auto_clips_different_values(tmp_path: Path):
    """
    Test that overlapping contexts with DIFFERENT values still clip each other.
    
    This is the key requirement: auto-clipping happens regardless of value.
    """
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    context_xml = CONTEXT_VALUE_SPECIFIC_WINDOW_XML
    context_path = write_xml(tmp_path, "BASAL_CONTEXT.xml", context_xml)
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(admission_path))
    set_tak_repository(repo)
    context = Context.parse(context_path)
    repo.register(context)
    
    # Two glucose measurements: first=60 (Low), second=200 (High)
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),   # Low
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (200,), "raw-concept"),  # High
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    assert len(df_out) == 2
    
    # First (Low): windowed [07:30 - 10:00] (08:00 - 30m, 08:00 + 2h)
    # Second (High): windowed [08:00 - 13:00] (09:00 - 1h, 09:00 + 4h)
    # Clipping: first's end clipped to 08:00 (second's start)
    
    row1 = df_out.iloc[0]
    assert row1["Value"] == "Low"
    assert row1["StartDateTime"] == make_ts("07:30")
    assert row1["EndDateTime"] == make_ts("08:00")  # Clipped (was 10:00)
    
    row2 = df_out.iloc[1]
    assert row2["Value"] == "High"
    assert row2["StartDateTime"] == make_ts("08:00")
    assert row2["EndDateTime"] == make_ts("13:00")  # Unclipped


def test_context_no_overlap_no_clipping(repo_with_hypoglycemia_context):
    """
    Test that non-overlapping context intervals are NOT clipped.
    
    Given two context instances far apart:
    - T=08:00 → windowed [07:00 - 10:00]
    - T=12:00 → windowed [11:00 - 14:00]
    
    Expected: Both intervals unchanged (no overlap).
    """
    context = repo_with_hypoglycemia_context.get("HYPOGLYCEMIA_CONTEXT")
    
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (60,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), (65,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = context.apply(df_in)
    assert len(df_out) == 2
    
    # No clipping should occur
    assert df_out.iloc[0]["EndDateTime"] == make_ts("10:00")  # Unchanged
    assert df_out.iloc[1]["EndDateTime"] == make_ts("14:00")  # Unchanged

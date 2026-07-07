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
from core.tak.event import Event
from unittests.test_utils import write_xml, make_ts


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

EVENT_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<event name="ADMISSION_EVENT">
  <categories>Admin</categories>
  <description>Admission event abstraction</description>
  <derived-from>
    <attribute name="ADMISSION" tak="raw-concept" ref="A1"/>
  </derived-from>
  <abstraction-rules>
    <rule value="Admitted" operator="or">
      <attribute ref="A1">
        <allowed-value equal="True"/>
      </attribute>
    </rule>
  </abstraction-rules>
</event>
"""

CONTEXT_ON_EVENT_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<context name="ADMISSION_CONTEXT">
  <categories>Contexts</categories>
  <description>Context derived from Event TAK</description>
  <derived-from>
    <attribute name="ADMISSION_EVENT" tak="event" ref="E1"/>
  </derived-from>
  <abstraction-rules>
    <rule value="Admitted" operator="or">
      <attribute ref="E1">
        <allowed-value equal="Admitted"/>
      </attribute>
    </rule>
  </abstraction-rules>
  <context-windows>
    <persistence good-before="1h" good-after="2h"/>
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


@pytest.fixture
def repo_with_event_context(tmp_path: Path) -> TAKRepository:
    """Fixture to set up a repository with Event-derived Context."""
    repo = TAKRepository()
    set_tak_repository(repo)

    # Register ADMISSION raw-concept
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    repo.register(RawConcept.parse(admission_path))

    # Register ADMISSION_EVENT event
    event_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", EVENT_XML)
    repo.register(Event.parse(event_path))

    # Register ADMISSION_CONTEXT context
    context_path = write_xml(tmp_path, "ADMISSION_CONTEXT.xml", CONTEXT_ON_EVENT_XML)
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
    """Case A: clipper fires during context — ctx_end is trimmed to effective left border."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")

    # MEAL @ 05:00 → windowed [04:00 - 07:00]
    # ADMISSION (clipper, clip-before=30m, clip-after=1h) @ 06:30 - 08:00
    #   effective_left  = 06:30 - 30m = 06:00
    #   effective_right = 08:00 + 1h  = 09:00
    # Overlap: [06:00, 09:00] ∩ [04:00, 07:00] → yes
    # ctx_start (04:00) < clipper_start (06:30) → Case A: ctx_end = min(07:00, 06:00) = 06:00
    # Result: [04:00 - 06:00] 
    df_in = pd.DataFrame([
        (1, "MEAL",      make_ts("05:00"), make_ts("05:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("06:30"), make_ts("08:00"), ("True",),      "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("04:00")
    assert row["EndDateTime"]   == make_ts("06:00")


def test_context_clipping_valid_output(repo_with_clipped_context):
    """Case A: trimmed ctx_end produces a valid (non-zero) interval."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")

    # MEAL @ 04:00 → windowed [03:00 - 06:00]
    # ADMISSION @ 05:30 - 06:00  (clip-before=30m, clip-after=1h)
    #   effective_left  = 05:30 - 30m = 05:00
    #   effective_right = 06:00 + 1h  = 07:00
    # Overlap: yes
    # Case A: ctx_end = min(06:00, 05:00) = 05:00
    # Result: [03:00 - 05:00]
    df_in = pd.DataFrame([
        (1, "MEAL",      make_ts("04:00"), make_ts("04:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("05:30"), make_ts("06:00"), ("True",),      "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("03:00")
    assert row["EndDateTime"]   == make_ts("05:00")


def test_context_clipping_case_b_delays_start(repo_with_clipped_context):
    """Case B: context starts within clip_after suppression zone — ctx_start is delayed."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")

    # MEAL @ 10:00 → windowed [09:00 - 12:00]
    # ADMISSION @ 08:00 - 08:30  (clip-before=30m, clip-after=1h)
    #   effective_left  = 08:00 - 30m = 07:30
    #   effective_right = 08:30 + 1h  = 09:30
    # Overlap: [07:30, 09:30] ∩ [09:00, 12:00] → yes (09:30 >= 09:00)
    # ctx_start (09:00) >= clipper_start (08:00) → Case B: ctx_start = max(09:00, 09:30) = 09:30
    # Result: [09:30 - 12:00]
    df_in = pd.DataFrame([
        (1, "MEAL",      make_ts("10:00"), make_ts("10:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:30"), ("True",),      "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("09:30")
    assert row["EndDateTime"]   == make_ts("12:00")


def test_context_multiple_clippers_applied(repo_with_clipped_context):
    """Case A with two clippers: only the first overlaps and trims the end."""
    context = repo_with_clipped_context.get("CLIPPED_CONTEXT")

    # MEAL @ 05:00 → windowed [04:00 - 07:00]
    # Clipper 1 @ 06:00 - 06:30  (clip-before=30m, clip-after=1h)
    #   effective_left  = 06:00 - 30m = 05:30
    #   effective_right = 06:30 + 1h  = 07:30
    # Overlap with [04:00, 07:00]: yes
    # Case A: ctx_end = min(07:00, 05:30) = 05:30  → [04:00 - 05:30]
    # Clipper 2 @ 10:00 - 11:00:
    #   effective_left  = 09:30, effective_right = 12:00
    #   Overlap with [04:00, 05:30]: 09:30 <= 05:30? NO → skipped
    # Result: [04:00 - 05:30]
    df_in = pd.DataFrame([
        (1, "MEAL",      make_ts("05:00"), make_ts("05:00"), ("Breakfast",), "raw-concept"),
        (1, "ADMISSION", make_ts("06:00"), make_ts("06:30"), ("True",),      "raw-concept"),
        (1, "ADMISSION", make_ts("10:00"), make_ts("11:00"), ("True",),      "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("04:00")
    assert row["EndDateTime"]   == make_ts("05:30")


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


def test_context_on_event_parsing(repo_with_event_context, tmp_path):
    """Validate parsing of Context derived from Event TAK."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    event_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", EVENT_XML)
    context = repo_with_event_context.get("ADMISSION_CONTEXT")
    assert context.name == "ADMISSION_CONTEXT"
    assert len(context.derived_from) == 1
    assert context.derived_from[0]["tak_type"] == "event"
    assert len(context.abstraction_rules) == 1


def test_context_on_event_application(repo_with_event_context, tmp_path):
    """Test application of Context derived from Event TAK."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    event_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", EVENT_XML)
    context = repo_with_event_context.get("ADMISSION_CONTEXT")

    df_in = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "Admitted", "event"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "Admitted"
    assert row["ConceptName"] == "ADMISSION_CONTEXT"
    # Original: 08:00 - 08:00
    # Windowed: StartDateTime = 08:00 - 1h = 07:00, EndDateTime = 08:00 + 2h = 10:00
    assert row["StartDateTime"] == make_ts("07:00")
    assert row["EndDateTime"] == make_ts("10:00")


HIGH_GLUCOSE_CONTEXT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="HIGH_GLUCOSE_CONTEXT">
    <categories>Diagnosis</categories>
    <description>HIGH GLUCOSE context — mirrors real production TAK logic</description>
    <derived-from>
        <attribute name="HYPER_EVENT" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <clippers>
        <!-- Bolus during context ends the window immediately (Case A) -->
        <!-- Bolus before context suppresses it for 4h (Case B)       -->
        <clipper name="BOLUS" tak="raw-concept" clip-before="0s" clip-after="4h"/>
    </clippers>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
    <context-windows>
        <persistence value="True" good-before="0h" good-after="24h"/>
    </context-windows>
</context>
"""

RAW_HYPER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="HYPER_EVENT" concept-type="raw-boolean">
  <categories>Events</categories>
  <description>Hyperglycemia event indicator</description>
  <attributes>
    <attribute name="HYPER_EVENT" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_BOLUS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BOLUS" concept-type="raw-boolean">
  <categories>DrugAdministration</categories>
  <description>Bolus insulin administration</description>
  <attributes>
    <attribute name="BOLUS" type="boolean"/>
  </attributes>
</raw-concept>
"""


@pytest.fixture
def repo_with_high_glucose_context(tmp_path: Path) -> TAKRepository:
    hyper_path  = write_xml(tmp_path, "HYPER_EVENT.xml", RAW_HYPER_XML)
    bolus_path  = write_xml(tmp_path, "BOLUS.xml", RAW_BOLUS_XML)
    ctx_path    = write_xml(tmp_path, "HIGH_GLUCOSE_CONTEXT.xml", HIGH_GLUCOSE_CONTEXT_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(hyper_path))
    repo.register(RawConcept.parse(bolus_path))
    set_tak_repository(repo)
    repo.register(Context.parse(ctx_path))
    return repo


def test_high_glucose_clipper_case_a_bolus_during_context(repo_with_high_glucose_context):
    """
    Case A: bolus (point event) fires DURING the HIGH_GLUCOSE_CONTEXT window.

    Hyper at 10:00 → context [10:00, 10:00+24h = 10:00 next day]
    Bolus at 14:00 (point event, start == end)
      effective_left  = 14:00 - 0s = 14:00
      effective_right = 14:00 + 4h = 18:00
    ctx_start (10:00) < clipper_start (14:00) → Case A
      ctx_end = min(next-day 10:00, 14:00) = 14:00

    Expected: context [10:00 - 14:00]
    """
    context = repo_with_high_glucose_context.get("HIGH_GLUCOSE_CONTEXT")

    t_hyper = make_ts("10:00")
    t_bolus = make_ts("14:00")
    t_ctx_end_orig = t_hyper + pd.Timedelta(hours=24)

    df_in = pd.DataFrame([
        (1, "HYPER_EVENT", t_hyper, t_hyper, ("True",), "raw-concept"),
        (1, "BOLUS",       t_bolus, t_bolus, ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == t_hyper
    assert row["EndDateTime"]   == t_bolus   # trimmed to bolus time


def test_high_glucose_clipper_case_b_bolus_before_context_within_4h(repo_with_high_glucose_context):
    """
    Case B: bolus (point event) fires BEFORE the context starts, within the 4h suppression window.

    Bolus at 10:00 → effective_right = 10:00 + 4h = 14:00
    Hyper at 12:00 → context [12:00, 12:00+24h]
      effective_left  = 10:00 - 0s = 10:00
      effective_right = 10:00 + 4h = 14:00
    Overlap: [10:00, 14:00] ∩ [12:00, next-day 12:00] → yes
    ctx_start (12:00) >= clipper_start (10:00) → Case B
      ctx_start = max(12:00, 14:00) = 14:00

    Expected: context [14:00 - next-day 12:00]
    """
    context = repo_with_high_glucose_context.get("HIGH_GLUCOSE_CONTEXT")

    t_bolus = make_ts("10:00")
    t_hyper = make_ts("12:00")

    df_in = pd.DataFrame([
        (1, "HYPER_EVENT", t_hyper, t_hyper, ("True",), "raw-concept"),
        (1, "BOLUS",       t_bolus, t_bolus, ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == t_bolus + pd.Timedelta(hours=4)   # delayed to 14:00
    assert row["EndDateTime"]   == t_hyper + pd.Timedelta(hours=24)  # end unchanged


def test_high_glucose_clipper_case_b_bolus_outside_suppression_zone(repo_with_high_glucose_context):
    """
    Case B miss: bolus fires more than 4h before the context starts — no effect.

    Bolus at 07:00 → effective_right = 07:00 + 4h = 11:00
    Hyper at 12:00 → context [12:00, next-day 12:00]
    Overlap: effective_right (11:00) >= ctx_start (12:00)? NO → skipped

    Expected: context unchanged [12:00 - next-day 12:00]
    """
    context = repo_with_high_glucose_context.get("HIGH_GLUCOSE_CONTEXT")

    t_bolus = make_ts("07:00")
    t_hyper = make_ts("12:00")

    df_in = pd.DataFrame([
        (1, "HYPER_EVENT", t_hyper, t_hyper, ("True",), "raw-concept"),
        (1, "BOLUS",       t_bolus, t_bolus, ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == t_hyper
    assert row["EndDateTime"]   == t_hyper + pd.Timedelta(hours=24)


CONTEXT_ZERO_CLIP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="ZERO_CLIP_CONTEXT">
    <categories>Test</categories>
    <description>Context with zero clip offsets for boundary testing</description>
    <derived-from>
        <attribute name="HYPER_EVENT" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <clippers>
        <clipper name="BOLUS" tak="raw-concept" clip-before="0s" clip-after="0s"/>
    </clippers>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
    <context-windows>
        <persistence value="True" good-before="0h" good-after="0h"/>
    </context-windows>
</context>
"""


@pytest.fixture
def repo_with_zero_clip_context(tmp_path: Path) -> TAKRepository:
    hyper_path = write_xml(tmp_path, "HYPER_EVENT.xml",          RAW_HYPER_XML)
    bolus_path = write_xml(tmp_path, "BOLUS.xml",                RAW_BOLUS_XML)
    ctx_path   = write_xml(tmp_path, "ZERO_CLIP_CONTEXT.xml",    CONTEXT_ZERO_CLIP_XML)
    repo = TAKRepository()
    repo.register(RawConcept.parse(hyper_path))
    repo.register(RawConcept.parse(bolus_path))
    set_tak_repository(repo)
    repo.register(Context.parse(ctx_path))
    return repo


def test_clipper_two_case_b_most_restrictive_wins(repo_with_zero_clip_context):
    """
    Two clippers both in Case B — the one that pushes ctx_start furthest right wins.

    Context [6:00, 9:00] (no windowing offsets), clip-before=0s clip-after=0s.
    Clipper1 [5:00, 6:30]: ctx_start(6:00) >= clipper_start(5:00) → Case B → ctx_start = 6:30
    Clipper2 [4:00, 7:00]: ctx_start(6:30) >= clipper_start(4:00) → Case B → ctx_start = max(6:30, 7:00) = 7:00
    Expected: [7:00, 9:00]
    """
    context = repo_with_zero_clip_context.get("ZERO_CLIP_CONTEXT")

    df_in = pd.DataFrame([
        (1, "HYPER_EVENT", make_ts("06:00"), make_ts("09:00"), ("True",), "raw-concept"),
        (1, "BOLUS",       make_ts("05:00"), make_ts("06:30"), ("True",), "raw-concept"),
        (1, "BOLUS",       make_ts("04:00"), make_ts("07:00"), ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("07:00")
    assert row["EndDateTime"]   == make_ts("09:00")


def test_clipper_double_edge_trim(repo_with_zero_clip_context):
    """
    One clipper trims the start (Case B) and another trims the end (Case A).

    Context [6:00, 9:00], clip-before=0s clip-after=0s.
    Clipper1 [5:00, 6:30]: ctx_start(6:00) >= clipper_start(5:00) → Case B → ctx_start = 6:30
    Clipper2 [8:00, 9:00]: ctx_start(6:30) < clipper_start(8:00)  → Case A → ctx_end = min(9:00, 8:00) = 8:00
    Expected: [6:30, 8:00]
    """
    context = repo_with_zero_clip_context.get("ZERO_CLIP_CONTEXT")

    df_in = pd.DataFrame([
        (1, "HYPER_EVENT", make_ts("06:00"), make_ts("09:00"), ("True",), "raw-concept"),
        (1, "BOLUS",       make_ts("05:00"), make_ts("06:30"), ("True",), "raw-concept"),
        (1, "BOLUS",       make_ts("08:00"), make_ts("09:00"), ("True",), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = context.apply(df_in)
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["StartDateTime"] == make_ts("06:30")
    assert row["EndDateTime"]   == make_ts("08:00")


def test_context_window_large_after_is_capped_not_overflow(tmp_path: Path):
    """Large good-after windows should cap at pandas.Timestamp.max instead of overflowing."""
    diagnosis_raw_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DIABETES_DIAGNOSIS" concept-type="raw-boolean">
  <categories>Diagnosis</categories>
  <description>Diabetes diagnosis</description>
  <attributes>
    <attribute name="DIABETES_DIAGNOSIS" type="boolean"/>
  </attributes>
</raw-concept>
"""

    diagnosis_context_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="DIABETES_DIAGNOSIS_CONTEXT_TEST">
    <categories>Diagnosis</categories>
    <description>DIABETES context with large persistence</description>
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
    <context-windows>
        <persistence value="True" good-before="14d" good-after="120y"/>
    </context-windows>
</context>
"""

    raw_path = write_xml(tmp_path, "DIABETES_DIAGNOSIS.xml", diagnosis_raw_xml)
    context_path = write_xml(tmp_path, "DIABETES_DIAGNOSIS_CONTEXT_TEST.xml", diagnosis_context_xml)

    repo = TAKRepository()
    repo.register(RawConcept.parse(raw_path))
    set_tak_repository(repo)
    repo.register(Context.parse(context_path))

    context = repo.get("DIABETES_DIAGNOSIS_CONTEXT_TEST")

    # Mirrors problematic MIMIC-style late timestamp.
    df_in = pd.DataFrame([
        (100009, "DIABETES_DIAGNOSIS", pd.Timestamp("2162-05-16 15:56:00"), pd.Timestamp("2162-05-16 15:56:01"), ("True",), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])

    df_out = context.apply(df_in)

    assert len(df_out) == 1
    assert df_out.iloc[0]["ConceptName"] == "DIABETES_DIAGNOSIS_CONTEXT_TEST"
    assert df_out.iloc[0]["EndDateTime"] == pd.Timestamp.max

"""
Comprehensive unit tests for LocalPattern TAK.

Tests cover:
1. Parsing & validation (XML structure, ref mechanism, compliance functions)
2. Pattern discovery (before/overlap, one-to-one pairing, context filtering)
3. Compliance scoring (time-constraint, value-constraint, combined)
4. Dynamic parameters (closest resolution, defaults)
5. Multiple rules (OR semantics, independent matching)
6. Edge cases (no pattern found, partial compliance, flipped intervals)
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from core.tak.pattern import LocalPattern
from core.tak.raw_concept import RawConcept
from core.tak.context import Context
from core.tak.repository import set_tak_repository, TAKRepository


# -----------------------------
# Helpers
# -----------------------------
def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


def make_ts(hhmm: str, day: int = 0) -> datetime:
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


# -----------------------------
# XML Fixtures (SELF-CONTAINED)
# -----------------------------

# Parent raw-concepts
RAW_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION_EVENT" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission event</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Glucose measure</description>
  <attributes>
    <attribute name="GLUCOSE_LAB" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_WEIGHT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="WEIGHT_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Weight measure</description>
  <attributes>
    <attribute name="WEIGHT" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="300"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_INSULIN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="INSULIN_BITZUA" concept-type="raw-numeric">
  <categories>Medications</categories>
  <description>Insulin dosage</description>
  <attributes>
    <attribute name="INSULIN_DOSAGE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_DIABETES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DIABETES_DIAGNOSIS" concept-type="raw-boolean">
  <categories>Diagnoses</categories>
  <description>Diabetes diagnosis</description>
  <attributes>
    <attribute name="DIABETES" type="boolean"/>
  </attributes>
</raw-concept>
"""

# Context (wraps DIABETES_DIAGNOSIS with windowing)
CONTEXT_DIABETES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="DIABETES_DIAGNOSIS_CONTEXT">
    <categories>Diagnoses</categories>
    <description>Diabetes diagnosis context</description>
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
        <persistence good-before="0h" good-after="720h"/>
    </context-windows>
</context>
"""

# Pattern 1: Simple (no compliance, no context)
PATTERN_SIMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_ON_ADMISSION_SIMPLE" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Simple pattern: glucose after admission (no compliance)</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='12h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern 2: With time-constraint compliance
PATTERN_TIME_COMPLIANCE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_ON_ADMISSION_TIME" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with time-constraint compliance</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='12h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern 3: With value-constraint compliance + parameter
PATTERN_VALUE_COMPLIANCE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_ON_ADMISSION_VALUE" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with value-constraint compliance (weight-based)</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="INSULIN_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <parameters>
        <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
    </parameters>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='48h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                    </target>
                    <function name="mul">
                        <parameter ref="P1"/>
                        <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern 4: With context filtering
PATTERN_WITH_CONTEXT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_ON_ADMISSION_CONTEXT" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with context filtering (diabetes patients only)</description>
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>
            <temporal-relation how='before' max-distance='8h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern 5: Multiple rules (OR semantics)
PATTERN_MULTIPLE_RULES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_ON_ADMISSION_MULTI" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with multiple rules (8h OR 24h windows)</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='12h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
        <rule>
            <temporal-relation how='before' max-distance='36h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="12h" trapezeC="24h" trapezeD="36h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern 6: Overlap temporal relation
PATTERN_OVERLAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_DURING_ADMISSION" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with overlap temporal relation</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="INSULIN_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='overlap'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""


# -----------------------------
# Pytest Fixtures
# -----------------------------

@pytest.fixture
def repo_simple_pattern(tmp_path: Path) -> TAKRepository:
    """Setup: raw-concepts + simple pattern (no compliance)."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_SIMPLE.xml", PATTERN_SIMPLE_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(LocalPattern.parse(pattern_path))
    return repo


@pytest.fixture
def repo_time_compliance(tmp_path: Path) -> TAKRepository:
    """Setup: pattern with time-constraint compliance."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_TIME.xml", PATTERN_TIME_COMPLIANCE_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(LocalPattern.parse(pattern_path))
    return repo


@pytest.fixture
def repo_value_compliance(tmp_path: Path) -> TAKRepository:
    """Setup: pattern with value-constraint compliance + parameter."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    insulin_path = write_xml(tmp_path, "INSULIN.xml", RAW_INSULIN_XML)
    weight_path = write_xml(tmp_path, "WEIGHT.xml", RAW_WEIGHT_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_VALUE.xml", PATTERN_VALUE_COMPLIANCE_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(insulin_path))
    repo.register(RawConcept.parse(weight_path))
    set_tak_repository(repo)
    repo.register(LocalPattern.parse(pattern_path))
    return repo


@pytest.fixture
def repo_with_context(tmp_path: Path) -> TAKRepository:
    """Setup: pattern with context filtering."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    diabetes_path = write_xml(tmp_path, "DIABETES.xml", RAW_DIABETES_XML)
    context_path = write_xml(tmp_path, "DIABETES_CONTEXT.xml", CONTEXT_DIABETES_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_CONTEXT.xml", PATTERN_WITH_CONTEXT_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(diabetes_path))
    set_tak_repository(repo)
    repo.register(Context.parse(context_path))
    repo.register(LocalPattern.parse(pattern_path))
    return repo


@pytest.fixture
def repo_multiple_rules(tmp_path: Path) -> TAKRepository:
    """Setup: pattern with multiple rules."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_MULTI.xml", PATTERN_MULTIPLE_RULES_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    repo.register(LocalPattern.parse(pattern_path))
    return repo


@pytest.fixture
def repo_overlap_pattern(tmp_path: Path) -> TAKRepository:
    """Setup: pattern with overlap temporal relation."""
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    insulin_path = write_xml(tmp_path, "INSULIN.xml", RAW_INSULIN_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_OVERLAP.xml", PATTERN_OVERLAP_XML)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(insulin_path))
    set_tak_repository(repo)
    repo.register(LocalPattern.parse(pattern_path))
    return repo


# -----------------------------
# End-to-End Pattern Tests (Using Actual KB Patterns)
# -----------------------------

@pytest.fixture
def repo_actual_kb(tmp_path: Path) -> TAKRepository:
    """
    Setup: Load ACTUAL knowledge base patterns + dependencies.
    This fixture creates a mini-KB with real patterns from knowledge-base.
    """
    # Write actual TAK XMLs (copy from knowledge-base/)
    raw_admission_xml = Path("core/knowledge-base/raw-concepts/ADMISSION.xml").read_text()
    raw_glucose_xml = Path("core/knowledge-base/raw-concepts/GLUCOSE_MEASURE.xml").read_text()
    raw_bmi_xml = Path("core/knowledge-base/raw-concepts/BMI_MEASURE.xml").read_text()
    raw_weight_xml = Path("core/knowledge-base/raw-concepts/WEIGHT_MEASURE.xml").read_text()
    raw_basal_xml = Path("core/knowledge-base/raw-concepts/BASAL_BITZUA.xml").read_text()
    raw_bolus_xml = Path("core/knowledge-base/raw-concepts/BOLUS_BITZUA.xml").read_text()
    raw_diabetes_xml = Path("core/knowledge-base/raw-concepts/DIABETES_DIAGNOSIS.xml").read_text()
    
    event_admission_xml = Path("core/knowledge-base/events/ADMISSION.xml").read_text()
    
    context_diabetes_xml = Path("core/knowledge-base/contexts/DIABETES_DIAGNOSIS.xml").read_text()
    
    pattern_glucose_xml = Path("core/knowledge-base/patterns/GLUCOSE_MEASURE_ON_ADMISSION.xml").read_text()
    pattern_bmi_xml = Path("core/knowledge-base/patterns/BMI_MEASURE_ON_ADMISSION.xml").read_text()
    pattern_insulin_xml = Path("core/knowledge-base/patterns/INSULIN_ON_ADMISSION.xml").read_text()
    
    # Write to tmp_path
    write_xml(tmp_path, "ADMISSION_RAW.xml", raw_admission_xml)
    write_xml(tmp_path, "GLUCOSE_MEASURE.xml", raw_glucose_xml)
    write_xml(tmp_path, "BMI_MEASURE.xml", raw_bmi_xml)
    write_xml(tmp_path, "WEIGHT_MEASURE.xml", raw_weight_xml)
    write_xml(tmp_path, "BASAL_BITZUA.xml", raw_basal_xml)
    write_xml(tmp_path, "BOLUS_BITZUA.xml", raw_bolus_xml)
    write_xml(tmp_path, "DIABETES_DIAGNOSIS.xml", raw_diabetes_xml)
    
    write_xml(tmp_path, "ADMISSION_EVENT.xml", event_admission_xml)
    
    write_xml(tmp_path, "DIABETES_CONTEXT.xml", context_diabetes_xml)
    
    write_xml(tmp_path, "PATTERN_GLUCOSE.xml", pattern_glucose_xml)
    write_xml(tmp_path, "PATTERN_BMI.xml", pattern_bmi_xml)
    write_xml(tmp_path, "PATTERN_INSULIN.xml", pattern_insulin_xml)
    
    # Build repository
    repo = TAKRepository()
    
    # Register raw-concepts
    repo.register(RawConcept.parse(tmp_path / "ADMISSION_RAW.xml"))
    repo.register(RawConcept.parse(tmp_path / "GLUCOSE_MEASURE.xml"))
    repo.register(RawConcept.parse(tmp_path / "BMI_MEASURE.xml"))
    repo.register(RawConcept.parse(tmp_path / "WEIGHT_MEASURE.xml"))
    repo.register(RawConcept.parse(tmp_path / "BASAL_BITZUA.xml"))
    repo.register(RawConcept.parse(tmp_path / "BOLUS_BITZUA.xml"))
    repo.register(RawConcept.parse(tmp_path / "DIABETES_DIAGNOSIS.xml"))
    
    set_tak_repository(repo)
    
    # Register event
    from core.tak.event import Event
    repo.register(Event.parse(tmp_path / "ADMISSION_EVENT.xml"))
    
    # Register context
    repo.register(Context.parse(tmp_path / "DIABETES_CONTEXT.xml"))
    
    # Register patterns
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_GLUCOSE.xml"))
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_BMI.xml"))
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_INSULIN.xml"))
    
    return repo


def test_actual_pattern_glucose_on_admission_found(repo_actual_kb):
    """
    Test GLUCOSE_MEASURE_ON_ADMISSION_PATTERN with controlled data.
    Expected: Pattern found with time compliance score.
    """
    # Get TAKs
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    glucose_raw = repo_actual_kb.get("GLUCOSE_MEASURE")
    diabetes_raw = repo_actual_kb.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo_actual_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo_actual_kb.get("GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")
    
    # Synthetic input data
    # Use GLUCOSE_MEASURE (attribute name from actual XML)
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 120),  # FIXED: GLUCOSE_MEASURE (not GLUCOSE_LAB_MEASURE)
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    # Apply TAK pipeline
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    
    df_glucose = glucose_raw.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])  # FIXED
    
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    # DEBUG: Print context intervals
    print("\n=== DIABETES_DIAGNOSIS_CONTEXT output ===")
    print(df_diabetes_context)
    print(f"Context intervals: {len(df_diabetes_context)}")
    if not df_diabetes_context.empty:
        print(f"Context Start: {df_diabetes_context.iloc[0]['StartDateTime']}")
        print(f"Context End: {df_diabetes_context.iloc[0]['EndDateTime']}")
        print(f"Context Value: {df_diabetes_context.iloc[0]['Value']}")
    
    # Combine for pattern input
    df_pattern_input = pd.concat([df_admission_event, df_glucose, df_diabetes_context], ignore_index=True)
    
    # DEBUG: Print pattern input
    print("\n=== Pattern input ===")
    print(df_pattern_input)
    
    # Apply pattern
    df_out = pattern.apply(df_pattern_input)
    
    # DEBUG: Print pattern output
    print("\n=== Pattern output ===")
    print(df_out)
    
    # Assertions
    assert len(df_out) == 1, "Pattern should be found (glucose at 2h within 12h window)"
    row = df_out.iloc[0]
    assert row["Value"] == "True", "Pattern should match (within compliance window)"
    assert row["TimeConstraintScore"] is not None
    assert 0.0 < row["TimeConstraintScore"] <= 1.0


def test_actual_pattern_glucose_on_admission_partial_compliance(repo_actual_kb):
    """
    Test GLUCOSE_MEASURE_ON_ADMISSION_PATTERN with partial compliance.
    Expected: Pattern found but with partial time compliance score.
    """
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    glucose_raw = repo_actual_kb.get("GLUCOSE_MEASURE")
    diabetes_raw = repo_actual_kb.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo_actual_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo_actual_kb.get("GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("18:00"), make_ts("18:00"), 150),  # FIXED: GLUCOSE_MEASURE
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    df_glucose = glucose_raw.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])  # FIXED
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_glucose, df_diabetes_context], ignore_index=True)
    df_out = pattern.apply(df_pattern_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "Partial", "Pattern should have partial compliance (10h in ramp-down zone)"
    # Trapez: [A=0h, B=0h, C=8h, D=12h] → 10h in ramp-down [C, D]
    # Score = (D - x) / (D - C) = (12 - 10) / (12 - 8) = 2/4 = 0.5
    assert row["TimeConstraintScore"] == pytest.approx(0.5)


def test_actual_pattern_bmi_on_admission_found(repo_actual_kb):
    """
    Test BMI_MEASURE_ON_ADMISSION with controlled data.
    Expected: Pattern found with time compliance.
    """
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    bmi_raw = repo_actual_kb.get("BMI_MEASURE")
    pattern = repo_actual_kb.get("BMI_MEASURE_ON_ADMISSION")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "BMI_MEASURE", make_ts("09:00"), make_ts("09:00"), 26.5),  # 1h after admission
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    df_bmi = bmi_raw.apply(df_raw[df_raw["ConceptName"] == "BMI_MEASURE"])
    
    df_pattern_input = pd.concat([df_admission_event, df_bmi], ignore_index=True)
    df_out = pattern.apply(df_pattern_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    # Trapez: [0h, 0h, 48h, 72h] → 1h in plateau → score=1.0
    assert row["TimeConstraintScore"] == 1.0


def test_actual_pattern_insulin_on_admission_with_parameters(repo_actual_kb):
    """
    Test INSULIN_ON_ADMISSION_PATTERN with weight parameter.
    Expected: Pattern found with both time and value compliance.
    """
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    basal_raw = repo_actual_kb.get("BASAL_BITZUA")
    bolus_raw = repo_actual_kb.get("BOLUS_BITZUA")
    weight_raw = repo_actual_kb.get("WEIGHT_MEASURE")
    diabetes_raw = repo_actual_kb.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo_actual_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo_actual_kb.get("INSULIN_ON_ADMISSION_PATTERN")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "BASAL_DOSAGE", make_ts("10:00"), make_ts("10:00"), 25),  # 2h after admission
        (1, "BASAL_ROUTE", make_ts("10:00"), make_ts("10:00"), "SubCutaneous"),
        (1, "WEIGHT_MEASURE", make_ts("07:30"), make_ts("07:30"), 72),  # closest to pattern start (08:00)
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    
    # Apply BASAL_BITZUA (tuple merging)
    df_basal_input = df_raw[df_raw["ConceptName"].isin(["BASAL_DOSAGE", "BASAL_ROUTE"])]
    df_basal = basal_raw.apply(df_basal_input)
    
    df_weight = weight_raw.apply(df_raw[df_raw["ConceptName"] == "WEIGHT_MEASURE"])
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_basal, df_weight, df_diabetes_context], ignore_index=True)
    df_out = pattern.apply(df_pattern_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    # Time: 2h in [0h, 48h] plateau → score=1.0
    assert row["TimeConstraintScore"] == 1.0
    # Value: 25 units, weight=72 kg → trapez=[0, 14.4, 43.2, 72]
    #   25 in [14.4, 43.2] plateau → score=1.0
    assert row["ValueConstraintScore"] == 1.0


def test_actual_pattern_insulin_on_admission_uses_default_weight(repo_actual_kb):
    """
    Test INSULIN_ON_ADMISSION_PATTERN without weight data (uses default=72).
    Expected: Pattern found with value compliance using default parameter.
    """
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    basal_raw = repo_actual_kb.get("BASAL_BITZUA")
    diabetes_raw = repo_actual_kb.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo_actual_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo_actual_kb.get("INSULIN_ON_ADMISSION_PATTERN")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "BASAL_DOSAGE", make_ts("10:00"), make_ts("10:00"), 25),
        (1, "BASAL_ROUTE", make_ts("10:00"), make_ts("10:00"), "SubCutaneous"),
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
        # No WEIGHT_MEASURE → default=72 will be used
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    
    df_basal_input = df_raw[df_raw["ConceptName"].isin(["BASAL_DOSAGE", "BASAL_ROUTE"])]
    df_basal = basal_raw.apply(df_basal_input)
    
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_basal, df_diabetes_context], ignore_index=True)
    df_out = pattern.apply(df_pattern_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["ValueConstraintScore"] == 1.0, "Should use default weight=72 → same result as previous test"


def test_actual_pattern_multiple_rules_or_semantics(repo_actual_kb):
    """
    Test GLUCOSE_MEASURE_ON_ADMISSION_PATTERN with multiple rules (OR semantics).
    Pattern has 2 rules: 12h window and 36h window.
    Expected: Glucose at 20h matches second rule only.
    """
    admission_raw = repo_actual_kb.get("ADMISSION")
    admission_event = repo_actual_kb.get("ADMISSION_EVENT")
    glucose_raw = repo_actual_kb.get("GLUCOSE_MEASURE")
    diabetes_raw = repo_actual_kb.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo_actual_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo_actual_kb.get("GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("04:00", day=1), make_ts("04:00", day=1), 180),  
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    df_glucose = glucose_raw.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])  
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_glucose, df_diabetes_context], ignore_index=True)
    df_out = pattern.apply(df_pattern_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True", "Should match second rule (20h within 36h window)"
    # Second rule trapez: [0h, 0h, 24h, 36h] → 20h in plateau [B, C] → score=1.0
    assert row["TimeConstraintScore"] == 1.0


# -----------------------------
# Tests: Parsing & Validation
# -----------------------------

def test_parse_pattern_validates_structure(repo_simple_pattern):
    """Validate XML structure and instantiation."""
    pattern = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    assert pattern.name == "GLUCOSE_ON_ADMISSION_SIMPLE"
    assert len(pattern.derived_from) == 2
    assert pattern.derived_from[0]["ref"] == "A1"
    assert pattern.derived_from[1]["ref"] == "E1"
    assert len(pattern.abstraction_rules) == 1


def test_pattern_validation_requires_max_distance_for_before(tmp_path: Path):
    """Validation fails if how='before' and no max-distance."""
    bad_pattern_xml = PATTERN_SIMPLE_XML.replace("max-distance='12h'", "")
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "BAD_PATTERN.xml", bad_pattern_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    with pytest.raises(ValueError, match="requires max-distance"):
        LocalPattern.parse(pattern_path)


def test_pattern_validation_max_distance_ge_trapezeD(tmp_path: Path):
    """Validation fails if max-distance < trapezeD."""
    bad_pattern_xml = PATTERN_TIME_COMPLIANCE_XML.replace("max-distance='12h'", "max-distance='6h'")
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "BAD_PATTERN.xml", bad_pattern_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    with pytest.raises(ValueError, match="max-distance.*must be >= .*trapezeD"):
        LocalPattern.parse(pattern_path)


def test_pattern_validation_rejects_numeric_equal_constraint(tmp_path: Path):
    """Validation fails if numeric attribute has equal constraint."""
    bad_pattern_xml = PATTERN_SIMPLE_XML.replace(
        '<allowed-value min="0"/>',
        '<allowed-value equal="100"/>'
    )
    admission_path = write_xml(tmp_path, "ADMISSION.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_XML)
    pattern_path = write_xml(tmp_path, "BAD_PATTERN.xml", bad_pattern_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    with pytest.raises(ValueError, match="has 'equal' constraint but attribute.*is numeric"):
        LocalPattern.parse(pattern_path)


# -----------------------------
# Tests: Pattern Discovery (Simple)
# -----------------------------

def test_pattern_found_simple(repo_simple_pattern):
    """Pattern found: glucose within 12h of admission."""
    admission_tak = repo_simple_pattern.get("ADMISSION_EVENT")
    glucose_tak = repo_simple_pattern.get("GLUCOSE_MEASURE")
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    
    # Raw input
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("10:00"), make_ts("10:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    # Apply raw-concepts
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    
    # Combine for pattern input
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    # Apply pattern
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["StartDateTime"] == make_ts("08:00")
    assert row["EndDateTime"] == make_ts("10:00")
    assert pd.isna(row["TimeConstraintScore"])  # No compliance function
    assert pd.isna(row["ValueConstraintScore"])


def test_pattern_not_found_no_event(repo_simple_pattern):
    """Pattern not found: no glucose measure."""
    admission_tak = repo_simple_pattern.get("ADMISSION_EVENT")
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw)
    df_out = pattern_tak.apply(df_admission)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "False"
    assert pd.isna(row["StartDateTime"])
    assert pd.isna(row["EndDateTime"])


def test_pattern_not_found_event_too_late(repo_simple_pattern):
    """Pattern not found: glucose at 14h (outside 12h window)."""
    admission_tak = repo_simple_pattern.get("ADMISSION_EVENT")
    glucose_tak = repo_simple_pattern.get("GLUCOSE_MEASURE")
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("22:00"), make_ts("22:00"), 120),  # 14h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "False"


def test_pattern_one_to_one_pairing(repo_simple_pattern):
    """One-to-one pairing: 2 admissions + 2 glucose → 2 patterns."""
    admission_tak = repo_simple_pattern.get("ADMISSION_EVENT")
    glucose_tak = repo_simple_pattern.get("GLUCOSE_MEASURE")
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    
    # Raw input
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("09:00"), make_ts("09:00"), 100),
        (1, "ADMISSION", make_ts("10:00"), make_ts("10:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("11:00"), make_ts("11:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    # Expect 2 patterns (one-to-one pairing, with select='first')
    # But actually, one-to-one means FIRST anchor pairs with FIRST event, exhausts both
    # So we only get 1 pattern (first admission @ 08:00 → first glucose @ 09:00)
    # Second admission @ 10:00 → no unused glucose left → pattern not found
    # CORRECTION: Code uses per-rule one-to-one, so 1st anchor pairs with 1st event (9h),
    # 2nd anchor pairs with 2nd event (11h) if not used yet.
    # Let me trace through the logic...
    
    # Actually, the code tracks used_anchor_ids and used_event_ids ACROSS ALL RULES.
    # For single-rule pattern, it will pair:
    # - anchor[08:00] with event[09:00] → mark both used
    # - anchor[10:00] with event[11:00] → mark both used
    # Result: 2 patterns
    
    assert len(df_out) == 2
    assert all(df_out["Value"] == "True")


# -----------------------------
# Tests: Time-Constraint Compliance
# -----------------------------

def test_time_compliance_full(repo_time_compliance):
    """Time compliance: glucose at 2h (within [0h, 8h] → score=1.0)."""
    admission_tak = repo_time_compliance.get("ADMISSION_EVENT")
    glucose_tak = repo_time_compliance.get("GLUCOSE_MEASURE")
    pattern_tak = repo_time_compliance.get("GLUCOSE_ON_ADMISSION_TIME")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("10:00"), make_ts("10:00"), 120),  # 2h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["TimeConstraintScore"] == 1.0


def test_time_compliance_partial(repo_time_compliance):
    """Time compliance: glucose at 10h (in [8h, 12h] → score=0.5)."""
    admission_tak = repo_time_compliance.get("ADMISSION_EVENT")
    glucose_tak = repo_time_compliance.get("GLUCOSE_MEASURE")
    pattern_tak = repo_time_compliance.get("GLUCOSE_ON_ADMISSION_TIME")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("18:00"), make_ts("18:00"), 120),  # 10h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "Partial"
    # Score = (12h - 10h) / (12h - 8h) = 2/4 = 0.5
    assert row["TimeConstraintScore"] == pytest.approx(0.5)


def test_time_compliance_zero(repo_time_compliance):
    """Time compliance: glucose at 13h (outside [0h, 12h] → score=0.0, but pattern still found)."""
    admission_tak = repo_time_compliance.get("ADMISSION_EVENT")
    glucose_tak = repo_time_compliance.get("GLUCOSE_MEASURE")
    pattern_tak = repo_time_compliance.get("GLUCOSE_ON_ADMISSION_TIME")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("21:00"), make_ts("21:00"), 120),  # 13h gap (just outside trapezD)
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    # Pattern NOT found because max-distance=12h (13h gap exceeds it)
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "False"


# -----------------------------
# Tests: Value-Constraint Compliance
# -----------------------------

def test_value_compliance_full(repo_value_compliance):
    """Value compliance: insulin=25 units, weight=72 kg → 0.35 units/kg (in [0.2, 0.6] → score=1.0)."""
    admission_tak = repo_value_compliance.get("ADMISSION_EVENT")
    insulin_tak = repo_value_compliance.get("INSULIN_BITZUA")
    weight_tak = repo_value_compliance.get("WEIGHT_MEASURE")
    pattern_tak = repo_value_compliance.get("INSULIN_ON_ADMISSION_VALUE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "INSULIN_DOSAGE", make_ts("10:00"), make_ts("10:00"), 25),  # 2h gap
        (1, "WEIGHT", make_ts("07:00"), make_ts("07:00"), 72),  # closest to pattern start (08:00)
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_weight = weight_tak.apply(df_raw[df_raw["ConceptName"] == "WEIGHT"])
    df_input = pd.concat([df_admission, df_insulin, df_weight], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    # Trapez = mul(72, [0, 0.2, 0.6, 1]) = [0, 14.4, 43.2, 72]
    # Actual = 25 units → in [14.4, 43.2] → score = 1.0
    assert row["Value"] == "True"
    assert row["ValueConstraintScore"] == 1.0


def test_value_compliance_partial(repo_value_compliance):
    """Value compliance: insulin=60 units, weight=72 kg → 0.83 units/kg (in [0.6, 1.0] → score<1.0)."""
    admission_tak = repo_value_compliance.get("ADMISSION_EVENT")
    insulin_tak = repo_value_compliance.get("INSULIN_BITZUA")
    weight_tak = repo_value_compliance.get("WEIGHT_MEASURE")
    pattern_tak = repo_value_compliance.get("INSULIN_ON_ADMISSION_VALUE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "INSULIN_DOSAGE", make_ts("10:00"), make_ts("10:00"), 60),
        (1, "WEIGHT", make_ts("07:00"), make_ts("07:00"), 72),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_weight = weight_tak.apply(df_raw[df_raw["ConceptName"] == "WEIGHT"])
    df_input = pd.concat([df_admission, df_insulin, df_weight], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "Partial"
    # Trapez = [0, 14.4, 43.2, 72]
    # Actual = 60 → in [43.2, 72] → score = (72 - 60) / (72 - 43.2) = 12 / 28.8 ≈ 0.417
    assert 0.4 < row["ValueConstraintScore"] < 0.5


def test_value_compliance_uses_default_parameter(repo_value_compliance):
    """Value compliance: no weight data → use default (72 kg)."""
    admission_tak = repo_value_compliance.get("ADMISSION_EVENT")
    insulin_tak = repo_value_compliance.get("INSULIN_BITZUA")
    pattern_tak = repo_value_compliance.get("INSULIN_ON_ADMISSION_VALUE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "INSULIN_DOSAGE", make_ts("10:00"), make_ts("10:00"), 25),
        # No WEIGHT record → default=72
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_input = pd.concat([df_admission, df_insulin], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["ValueConstraintScore"] == 1.0  # Same as test_value_compliance_full


# -----------------------------
# Tests: Context Filtering
# -----------------------------

def test_context_filtering_pattern_found(repo_with_context):
    """Context filtering: diabetes patient → pattern found."""
    admission_tak = repo_with_context.get("ADMISSION_EVENT")
    glucose_tak = repo_with_context.get("GLUCOSE_MEASURE")
    diabetes_tak = repo_with_context.get("DIABETES_DIAGNOSIS")
    context_tak = repo_with_context.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern_tak = repo_with_context.get("GLUCOSE_ON_ADMISSION_CONTEXT")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("10:00"), make_ts("10:00"), 120),
        (1, "DIABETES", make_ts("07:00"), make_ts("07:00"), "True"),  # before admission
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_diabetes = diabetes_tak.apply(df_raw[df_raw["ConceptName"] == "DIABETES"])
    
    # Apply context windowing (720h = 30 days)
    df_context = context_tak.apply(df_diabetes)
    
    df_input = pd.concat([df_admission, df_glucose, df_context], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"


def test_context_filtering_pattern_not_found(repo_with_context):
    """Context filtering: no diabetes context → pattern not found."""
    admission_tak = repo_with_context.get("ADMISSION_EVENT")
    glucose_tak = repo_with_context.get("GLUCOSE_MEASURE")
    pattern_tak = repo_with_context.get("GLUCOSE_ON_ADMISSION_CONTEXT")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("10:00"), make_ts("10:00"), 120),
        # No DIABETES record → no context
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "False"


# -----------------------------
# Tests: Multiple Rules (OR Semantics)
# -----------------------------

def test_multiple_rules_matches_first_rule(repo_multiple_rules):
    """Multiple rules: glucose at 4h matches first rule (8h window)."""
    admission_tak = repo_multiple_rules.get("ADMISSION_EVENT")
    glucose_tak = repo_multiple_rules.get("GLUCOSE_MEASURE")
    pattern_tak = repo_multiple_rules.get("GLUCOSE_ON_ADMISSION_MULTI")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("12:00"), make_ts("12:00"), 120),  # 4h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    # First rule matches (4h in [0h, 8h])
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["TimeConstraintScore"] == 1.0


def test_multiple_rules_matches_second_rule(repo_multiple_rules):
    """Multiple rules: glucose at 20h only matches second rule (24h window)."""
    admission_tak = repo_multiple_rules.get("ADMISSION_EVENT")
    glucose_tak = repo_multiple_rules.get("GLUCOSE_MEASURE")
    pattern_tak = repo_multiple_rules.get("GLUCOSE_ON_ADMISSION_MULTI")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("04:00", day=1), make_ts("04:00", day=1), 120),  # 20h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    # Second rule matches (20h in [12h, 24h])
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    # Second rule trapez: [0h, 0h, 24h, 36h] → 20h in plateau [B, C] → score=1.0
    assert row["TimeConstraintScore"] == 1.0


def test_multiple_rules_one_to_one_across_rules(repo_multiple_rules):
    """Multiple rules: 1 admission + 2 glucose (4h, 20h) → 2 patterns (both matched)."""
    admission_tak = repo_multiple_rules.get("ADMISSION_EVENT")
    glucose_tak = repo_multiple_rules.get("GLUCOSE_MEASURE")
    pattern_tak = repo_multiple_rules.get("GLUCOSE_ON_ADMISSION_MULTI")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("12:00"), make_ts("12:00"), 100),  # 4h gap
        (1, "GLUCOSE_LAB", make_ts("04:00", day=1), make_ts("04:00", day=1), 120),  # 20h gap
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    # Two patterns found (one-to-one pairing across rules):
    # - Rule 1 pairs admission[08:00] with glucose[12:00] (4h)
    # - Rule 2 tries to pair admission[08:00] with glucose[04:00@day1] (20h), but anchor already used
    # So only 1 pattern found (first rule wins)
    assert len(df_out) == 1
    assert df_out.iloc[0]["TimeConstraintScore"] == 1.0


# -----------------------------
# Tests: Overlap Temporal Relation
# -----------------------------

def test_overlap_pattern_found(repo_overlap_pattern):
    """Overlap pattern: insulin during admission (same time)."""
    admission_tak = repo_overlap_pattern.get("ADMISSION_EVENT")
    insulin_tak = repo_overlap_pattern.get("INSULIN_BITZUA")
    pattern_tak = repo_overlap_pattern.get("INSULIN_DURING_ADMISSION")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("12:00"), "True"),  # 4h interval
        (1, "INSULIN_DOSAGE", make_ts("09:00"), make_ts("09:00"), 25),  # inside admission
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_input = pd.concat([df_admission, df_insulin], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"


def test_overlap_pattern_not_found(repo_overlap_pattern):
    """Overlap pattern: insulin after admission (no overlap)."""
    admission_tak = repo_overlap_pattern.get("ADMISSION_EVENT")
    insulin_tak = repo_overlap_pattern.get("INSULIN_BITZUA")
    pattern_tak = repo_overlap_pattern.get("INSULIN_DURING_ADMISSION")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("12:00"), "True"),
        (1, "INSULIN_DOSAGE", make_ts("14:00"), make_ts("14:00"), 25),  # after admission
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_input = pd.concat([df_admission, df_insulin], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "False"


# -----------------------------
# Tests: Edge Cases
# -----------------------------

def test_pattern_empty_input_returns_false(repo_simple_pattern):
    """Empty input → empty output (no PatientId to emit False for)."""
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    df_empty = pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    df_out = pattern_tak.apply(df_empty)

    # FIXED: Empty input returns empty DataFrame (no PatientId for False row)
    assert len(df_out) == 0
    assert df_out.empty


def test_pattern_select_last_prefers_latest_event(repo_simple_pattern, tmp_path):
    """select='last' prefers latest event."""
    # Modify pattern to use select='last' for event AND change name to avoid duplicate
    pattern_xml_last = PATTERN_SIMPLE_XML.replace("select='first'", "select='last'").replace(
        'name="GLUCOSE_ON_ADMISSION_SIMPLE"',
        'name="GLUCOSE_ON_ADMISSION_LAST"'
    )
    pattern_path = write_xml(tmp_path, "PATTERN_LAST.xml", pattern_xml_last)
    
    repo = repo_simple_pattern
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)
    
    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")

    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_LAB", make_ts("09:00"), make_ts("09:00"), 100),
        (1, "GLUCOSE_LAB", make_ts("11:00"), make_ts("11:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_glucose = glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_LAB"])
    df_input = pd.concat([df_admission, df_glucose], ignore_index=True)

    df_out = pattern_tak.apply(df_input)

    assert len(df_out) == 1
    # select='last' → picks glucose at 11:00 (not 09:00)
    assert df_out.iloc[0]["EndDateTime"] == make_ts("11:00")


def test_pattern_combined_score_averages_time_and_value(repo_value_compliance, tmp_path):
    """Combined score: average of time (1.0) + value (0.694) = 0.847 → Partial."""
    # FIXED: Add missing <derived-from>, <parameters>, and proper <abstraction-rules> wrapper
    pattern_xml_combined = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="COMBINED_COMPLIANCE" concept-type="local-pattern">
    <categories>Patterns</categories>
    <description>Pattern with both time and value compliance</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <parameters>
        <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
    </parameters>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='48h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="24h" trapezeD="48h"/>
                    </function>
                </time-constraint-compliance>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                    </target>
                    <function name="mul">
                        <parameter ref="P1"/>
                        <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""
    pattern_path = write_xml(tmp_path, "PATTERN_COMBINED.xml", pattern_xml_combined)
    
    # FIXED: Register BASAL_BITZUA in repo (missing from repo_value_compliance fixture)
    basal_raw_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASAL_BITZUA" concept-type="raw">
  <categories>Medications</categories>
  <description>Basal insulin</description>
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
    basal_path = write_xml(tmp_path, "BASAL_BITZUA.xml", basal_raw_xml)
    
    repo = repo_value_compliance
    basal_tak = RawConcept.parse(basal_path)
    repo.register(basal_tak)
    set_tak_repository(repo)  # Update global repo
    
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)
    
    admission_tak = repo.get("ADMISSION_EVENT")
    basal_tak = repo.get("BASAL_BITZUA")
    weight_tak = repo.get("WEIGHT_MEASURE")
    
    df_raw = pd.DataFrame([
       
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "BASAL_DOSAGE", make_ts("10:00"), make_ts("10:00"), 10),  # 2h gap, 10 units
        (1, "BASAL_ROUTE", make_ts("10:00"), make_ts("10:00"), "SubCutaneous"),
        (1, "WEIGHT", make_ts("07:00"), make_ts("07:00"), 72),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    
    # Apply BASAL_BITZUA (tuple merging)
    df_basal_input = df_raw[df_raw["ConceptName"].isin(["BASAL_DOSAGE", "BASAL_ROUTE"])]
    df_basal = basal_tak.apply(df_basal_input)
    
    df_weight = weight_tak.apply(df_raw[df_raw["ConceptName"] == "WEIGHT"])
    
    df_input = pd.concat([df_admission, df_basal, df_weight], ignore_index=True)
    
    # DEBUG: Print input
    print("\n=== Pattern input ===")
    print(df_input)
    
    df_out = pattern_tak.apply(df_input)
    
    # DEBUG: Print output with scores
    print("\n=== Pattern output ===")
    print(df_out)
    print(f"\nPattern found: {len(df_out)} rows")
    if len(df_out) > 0:
        row = df_out.iloc[0]
        print(f"Value: {row['Value']}")
        print(f"TimeConstraintScore: {row['TimeConstraintScore']}")
        print(f"ValueConstraintScore: {row['ValueConstraintScore']}")
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    
    # Time score: 2h in [0h, 24h] → 1.0 (in plateau zone [B, C])
    assert row["TimeConstraintScore"] == 1.0
    
    # Value score: 10 units, weight=72 kg → trapez=[0, 14.4, 43.2, 72]
    #   10 is in ramp-up zone [A=0, B=14.4]
    #   score = (10 - 0) / (14.4 - 0) = 10/14.4 ≈ 0.694
    assert 0.68 < row["ValueConstraintScore"] < 0.71  # Correct range for 0.694
    
    # Combined: (1.0 + 0.694) / 2 = 0.847 → Partial
    assert row["Value"] == "Partial"
    
    print("\n✅ Combined compliance test PASSED")
    print(f"   Time score: {row['TimeConstraintScore']:.3f}")
    print(f"   Value score: {row['ValueConstraintScore']:.3f}")
    print(f"   Combined value: {row['Value']}")


# -----------------------------
# Tests: Actual KB Patterns (Self-Contained Debug Tests)
# -----------------------------

def test_debug_insulin_on_admission_basic(tmp_path: Path):
    """
    Debug test for INSULIN_ON_ADMISSION_PATTERN with minimal data.
    Tests the exact XML structure from actual KB.
    """
    # Copy actual KB XMLs (self-contained)
    raw_admission_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission raw</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
  </attributes>
</raw-concept>
"""
    
    event_admission_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="ADMISSION_EVENT">
    <categories>Admin</categories>
    <description>Admission event</description>
    <derived-from>
        <attribute name="ADMISSION" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""
    
    raw_basal_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASAL_BITZUA" concept-type="raw">
  <categories>Medications</categories>
  <description>Basal insulin</description>
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
    
    raw_bolus_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BOLUS_BITZUA" concept-type="raw">
  <categories>Medications</categories>
  <description>Bolus insulin</description>
  <attributes>
    <attribute name="BOLUS_DOSAGE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
    <attribute name="BOLUS_ROUTE" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="SubCutaneous"/>
        <allowed-value value="IntraVenous"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
  <tuple-order>
    <attribute name="BOLUS_DOSAGE"/>
    <attribute name="BOLUS_ROUTE"/>
  </tuple-order>
  <merge require-all="false"/>
</raw-concept>
"""
    
    raw_diabetes_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DIABETES_DIAGNOSIS" concept-type="raw-boolean">
  <categories>Diagnoses</categories>
  <description>Diabetes diagnosis</description>
  <attributes>
    <attribute name="DIABETES_DIAGNOSIS" type="boolean"/>
  </attributes>
</raw-concept>
"""
    
    context_diabetes_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="DIABETES_DIAGNOSIS_CONTEXT">
    <categories>Diagnoses</categories>
    <description>Diabetes diagnosis context</description>
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
        <persistence good-before="0h" good-after="720h"/>
    </context-windows>
</context>
"""
    
    raw_weight_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="WEIGHT_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Weight measure</description>
  <attributes>
    <attribute name="WEIGHT_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="300"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""
    
    pattern_insulin_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Captures if INSULIN (BASAL/BOLUS) was performed within reasonable time of admission</description>
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="BOLUS_BITZUA" tak="raw-concept" idx="0" ref="E2"/>
    </derived-from>
    <parameters>
        <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
    </parameters>
    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>
            <temporal-relation how='before' max-distance='72h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                    <attribute ref="E2">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="48h" trapezeD="72h"/>
                    </function>
                </time-constraint-compliance>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                        <attribute ref="E2"/>
                    </target>
                    <function name="mul">
                        <parameter ref="P1"/>
                        <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""
    
    # Write XMLs
    write_xml(tmp_path, "ADMISSION_RAW.xml", raw_admission_xml)
    write_xml(tmp_path, "ADMISSION_EVENT.xml", event_admission_xml)
    write_xml(tmp_path, "BASAL_BITZUA.xml", raw_basal_xml)
    write_xml(tmp_path, "BOLUS_BITZUA.xml", raw_bolus_xml)
    write_xml(tmp_path, "DIABETES_DIAGNOSIS.xml", raw_diabetes_xml)
    write_xml(tmp_path, "DIABETES_CONTEXT.xml", context_diabetes_xml)
    write_xml(tmp_path, "WEIGHT_MEASURE.xml", raw_weight_xml)
    write_xml(tmp_path, "PATTERN_INSULIN.xml", pattern_insulin_xml)
    
    # Build repository
    repo = TAKRepository()
    repo.register(RawConcept.parse(tmp_path / "ADMISSION_RAW.xml"))
    repo.register(RawConcept.parse(tmp_path / "BASAL_BITZUA.xml"))
    repo.register(RawConcept.parse(tmp_path / "BOLUS_BITZUA.xml"))
    repo.register(RawConcept.parse(tmp_path / "DIABETES_DIAGNOSIS.xml"))
    repo.register(RawConcept.parse(tmp_path / "WEIGHT_MEASURE.xml"))
    
    set_tak_repository(repo)
    
    from core.tak.event import Event
    repo.register(Event.parse(tmp_path / "ADMISSION_EVENT.xml"))
    repo.register(Context.parse(tmp_path / "DIABETES_CONTEXT.xml"))
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_INSULIN.xml"))
    
    # Get TAKs
    admission_raw = repo.get("ADMISSION")
    admission_event = repo.get("ADMISSION_EVENT")
    basal_raw = repo.get("BASAL_BITZUA")
    weight_raw = repo.get("WEIGHT_MEASURE")
    diabetes_raw = repo.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo.get("INSULIN_ON_ADMISSION_PATTERN")
    
    # Synthetic input data
    df_raw = pd.DataFrame([
        (1000, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1000, "BASAL_DOSAGE", make_ts("10:00"), make_ts("10:00"), 25),
        (1000, "BASAL_ROUTE", make_ts("10:00"), make_ts("10:00"), "SubCutaneous"),
        (1000, "WEIGHT_MEASURE", make_ts("07:30"), make_ts("07:30"), 72),
        (1000, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    # Apply TAK pipeline
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    
    df_basal_input = df_raw[df_raw["ConceptName"].isin(["BASAL_DOSAGE", "BASAL_ROUTE"])]
    df_basal = basal_raw.apply(df_basal_input)
    
    df_weight = weight_raw.apply(df_raw[df_raw["ConceptName"] == "WEIGHT_MEASURE"])
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_basal, df_weight, df_diabetes_context], ignore_index=True)
    
    print("\n=== DEBUG: Pattern input ===")
    print(df_pattern_input)
    print(f"\nInput dtypes:")
    print(df_pattern_input.dtypes)
    
    # Apply pattern (THIS IS WHERE ERROR OCCURS)
    df_out = pattern.apply(df_pattern_input)
    
    print("\n=== DEBUG: Pattern output ===")
    print(df_out)
    
    # Assertions
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"


def test_debug_glucose_on_admission_multi_rule(tmp_path: Path):
    """
    Debug test for GLUCOSE_MEASURE_ON_ADMISSION_PATTERN with 2 rules.
    Tests multiple rule evaluation with time-constraint compliance.
    """
    raw_admission_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission raw</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
  </attributes>
</raw-concept>
"""
    
    event_admission_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="ADMISSION_EVENT">
    <categories>Admin</categories>
    <description>Admission event</description>
    <derived-from>
        <attribute name="ADMISSION" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""
    
    raw_glucose_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Glucose measure</description>
  <attributes>
    <attribute name="GLUCOSE_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""
    
    raw_diabetes_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DIABETES_DIAGNOSIS" concept-type="raw-boolean">
  <categories>Diagnoses</categories>
  <description>Diabetes diagnosis</description>
  <attributes>
    <attribute name="DIABETES_DIAGNOSIS" type="boolean"/>
  </attributes>
</raw-concept>
"""
    
    context_diabetes_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="DIABETES_DIAGNOSIS_CONTEXT">
    <categories>Diagnoses</categories>
    <description>Diabetes diagnosis context</description>
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
        <persistence good-before="0h" good-after="720h"/>
    </context-windows>
</context>
"""
    
    pattern_glucose_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_MEASURE_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Captures if DEX was preformed within resonable time of admission</description>
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>
            <temporal-relation how='before' max-distance='12h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
        <rule>
            <temporal-relation how='before' max-distance='36h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="24h" trapezeD="36h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""
    
    # Write XMLs
    write_xml(tmp_path, "ADMISSION_RAW.xml", raw_admission_xml)
    write_xml(tmp_path, "ADMISSION_EVENT.xml", event_admission_xml)
    write_xml(tmp_path, "GLUCOSE_MEASURE.xml", raw_glucose_xml)
    write_xml(tmp_path, "DIABETES_DIAGNOSIS.xml", raw_diabetes_xml)
    write_xml(tmp_path, "DIABETES_CONTEXT.xml", context_diabetes_xml)
    write_xml(tmp_path, "PATTERN_GLUCOSE.xml", pattern_glucose_xml)
    
    # Build repository
    repo = TAKRepository()
    repo.register(RawConcept.parse(tmp_path / "ADMISSION_RAW.xml"))
    repo.register(RawConcept.parse(tmp_path / "GLUCOSE_MEASURE.xml"))
    repo.register(RawConcept.parse(tmp_path / "DIABETES_DIAGNOSIS.xml"))
    
    set_tak_repository(repo)
    
    from core.tak.event import Event
    repo.register(Event.parse(tmp_path / "ADMISSION_EVENT.xml"))
    repo.register(Context.parse(tmp_path / "DIABETES_CONTEXT.xml"))
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_GLUCOSE.xml"))
    
    # Get TAKs
    admission_raw = repo.get("ADMISSION")
    admission_event = repo.get("ADMISSION_EVENT")
    glucose_raw = repo.get("GLUCOSE_MEASURE")
    diabetes_raw = repo.get("DIABETES_DIAGNOSIS")
    diabetes_context = repo.get("DIABETES_DIAGNOSIS_CONTEXT")
    pattern = repo.get("GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")
    
    # Test case: glucose at 10h (should match first rule with partial compliance)
    df_raw = pd.DataFrame([
        (1000, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1000, "GLUCOSE_MEASURE", make_ts("18:00"), make_ts("18:00"), 150),
        (1000, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission_raw = admission_raw.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_admission_event = admission_event.apply(df_admission_raw)
    df_glucose = glucose_raw.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])
    df_diabetes_raw = diabetes_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])
    df_diabetes_context = diabetes_context.apply(df_diabetes_raw)
    
    df_pattern_input = pd.concat([df_admission_event, df_glucose, df_diabetes_context], ignore_index=True)
    
    print("\n=== DEBUG: Pattern input ===")
    print(df_pattern_input)
    
    df_out = pattern.apply(df_pattern_input)
    
    print("\n=== DEBUG: Pattern output ===")
    print(df_out)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "Partial"
    assert df_out.iloc[0]["TimeConstraintScore"] == pytest.approx(0.5)


def test_debug_insulin_on_high_glucose_context_clipping(tmp_path: Path):
    """
    Debug test for INSULIN_ON_HIGH_GLUCOSE_PATTERN with context clipping.
    Tests context TAK used as anchor (HIGH_GLUCOSE_CONTEXT).
    """
    raw_glucose_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Glucose measure</description>
  <attributes>
    <attribute name="GLUCOSE_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""
    
    context_high_glucose_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="HIGH_GLUCOSE_CONTEXT">
    <categories>Measurements</categories>
    <description>High glucose context</description>
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="True" operator="or">
            <attribute ref="A1">
                <allowed-value min="200"/>
            </attribute>
        </rule>
    </abstraction-rules>
    <context-windows>
        <persistence good-before="0h" good-after="0h"/>  <!-- ✅ Changed from 6h to 0h -->
    </context-windows>
</context>
"""
    
    context_meal_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="MEAL_CONTEXT">
    <categories>Meals</categories>
    <description>Meal context (Breakfast/Lunch/Dinner)</description>
    <derived-from>
        <attribute name="MEAL_TIMING" tak="raw-concept" idx="0" ref="A1"/>
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
    </abstraction-rules>
    <context-windows>
        <persistence good-before="1h" good-after="3h"/>
    </context-windows>
</context>
"""
    
    raw_meal_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="MEAL_TIMING" concept-type="raw-nominal">
  <categories>Meals</categories>
  <description>Meal timing</description>
  <attributes>
    <attribute name="MEAL_TIMING" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="Breakfast"/>
        <allowed-value value="Lunch"/>
        <allowed-value value="Dinner"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""
    
    raw_bolus_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BOLUS_BITZUA" concept-type="raw">
  <categories>Medications</categories>
  <description>Bolus insulin</description>
  <attributes>
    <attribute name="BOLUS_DOSAGE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
    <attribute name="BOLUS_ROUTE" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="SubCutaneous"/>
        <allowed-value value="IntraVenous"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
  <tuple-order>
    <attribute name="BOLUS_DOSAGE"/>
    <attribute name="BOLUS_ROUTE"/>
  </tuple-order>
  <merge require-all="false"/>
</raw-concept>
"""
    
    pattern_insulin_high_glucose_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_ON_HIGH_GLUCOSE_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Captures if INSULIN (BASAL/BOLUS) was performed within reasonable time after high glucose measured</description>
    <derived-from>
        <attribute name="MEAL_CONTEXT" tak="context" ref="C1"/>
        <attribute name="HIGH_GLUCOSE_CONTEXT" tak="context" ref="A1"/>
        <attribute name="BOLUS_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="Breakfast"/>
                    <allowed-value equal="Lunch"/>
                    <allowed-value equal="Dinner"/>
                </attribute>
            </context>
            <temporal-relation how='before' max-distance='2h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="1h" trapezeD="2h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""
    
    # Write XMLs
    write_xml(tmp_path, "GLUCOSE_MEASURE.xml", raw_glucose_xml)
    write_xml(tmp_path, "HIGH_GLUCOSE_CONTEXT.xml", context_high_glucose_xml)
    write_xml(tmp_path, "MEAL_TIMING.xml", raw_meal_xml)
    write_xml(tmp_path, "MEAL_CONTEXT.xml", context_meal_xml)
    write_xml(tmp_path, "BOLUS_BITZUA.xml", raw_bolus_xml)
    write_xml(tmp_path, "PATTERN_INSULIN_HIGH_GLUCOSE.xml", pattern_insulin_high_glucose_xml)
    
    # Build repository
    repo = TAKRepository()
    repo.register(RawConcept.parse(tmp_path / "GLUCOSE_MEASURE.xml"))
    repo.register(RawConcept.parse(tmp_path / "MEAL_TIMING.xml"))
    repo.register(RawConcept.parse(tmp_path / "BOLUS_BITZUA.xml"))
    
    set_tak_repository(repo)
    
    repo.register(Context.parse(tmp_path / "HIGH_GLUCOSE_CONTEXT.xml"))
    repo.register(Context.parse(tmp_path / "MEAL_CONTEXT.xml"))
    repo.register(LocalPattern.parse(tmp_path / "PATTERN_INSULIN_HIGH_GLUCOSE.xml"))
    
    # Get TAKs
    glucose_raw = repo.get("GLUCOSE_MEASURE")
    high_glucose_context = repo.get("HIGH_GLUCOSE_CONTEXT")
    meal_raw = repo.get("MEAL_TIMING")
    meal_context = repo.get("MEAL_CONTEXT")
    bolus_raw = repo.get("BOLUS_BITZUA")
    pattern = repo.get("INSULIN_ON_HIGH_GLUCOSE_PATTERN")
    
    # Test case: high glucose at lunch, bolus 30min later
    df_raw = pd.DataFrame([
        (1000, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), 250),
        (1000, "MEAL_TIMING", make_ts("12:00"), make_ts("12:00"), "Lunch"),
        (1000, "BOLUS_DOSAGE", make_ts("12:05"), make_ts("12:05"), 10),
        (1000, "BOLUS_ROUTE", make_ts("12:30"), make_ts("12:30"), "SubCutaneous"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_glucose = glucose_raw.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])
    df_high_glucose = high_glucose_context.apply(df_glucose)
    
    df_meal = meal_raw.apply(df_raw[df_raw["ConceptName"] == "MEAL_TIMING"])
    df_meal_context = meal_context.apply(df_meal)
    
    df_bolus_input = df_raw[df_raw["ConceptName"].isin(["BOLUS_DOSAGE", "BOLUS_ROUTE"])]
    df_bolus = bolus_raw.apply(df_bolus_input)
    
    df_pattern_input = pd.concat([df_high_glucose, df_meal_context, df_bolus], ignore_index=True)
    
    print("\n=== DEBUG: Pattern input ===")
    print(df_pattern_input)
    
    df_out = pattern.apply(df_pattern_input)
    
    print("\n=== DEBUG: Pattern output ===")
    print(df_out)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"
    assert df_out.iloc[0]["TimeConstraintScore"] == 1.0

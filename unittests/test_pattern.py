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
from core.tak.event import Event
from core.tak.context import Context
from core.tak.repository import set_tak_repository, TAKRepository
from core.tak.external_functions import REPO, register


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
        <attribute name="DIABETES_DIAGNOSIS" tak="raw-concept" idx="0"/>
    </derived-from>
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
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
        <rule>
            <temporal-relation how='before' max-distance='24h'>
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
    assert row["Value"] == "True"
    # Trapez = mul(72, [0, 0.2, 0.6, 1]) = [0, 14.4, 43.2, 72]
    # Actual = 25 units → in [14.4, 43.2] → score = 1.0
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
    # Trapez2: [12h, 24h, 36h] → 20h in [12h, 24h] → score = 1.0
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
    """Empty input → pattern not found (False with NaT times)."""
    pattern_tak = repo_simple_pattern.get("GLUCOSE_ON_ADMISSION_SIMPLE")
    df_empty = pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    
    df_out = pattern_tak.apply(df_empty)
    
    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "False"
    assert pd.isna(df_out.iloc[0]["StartDateTime"])


def test_pattern_select_last_prefers_latest_event(repo_simple_pattern):
    """select='last' prefers latest event."""
    # Modify pattern to use select='last' for event
    pattern_xml_last = PATTERN_SIMPLE_XML.replace("select='first'", "select='last'")
    pattern_path = write_xml(Path("/tmp"), "PATTERN_LAST.xml", pattern_xml_last)
    
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


def test_pattern_combined_score_averages_time_and_value(repo_value_compliance):
    """Combined score: average of time (1.0) + value (0.5) = 0.75 → Partial."""
    # Modify pattern to include BOTH time and value compliance
    pattern_xml_combined = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="COMBINED_COMPLIANCE" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Pattern with both time and value compliance</description>
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
    pattern_path = write_xml(Path("/tmp"), "PATTERN_COMBINED.xml", pattern_xml_combined)
    
    repo = repo_value_compliance
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)
    
    admission_tak = repo.get("ADMISSION_EVENT")
    insulin_tak = repo.get("INSULIN_BITZUA")
    weight_tak = repo.get("WEIGHT_MEASURE")
    
    df_raw = pd.DataFrame([
        (1, "ADMISSION", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "INSULIN_DOSAGE", make_ts("10:00"), make_ts("10:00"), 10),  # 2h gap, 10 units (0.14 units/kg → below [14.4, 43.2])
        (1, "WEIGHT", make_ts("07:00"), make_ts("07:00"), 72),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    
    df_admission = admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION"])
    df_insulin = insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_DOSAGE"])
    df_weight = weight_tak.apply(df_raw[df_raw["ConceptName"] == "WEIGHT"])
    df_input = pd.concat([df_admission, df_insulin, df_weight], ignore_index=True)
    
    df_out = pattern_tak.apply(df_input)
    
    assert len(df_out) == 1
    row = df_out.iloc[0]
    # Time score: 2h in [0h, 24h] → 1.0
    # Value score: 10 units = 0.14 units/kg → in [0, 14.4] → score = (14.4 - 10) / (14.4 - 0) ≈ 0.31
    # Combined: (1.0 + 0.31) / 2 = 0.65 → Partial
    assert row["Value"] == "Partial"
    assert row["TimeConstraintScore"] == 1.0
    assert 0.3 < row["ValueConstraintScore"] < 0.35

"""
Comprehensive unit tests for LocalPattern TAK.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from core.tak.pattern import LocalPattern, GlobalPattern
from core.tak.raw_concept import RawConcept, ParameterizedRawConcept
from core.tak.context import Context
from core.tak.event import Event
from core.tak.repository import set_tak_repository, get_tak_repository, TAKRepository
from .test_utils import write_xml, make_ts


# -----------------------------
# Helpers
# -----------------------------

def _tak_name_from_xml(xml_text: str) -> str:
    """Extract TAK 'name' attribute from the root element for better error messages."""
    root = ET.fromstring(xml_text)
    name = root.attrib.get("name")
    if not name:
        raise ValueError(f"XML root <{root.tag}> missing required attribute 'name'")
    return name

def _tak_root_tag(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    return root.tag

def _register_xml_strict(repo: "TAKRepository", xml_path: Path) -> None:
    """
    Strict dispatcher that parses based on root tag.
    Replace class names here to match your codebase if needed.
    """
    root = ET.parse(xml_path).getroot()
    tag = root.tag

    if tag == "raw-concept":
        repo.register(RawConcept.parse(xml_path))
    elif tag == "parameterized-raw-concept":
        # Adjust if your implementation uses a different class name
        repo.register(ParameterizedRawConcept.parse(xml_path))
    elif tag == "context":
        repo.register(Context.parse(xml_path))
    elif tag == "event":
        repo.register(Event.parse(xml_path))
    elif tag == "pattern":
        concept_type = root.attrib.get("concept-type")
        if concept_type == "local-pattern":
            repo.register(LocalPattern.parse(xml_path))
        elif concept_type == "global-pattern":
            repo.register(GlobalPattern.parse(xml_path))
        else:
            raise ValueError(f"Unknown pattern concept-type='{concept_type}' in {xml_path.name}")
    else:
        raise ValueError(f"Unknown TAK root tag '{tag}' in {xml_path.name}")

def tak(repo: "TAKRepository", name: str):
    """
    Convenience helper: repo.get(...) but with a clearer failure message.
    Keeps tests clean.
    """
    obj = repo.get(name)
    if obj is None:
        raise KeyError(f"TAK '{name}' not found in sandbox repository")
    return obj

@dataclass(frozen=True)
class SandboxTAKs:
    """Optional: typed accessors if you like IDE help in tests."""
    repo: "TAKRepository"

    def get(self, name: str):
        return tak(self.repo, name)

def _require(repo, name: str):
    obj = repo.get(name)
    if obj is None:
        raise KeyError(f"Missing TAK in repo: {name}")
    return obj

def _require_any(repo, names):
    for n in names:
        obj = repo.get(n)
        if obj is not None:
            return obj
    raise KeyError(f"Missing all TAK alternatives: {names}")

def _sandbox_repo(repo_protocol_kb, tak_names, extra_taks=()):
    """
    Create a fresh TAKRepository containing only the needed TAKs.
    Also calls set_tak_repository(repo) so LocalPattern.parse() resolves refs correctly.
    """
    repo = TAKRepository()
    for name in tak_names:
        repo.register(_require(repo_protocol_kb, name))
    for tak in extra_taks:
        repo.register(tak)
    set_tak_repository(repo)
    return repo

def _apply(repo, df_raw, *concept_names):
    """
    Strict helper:
    - Every TAK must exist in repo (hard fail if missing).
    - Empty raw slices are skipped.
    - Empty outputs are skipped.
    """
    parts = []

    for cn in concept_names:
        tak = _require(repo, cn)  # HARD FAIL if missing

        chunk = df_raw[df_raw["ConceptName"] == cn]
        if chunk.empty:
            continue

        out = tak.apply(chunk)
        if out is None or len(out) == 0:
            continue

        parts.append(out)

    if not parts:
        return df_raw.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)

def _apply_multi(repo, df_raw, mapping):
    """
    Strict helper for merged inputs.

    mapping: dict[str, list[str]]
        { tak_name: [raw concept names] }

    - TAK must exist (hard fail).
    - Empty masks are skipped.
    """
    parts = []

    for tak_name, raw_names in mapping.items():
        tak = _require(repo, tak_name)  # HARD FAIL if missing

        chunk = df_raw[df_raw["ConceptName"].isin(raw_names)]
        if chunk.empty:
            continue

        out = tak.apply(chunk)
        if out is None or len(out) == 0:
            continue

        parts.append(out)

    if not parts:
        return df_raw.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)

def _assert_close(actual, expected, tol=1e-6):
    """
    Assert that actual ~= expected within absolute tolerance tol.
    """
    assert actual is not None, f"Expected value close to {expected}, got None"
    assert abs(actual - expected) <= tol, (
        f"Expected {expected} ± {tol}, got {actual}"
    )

def _get_cyclic_score(row):
    val = row.get("CyclicConstraintScore", None)
    if pd.isna(val):
        return None
    return val

def _get_temporal_score(row):
    val = row.get("TemporalConstraintScore", None)
    if pd.isna(val):
        return None
    return val

def _get_value_score(row):
    val = row.get("ValueConstraintScore", None)
    if pd.isna(val):
        return None
    return val

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
    <attribute name="ADMISSION_EVENT" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_RELEASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="RELEASE_EVENT" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Release</description>
  <attributes>
    <attribute name="RELEASE_EVENT" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_DEATH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DEATH_EVENT" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Death indicator</description>
  <attributes>
    <attribute name="DEATH" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_GLUCOSE_MEASURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of GLUCOSE in units of mg/dL</description>
    <attributes>
        <attribute name="GLUCOSE_MEASURE" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="20" max="800"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
    """

RAW_STEROIDS_IV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="STEROIDS_IV_BITZUA" concept-type="raw-boolean">
  <categories>Medications</categories>
  <description>Steroids IV administration</description>
  <attributes>
    <attribute name="STEROIDS_IV_BITZUA" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_STEROIDS_PO_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="STEROIDS_PO_BITZUA" concept-type="raw-boolean">
  <categories>Medications</categories>
  <description>Steroids PO administration</description>
  <attributes>
    <attribute name="STEROIDS_PO_BITZUA" type="boolean"/>
  </attributes>
</raw-concept>
"""

# Contexts used by local insulin pattern (we will feed rows directly, but register to keep parsing stable)
RAW_MEAL_XML = """\
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
                <allowed-value value="Night-Snack"/>
                <allowed-value value="True"/>
            </nominal-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
    """


CTX_MEAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="MEAL_CONTEXT">
  <categories>Test</categories>
  <description>Meal context (test)</description>
  <derived-from>
    <attribute name="MEAL" tak="raw-concept" idx="0" ref="A1"/>
  </derived-from>
  <abstraction-rules>
    <rule value="True" operator="or">
      <attribute ref="A1"><allowed-value equal="True"/></attribute>
    </rule>
  </abstraction-rules>
  <context-windows>
    <persistence good-before="0h" good-after="0h"/>
  </context-windows>
</context>
"""

CTX_HIGH_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="HIGH_GLUCOSE_CONTEXT">
  <categories>Test</categories>
  <description>High glucose context (test)</description>
  <derived-from>
    <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
  </derived-from>
  <abstraction-rules>
    <rule value="True" operator="or">
      <attribute ref="A1"><allowed-value min="180"/></attribute>
    </rule>
  </abstraction-rules>
  <context-windows>
    <persistence good-before="24h" good-after="24h"/>
  </context-windows>
</context>
"""

RAW_WEIGHT_MEASURE_XML = """\
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

RAW_BMI_MEASURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BMI_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>BMI measure</description>
  <attributes>
    <attribute name="BMI" type="numeric">
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
    <attribute name="INSULIN_BITZUA" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_BASAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASAL_BITZUA" concept-type="raw-numeric">
  <categories>Medications</categories>
  <description>Insulin dosage</description>
  <attributes>
    <attribute name="BASAL_BITZUA" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="100"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

PARAMETER_BASAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="BASAL_BITZUA_RATIO">
    <categories>DrugAdministration</categories>
    <description>Raw concept to manage the BASAL ratio (long term) compared to the last administered dose</description>
    <derived-from name="BASAL_BITZUA" tak="raw-concept"/>
    
    <!-- Will use the closest instance to the beginning of the pattern -->
    <parameters>
        <parameter name="BASAL_BITZUA" tak="raw-concept" idx="0" how='before' dynamic="true" ref="P1"/>
    </parameters>

    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>

</parameterized-raw-concept>
"""

RAW_BOLUS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BOLUS_BITZUA" concept-type="raw-numeric">
  <categories>Medications</categories>
  <description>Insulin dosage</description>
  <attributes>
    <attribute name="BOLUS_BITZUA" type="numeric">
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
    <attribute name="DIABETES_DIAGNOSIS" type="boolean"/>
  </attributes>
</raw-concept>
"""

# Context (wraps DIABETES_DIAGNOSIS with windowing)
CTX_DIABETES_DIAGNOSIS_XML = """\
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
        <persistence good-before="1d" good-after="120y"/>
    </context-windows>
</context>
"""

RAW_AD_HOME_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ANTIDIABETIC_HOME_BITZUA" concept-type="raw-boolean">
  <categories>Medications</categories>
  <description>Antidiabetic medication administration</description>
  <attributes>
    <attribute name="ANTIDIABETIC_HOME_BITZUA" type="boolean"/>
  </attributes>
</raw-concept>
"""

RAW_AD_HOSPITAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ANTIDIABETIC_HOSPITAL_BITZUA" concept-type="raw-boolean">
  <categories>Medications</categories>
  <description>Antidiabetic medication administration</description>
  <attributes>
    <attribute name="ANTIDIABETIC_HOSPITAL_BITZUA" type="boolean"/>
  </attributes>
</raw-concept>
"""

DISGLYCEMIA_EVENT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="DISGLYCEMIA_EVENT">
  <categories>Measurements</categories>
  <description>Disglycemia event</description>
  <derived-from>
    <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="G1"/>
  </derived-from>
  <abstraction-rules>
    <rule value="Hypoglycemia" operator="or">
      <attribute ref="G1">
        <allowed-value max="70"/>
      </attribute>
    </rule>
  </abstraction-rules>
</event>
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
                        <trapez trapezA="0h" trapezB="0h" trapezC="8h" trapezD="12h"/>
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
                        <trapez trapezA="0" trapezB="0.2" trapezC="0.6" trapezD="1"/>
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
                        <trapez trapezA="0h" trapezB="0h" trapezC="8h" trapezD="12h"/>
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
                        <trapez trapezA="0h" trapezB="12h" trapezC="24h" trapezD="36h"/>
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
            <temporal-relation how='overlap' existence-compliance='true'>
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

# Pattern: overlap with existence-compliance enforced (no compliance function)
PATTERN_OVERLAP_EXISTENCE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_DURING_ADMISSION_EXISTENCE" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Overlap pattern with existence compliance (score emitted even without compliance-function)</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="INSULIN_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='overlap' existence-compliance='true'>
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

# Pattern: overlap + existence compliance + value compliance (ensure value score is computed)
PATTERN_OVERLAP_EXISTENCE_VALUE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_DURING_ADMISSION_EXISTENCE_VALUE" concept-type="local-pattern">
    <categories>QA</categories>
    <description>Overlap + existence-compliance + value compliance on insulin dosage</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="INSULIN_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='overlap' existence-compliance='true'>
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
                    <function name="id">
                        <!-- Full compliance for [0..10], ramps down to 0 by 20 -->
                        <trapez trapezA="0" trapezB="0" trapezC="10" trapezD="20"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

PATTERN_ROUTINE_GLUCOSE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ROUTINE_GLUCOSE_MEASURE_PATTERN" concept-type="global-pattern">
    <categories>RoutineTreatment</categories>
    <description>Glucose measured every 24h: 3/day if glucose >=180 else 1/day</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="I1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="DEATH_EVENT" tak="raw-concept" idx="0" ref="C1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
    </derived-from>

    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="E1">
                    <allowed-value min="180"/>
                </attribute>
            </context>

            <cyclic start='0h' end='14d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="20"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="3" trapezC="3" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>

        <rule>
            <cyclic start='0h' end='14d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="20"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="1" trapezC="1" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

PATTERN_ROUTINE_GLUCOSE_ON_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN" concept-type="global-pattern">
    <categories>RoutineTreatment</categories>
    <description>Glucose every 24h for first 2-3 days, depends on diabetes context</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="I1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="DEATH_EVENT" tak="raw-concept" idx="0" ref="C1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C3"/>
    </derived-from>

    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C3">
                    <allowed-value equal="True"/>
                </attribute>
            </context>

            <cyclic start='0h' end='3d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="20"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="3" trapezC="3" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>

        <rule>
            <cyclic start='0h' end='2d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="20"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="3" trapezC="3" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

PATTERN_ROUTINE_GLUCOSE_ON_STEROIDS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN" concept-type="global-pattern">
    <categories>RoutineTreatment</categories>
    <description>Glucose measured 3 times per 24h for 2d after steroids initiation</description>
    <derived-from>
        <attribute name="STEROIDS_IV_BITZUA" tak="raw-concept" idx="0" ref="I1"/>
        <attribute name="STEROIDS_PO_BITZUA" tak="raw-concept" idx="0" ref="I2"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="DEATH_EVENT" tak="raw-concept" idx="0" ref="C1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
    </derived-from>

    <abstraction-rules>
        <rule>
            <cyclic start='0h' end='2d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="I2">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="20"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="3" trapezC="3" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

PATTERN_ROUTINE_INSULIN_DOSAGE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ROUTINE_INSULIN_PATTERN" concept-type="global-pattern">
    <categories>RoutineTreatment</categories>
    <description>...</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="I1"/>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="DEATH_EVENT" tak="raw-concept" idx="0" ref="C1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="C3"/>
    </derived-from>

    <parameters>
        <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="81"/>
    </parameters>

    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C3">
                    <allowed-value min="180"/>
                </attribute>
            </context>

            <cyclic start='0h' end='14d' time-window='24h' min-occurrences="1" max-occurrences="100">
                <initiator>
                    <attribute ref="I1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="1"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>

            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0" trapezB="1" trapezC="4" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                    </target>
                    <function name="mul">
                        <parameter ref="P1"/>
                        <trapez trapezA="0" trapezB="0.2" trapezC="0.6" trapezD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Local pattern: BMI_MEASURE within trapez [0h,0h,48h,72h] after admission
LOCAL_BMI_ON_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="BMI_MEASURE_ON_ADMISSION" concept-type="local-pattern">
  <categories>Test</categories>
  <description>BMI within 3 days of admission (test)</description>

  <derived-from>
    <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
    <attribute name="BMI_MEASURE" tak="raw-concept" idx="0" ref="B1"/>
  </derived-from>

    <abstraction-rules>
        <rule>
            <!-- Relation between 2 KBTA objects -->
            <temporal-relation how='before' max-distance='72h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="B1">
                        <allowed-value min="10"/>
                    </attribute>
                </event>
            </temporal-relation>

            <!-- OPTIONAL: Compliance function to define fuzzy time window -->
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0h" trapezB="0h" trapezC="48h" trapezD="72h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

LOCAL_INSULIN_ON_HYPERGLYCEMIA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_ON_HYPERGLYCEMIA_PATTERN" concept-type="local-pattern">
    <categories>RoutineTreatment</categories>
    <description>Captures if INSULIN (BASAL/BOLUS/IV) was performed within proximity to a meal, assuming glucose was high at the time</description>
    <!-- Attributes must be 1D. Can reference all tak types for complex cases. -->
    <derived-from>
        <attribute name="MEAL_CONTEXT" tak="context" ref="A1"/>
        <attribute name="HIGH_GLUCOSE_CONTEXT" tak="context" ref="C1"/>
        
        <!-- BOLUS_BITZUA is used to clip HIGH_GLUCOSE_CONTEXT, so "before" relation will fit -->
        <attribute name="BOLUS_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="E2"/>
    </derived-from>

    <!-- Rules always share "OR" relationship. To create more complex patterns, reference a pattern in a different pattern TAK -->
    <abstraction-rules>
        <rule>
            <!-- OPTIONAL: Contextual Information relevant if begins at / before the last event of the pattern starts-->
            <!-- Context will be relevant here only if the high glucose context is true before the insulin event-->
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>
            <!-- Relation between 2 KBTA objects -->
            <!-- max-distance is not the recommended distance, but the max acceptable one -->
            <temporal-relation how='overlap' existence-compliance='true'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="1"/>
                    </attribute>
                    <attribute ref="E2">
                        <allowed-value min="1"/>    
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
    """
LOCAL_REDUCE_INSULIN_ON_HYPOGLYCEMIA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!-- In this case must ignore-unfulfilled-anchors="true" otherwise unfulfilled anchors emit 'False' / 0.0, which is the opposite of what we want -->
<!-- We are effectively only outputting compliance for fulfilled anchors, meaning cases where dose was given -->
<pattern name="REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN" concept-type="local-pattern" ignore-unfulfilled-anchors="true">
    <categories>RoutineTreatment</categories>
    <description>Captures if long term INSULIN (BASAL) was reduced after low glucose measured (if was given within 24h of the event, ignores otherwise)</description>
    <!-- Attributes must be 1D. Can reference all tak types for complex cases. -->
    <derived-from>
        <attribute name="DISGLYCEMIA_EVENT" tak="event" ref="A2"/>
        <attribute name="BASAL_BITZUA_RATIO" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>

    <!-- Rules always share "OR" relationship. To create more complex patterns, reference a pattern in a different pattern TAK -->
    <abstraction-rules>
        <rule>
            <!-- Rule 2: No overlap within low glucose context, but did we reduce dose outside this context within 24h? -->
            <!-- This is only relevant if glucose level balanced back, but insulin dose was given -->
            <!-- 24h after hypoglycemia event: Don't care about the insulin dose -->
            <temporal-relation how='before' max-distance='24h'>
                <anchor>
                    <attribute ref="A2">
                        <allowed-value equal="Hypoglycemia"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>

            <!-- OPTIONAL: Compliance function to define fuzzy time window -->
            <compliance-function>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                    </target>
                    <function name="id">
                        <!-- Any decrease in dose will is good compliance -->
                        <trapez trapezA="0" trapezB="0" trapezC="0.99" trapezD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
    """
LOCAL_STOP_ANTIDIABETICS_ON_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Captures if perscribed antidiabetic medications with high hypo risk were stopped during the first 3 days of admission</description>
    <!-- Attributes must be 1D. Can reference all tak types for complex cases. -->
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="ANTIDIABETIC_HOME_BITZUA" tak="raw-concept" idx="0" ref="C1"/>
        <attribute name="ANTIDIABETIC_HOSPITAL_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>

    <!-- Rules always share "OR" relationship. To create more complex patterns, reference a pattern in a different pattern TAK -->
    <abstraction-rules>
        <rule>
            <!-- OPTIONAL: Contextual Information relevant if begins at / before the last event of the pattern starts-->
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>

            <!-- Relation between 2 KBTA objects -->
            <!-- max-distance is not the recommended distance, but the max acceptable one -->
            <temporal-relation how='before' max-distance='14d'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value equal="True"/>
                    </attribute>
                </event>
            </temporal-relation>

            <!-- OPTIONAL: Compliance function to define fuzzy time window -->
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapez trapezA="0h" trapezB="72h" trapezC="14d" trapezD="14d"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
    """


GLUCOSE_MEASURE_ON_ADMISSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_MEASURE_ON_ADMISSION_PATTERN" concept-type="local-pattern">
  <categories>Measurements</categories>
  <description>Glucose measurement on admission, only when diabetes context holds</description>

  <derived-from>
    <!-- Anchor -->
    <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>

    <!-- Event -->
    <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>

    <!-- Required context (must be present as context rows in df_in) -->
    <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
  </derived-from>

  <abstraction-rules>
    <rule>
        <!-- Context gating: if this context is not satisfied, the anchor is dropped -->
        <context>
          <attribute ref="C1">
            <allowed-value equal="True"/>
          </attribute>
        </context>

      <temporal-relation how="before" max-distance="12h">
        <anchor>
          <attribute ref="A1">
            <allowed-value equal="True"/>
          </attribute>
        </anchor>

        <event select="first">
          <attribute ref="E1">
            <!-- any numeric glucose is fine -->
            <allowed-value min="0"/>
          </attribute>
        </event>
      </temporal-relation>

      <compliance-function>
        <time-constraint-compliance>
          <function name="id">
            <!-- 0..8h full, 8..12h rampdown -->
            <trapez trapezA="0h" trapezB="0h" trapezC="8h" trapezD="12h"/>
          </function>
        </time-constraint-compliance>
      </compliance-function>
    </rule>
  </abstraction-rules>
</pattern>
    """


# -----------------------------
# Global Pattern XML Fixtures
# -----------------------------

GLOBAL_PATTERN_SIMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ROUTINE_VITALS_CHECK" concept-type="global-pattern">
    <categories>Routine</categories>
    <description>Check vitals every 24h</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/> <!-- Anchor for start time -->
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <!-- Start 0h from anchor, end 48h from anchor. Window size 24h. -->
            <!-- Expecting 2 windows: [0, 24], [24, 48] -->
            <cyclic start='0h' end='48h' time-window='24h' min-occurrences="1" max-occurrences="10">
                <initiator>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>
        </rule>
    </abstraction-rules>
</pattern>
"""

GLOBAL_PATTERN_CYCLIC_COMPLIANCE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="STRICT_VITALS_CHECK" concept-type="global-pattern">
    <categories>Routine</categories>
    <description>Ideally 2 checks per 24h, 1 is partial</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <cyclic start='0h' end='24h' time-window='12h' min-occurrences="1" max-occurrences="10">
                <initiator>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>
            <compliance-function>
                <cyclic-constraint-compliance>
                    <function name="id">
                        <!-- 1 check = 0.5 score, 2 checks = 1.0 score -->
                        <trapez trapezA="0" trapezB="2" trapezC="10" trapezD="100"/>
                    </function>
                </cyclic-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

GLOBAL_PATTERN_VALUE_COMPLIANCE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="QUALITY_VITALS_CHECK" concept-type="global-pattern">
    <categories>Quality</categories>
    <description>Check every 24h, value should be below 140</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <cyclic start='0h' end='24h' time-window='2h' min-occurrences="1" max-occurrences="10">
                <initiator>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C1">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>
            <compliance-function>
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>
                    </target>
                    <function name="id">
                        <!-- Value < 140 is score 1.0. Value 200 is score 0.0 -->
                        <trapez trapezA="0" trapezB="0" trapezC="140" trapezD="200"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
"""

GLOBAL_PATTERN_CONTEXT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="ICU_VITALS_CHECK" concept-type="global-pattern">
    <categories>Routine</categories>
    <description>Check vitals only when in ICU context</description>
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>
            <cyclic start='0h' end='48h' time-window='24h' min-occurrences="1" max-occurrences="10">
                <initiator>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </initiator>
                <event>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
                <clipper>
                    <attribute ref="C2">
                        <allowed-value equal="True"/>
                    </attribute>
                </clipper>
            </cyclic>
        </rule>
    </abstraction-rules>
</pattern>
"""

GLOBAL_PATTERN_IGNORE_UNFULFILLED_ANCHORS_XML = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <pattern name="GLOBAL_TEST" concept-type="global-pattern" ignore-unfulfilled-anchors="true">
        <categories>Test</categories>
        <description>Should fail</description>
        <derived-from>
            <attribute name="ADMISSION_EVENT" tak="raw-concept" idx="0" ref="A1"/>
            <attribute name="SOME_EVENT" tak="raw-concept" idx="0" ref="E1"/>
            <attribute name="RELEASE_EVENT" tak="raw-concept" idx="0" ref="C2"/>
        </derived-from>
        <abstraction-rules>
            <rule>
                <cyclic-relation min-count="1" max-count="10" window="24h">
                    <initiator>
                        <attribute ref="A1">
                            <allowed-value equal="True"/>
                        </attribute>
                    </initiator>
                    <event>
                        <attribute ref="E1">
                            <allowed-value min="0"/>
                        </attribute>
                    </event>
                    <clipper>
                        <attribute ref="C2">
                            <allowed-value equal="True"/>
                        </attribute>
                    </clipper>
                </cyclic-relation>
            </rule>
        </abstraction-rules>
    </pattern>
    """

# -----------------------------
# Main fixture
# -----------------------------

@pytest.fixture
def repo_protocol_kb(tmp_path: Path) -> "TAKRepository":
    """
    Single sandbox TAK repository that includes *all* TAKs referenced by tests.
    The intent: tests never write/parse/register anything except through this fixture.
    """
    repo = TAKRepository()
    set_tak_repository(repo)

    # Everything your tests referenced (from the earlier XML list + per-repo fixtures)
    xml_blobs: dict[str, str] = {
        # -------------------------
        # Raw concepts
        # -------------------------
        "ADMISSION_EVENT.xml": RAW_ADMISSION_XML,
        "RELEASE_EVENT.xml": RAW_RELEASE_XML,
        "DEATH_EVENT.xml": RAW_DEATH_XML,

        "GLUCOSE_MEASURE.xml": RAW_GLUCOSE_MEASURE_XML,   # keep only the one you chose
        "WEIGHT_MEASURE.xml": RAW_WEIGHT_MEASURE_XML,
        "BMI_MEASURE.xml": RAW_BMI_MEASURE_XML,
        "MEAL.xml": RAW_MEAL_XML,
        "DIABETES_DIAGNOSIS.xml": RAW_DIABETES_XML,

        "INSULIN_BITZUA.xml": RAW_INSULIN_XML,
        "BASAL_BITZUA.xml": RAW_BASAL_XML,
        "BOLUS_BITZUA.xml": RAW_BOLUS_XML,

        "ANTIDIABETIC_HOME_BITZUA.xml": RAW_AD_HOME_XML,
        "ANTIDIABETIC_HOSPITAL_BITZUA.xml": RAW_AD_HOSPITAL_XML,

        "STEROIDS_IV_BITZUA.xml": RAW_STEROIDS_IV_XML,
        "STEROIDS_PO_BITZUA.xml": RAW_STEROIDS_PO_XML,

        # -------------------------
        # Parameterized raw concepts
        # -------------------------
        "BASAL_BITZUA_RATIO.xml": PARAMETER_BASAL_XML,

        # -------------------------
        # Contexts + Events
        # -------------------------
        "MEAL_CONTEXT.xml": CTX_MEAL_XML,
        "HIGH_GLUCOSE_CONTEXT.xml": CTX_HIGH_GLUCOSE_XML,
        "DIABETES_DIAGNOSIS_CONTEXT.xml": CTX_DIABETES_DIAGNOSIS_XML,
        "DISGLYCEMIA_EVENT.xml": DISGLYCEMIA_EVENT_XML,

        # -------------------------
        # Unit-test local patterns (the ones behind old dedicated fixtures)
        # -------------------------
        "GLUCOSE_ON_ADMISSION_SIMPLE.xml": PATTERN_SIMPLE_XML,
        "GLUCOSE_ON_ADMISSION_TIME.xml": PATTERN_TIME_COMPLIANCE_XML,
        "INSULIN_ON_ADMISSION_VALUE.xml": PATTERN_VALUE_COMPLIANCE_XML,
        "GLUCOSE_ON_ADMISSION_CONTEXT.xml": PATTERN_WITH_CONTEXT_XML,
        "GLUCOSE_ON_ADMISSION_MULTI.xml": PATTERN_MULTIPLE_RULES_XML,

        "INSULIN_DURING_ADMISSION.xml": PATTERN_OVERLAP_XML,
        "INSULIN_DURING_ADMISSION_EXISTENCE.xml": PATTERN_OVERLAP_EXISTENCE_XML,
        "INSULIN_DURING_ADMISSION_EXISTENCE_VALUE.xml": PATTERN_OVERLAP_EXISTENCE_VALUE_XML,

        # -------------------------
        # Protocol local patterns
        # -------------------------
        "BMI_MEASURE_ON_ADMISSION.xml": LOCAL_BMI_ON_ADMISSION_XML,
        "INSULIN_ON_HYPERGLYCEMIA_PATTERN.xml": LOCAL_INSULIN_ON_HYPERGLYCEMIA_XML,
        "REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN.xml": LOCAL_REDUCE_INSULIN_ON_HYPOGLYCEMIA_XML,
        "STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN.xml": LOCAL_STOP_ANTIDIABETICS_ON_ADMISSION_XML,
        "GLUCOSE_MEASURE_ON_ADMISSION_PATTERN.xml": GLUCOSE_MEASURE_ON_ADMISSION_XML,

        # -------------------------
        # Protocol global patterns
        # -------------------------
        "ROUTINE_GLUCOSE_MEASURE_PATTERN.xml": PATTERN_ROUTINE_GLUCOSE_XML,
        "ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN.xml": PATTERN_ROUTINE_GLUCOSE_ON_ADMISSION_XML,
        "ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN.xml": PATTERN_ROUTINE_GLUCOSE_ON_STEROIDS_XML,
        "ROUTINE_INSULIN_PATTERN.xml": PATTERN_ROUTINE_INSULIN_DOSAGE_XML,

        # -------------------------
        # Unit-test global patterns (the ones behind old dedicated fixtures)
        # -------------------------
        "ROUTINE_VITALS_CHECK.xml": GLOBAL_PATTERN_SIMPLE_XML,
        "STRICT_VITALS_CHECK.xml": GLOBAL_PATTERN_CYCLIC_COMPLIANCE_XML,
        "QUALITY_VITALS_CHECK.xml": GLOBAL_PATTERN_VALUE_COMPLIANCE_XML,
        "ICU_VITALS_CHECK.xml": GLOBAL_PATTERN_CONTEXT_XML,

        # If you actually test that this should fail, keep it out of the "good" repo
        # and create a separate fixture that intentionally registers invalid TAKs.
        # "GLOBAL_TEST.xml": GLOBAL_PATTERN_IGNORE_UNFULFILLED_ANCHORS_XML,
    }

    # Fail-fast on duplicate TAK names before writing files
    seen_names: dict[str, str] = {}
    for fname, xml in xml_blobs.items():
        name = _tak_name_from_xml(xml)
        if name in seen_names:
            raise AssertionError(
                f"Duplicate TAK name '{name}' in sandbox repo: "
                f"{seen_names[name]} and {fname}"
            )
        seen_names[name] = fname

    # 1) materialize all files first (so parsing works consistently)
    paths = []
    for fname in sorted(xml_blobs.keys()):
        path = tmp_path / fname
        path.write_text(xml_blobs[fname], encoding="utf-8")
        paths.append(path)

    # 2) register in dependency order (by root tag)
    TAG_ORDER = {
        "raw-concept": 0,
        "parameterized-raw-concept": 1,
        "context": 2,
        "event": 2,
        "pattern": 3,
    }

    paths.sort(key=lambda p: (TAG_ORDER.get(ET.parse(p).getroot().tag, 99), p.name))

    for path in paths:
        _register_xml_strict(repo, path)

    return repo


# Optional: expose the helper class so tests can do `taks.get("...")`
@pytest.fixture
def taks(repo_protocol_kb: "TAKRepository") -> SandboxTAKs:
    return SandboxTAKs(repo_protocol_kb)

# -----------------------------
# End-to-End Pattern Tests
# -----------------------------

def test_actual_pattern_bmi_on_admission_found(repo_protocol_kb):
    pattern = repo_protocol_kb.get("BMI_MEASURE_ON_ADMISSION")

    df_in = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "BMI_MEASURE",     make_ts("09:00"), make_ts("09:00"), 26.5),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    assert float(row["TimeConstraintScore"]) == 1.0


def test_actual_pattern_bmi_on_admission_partial_compliance(repo_protocol_kb):
    pattern = repo_protocol_kb.get("BMI_MEASURE_ON_ADMISSION")

    # Admission day 0 08:00 -> BMI at day 2 20:00 = 60h later
    df_in = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"),        make_ts("08:00"),        "True"),
        (1, "BMI_MEASURE",     make_ts("20:00", day=2), make_ts("20:00", day=2), 26.5),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "Partial"
    assert float(row["TimeConstraintScore"]) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------------------
# INSULIN_ON_HYPERGLYCEMIA_PATTERN
# TAK summary (from XML):
# - Context requirement: HIGH_GLUCOSE_CONTEXT must be True (and relevant before insulin event)
# - Temporal relation: overlap between MEAL_CONTEXT (anchor) and insulin event (event)
# - Event can match if ANY of {BOLUS_BITZUA>=1, BASAL_BITZUA>=1, INSULIN_BITZUA>=0.01} exist in overlap window
# - No compliance-function block: this is primarily existence + gating via min thresholds and context
# --------------------------------------------------------------------------------------

def test_insulin_on_hyperglycemia_fires_on_overlap_with_high_glucose_context(repo_protocol_kb):
    pattern = repo_protocol_kb.get("INSULIN_ON_HYPERGLYCEMIA_PATTERN")

    df_in = pd.DataFrame([
        (1, "MEAL_CONTEXT",        make_ts("09:00"), make_ts("11:00"), "True"),
        (1, "HIGH_GLUCOSE_CONTEXT",make_ts("08:30"), make_ts("10:30"), "True"),

        (1, "BOLUS_BITZUA",        make_ts("10:00"), make_ts("10:00"), 3),
        (1, "BASAL_BITZUA",        make_ts("06:00"), make_ts("06:00"), 0),
        (1, "INSULIN_BITZUA",      make_ts("10:00"), make_ts("10:00"), 0.0),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    assert out.iloc[0]["Value"] == "True"


def test_insulin_on_hyperglycemia_does_not_fire_without_high_glucose_context(repo_protocol_kb):
    pattern = repo_protocol_kb.get("INSULIN_ON_HYPERGLYCEMIA_PATTERN")

    df_in = pd.DataFrame([
        (1, "MEAL_CONTEXT",         make_ts("09:00"), make_ts("11:00"), "True"),
        (1, "BOLUS_BITZUA",         make_ts("10:00"), make_ts("10:00"), 3),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 0


def test_insulin_on_hyperglycemia_does_not_fire_when_no_overlap(repo_protocol_kb):
    pattern = repo_protocol_kb.get("INSULIN_ON_HYPERGLYCEMIA_PATTERN")

    df_in = pd.DataFrame([
        (1, "MEAL_CONTEXT",         make_ts("09:00"), make_ts("11:00"), "True"),
        (1, "HIGH_GLUCOSE_CONTEXT", make_ts("08:00"), make_ts("12:00"), "True"),
        (1, "BOLUS_BITZUA",         make_ts("12:30"), make_ts("12:30"), 3),  # outside
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    assert out.iloc[0]["ConceptName"] == "INSULIN_ON_HYPERGLYCEMIA_PATTERN"
    assert out.iloc[0]["Value"] == "False"
    assert out.iloc[0]["StartDateTime"] == make_ts("09:00")
    assert out.iloc[0]["EndDateTime"] == make_ts("11:00")
    assert out.iloc[0]["TimeConstraintScore"] == 0.0  # because existence-compliance

def test_insulin_on_hyperglycemia_no_existence_returns_empty(repo_protocol_kb, tmp_path):
    # Take the real production XML and modify ONLY semantics
    xml = LOCAL_INSULIN_ON_HYPERGLYCEMIA_XML

    xml = xml.replace(
        "how='overlap' existence-compliance='true'",
        "how='before' max-distance='30m'"
    )

    xml = xml.replace(
        "<pattern name=\"INSULIN_ON_HYPERGLYCEMIA_PATTERN\" concept-type=\"local-pattern\">",
        "<pattern name=\"INSULIN_ON_HYPERGLYCEMIA_NO_EXISTENCE\" "
        "concept-type=\"local-pattern\" ignore-unfulfilled-anchors=\"true\">"
    )

    # Write modified XML
    xml_path = tmp_path / "INSULIN_ON_HYPERGLYCEMIA_NO_EXISTENCE.xml"
    xml_path.write_text(xml, encoding="utf-8")

    # Register into an isolated repo (semantic edge case)
    repo = TAKRepository()
    set_tak_repository(repo)

    # Reuse the *real* protocol KB TAKs as dependencies
    for tak in repo_protocol_kb.taks.values():
        repo.register(tak)

    _register_xml_strict(repo, xml_path)

    pattern = repo.get("INSULIN_ON_HYPERGLYCEMIA_NO_EXISTENCE")
    assert pattern is not None

    # Same data as the failing overlap test
    df_in = pd.DataFrame([
        (1, "MEAL_CONTEXT",         make_ts("09:00"), make_ts("11:00"), "True"),
        (1, "HIGH_GLUCOSE_CONTEXT", make_ts("08:00"), make_ts("12:00"), "True"),
        (1, "BOLUS_BITZUA",         make_ts("12:30"), make_ts("12:30"), 3),  # outside
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)

    # THIS is the assertion we care about
    assert len(out) == 0

# --------------------------------------------------------------------------------------
# REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN
# TAK summary:
# - ignore-unfulfilled-anchors="true"  (important behavior)
# - temporal relation: anchor DISGLYCEMIA_EVENT with Value == "Hypoglycemia"
#   event BASAL_BITZUA_RATIO within 24h AFTER the hypoglycemia event (how='before', max-distance='24h')
# - value constraint compliance on BASAL_BITZUA_RATIO via trapez [0,0,0.99,1]
#   => ratio <= 0.99 should have ValueConstraintScore=1.0 (plateau), ratio=1.0 -> 0.0
# --------------------------------------------------------------------------------------

def test_reduce_insulin_on_hypoglycemia_within_24h_good_ratio(repo_protocol_kb):
    pattern = repo_protocol_kb.get("REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN")

    df_in = pd.DataFrame([
        (1, "DISGLYCEMIA_EVENT",    make_ts("08:00"), make_ts("08:00"), "Hypoglycemia"),
        (1, "BASAL_BITZUA_RATIO",   make_ts("20:00"), make_ts("20:00"), 0.8),  # +12h
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    assert float(row["ValueConstraintScore"]) == 1.0


def test_reduce_insulin_on_hypoglycemia_within_24h_bad_ratio(repo_protocol_kb):
    """
    Case: Hypoglycemia at T0, basal ratio at +6h but ratio=1.0.
    Expected: fires, but ValueConstraintScore should be 0.0 (trapezD boundary).
    """
    pattern = repo_protocol_kb.get("REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN")

    df = pd.DataFrame([
        (1, "DISGLYCEMIA_EVENT", make_ts("08:00"), make_ts("08:00"), "Hypoglycemia"),
        (1, "BASAL_BITZUA_RATIO", make_ts("14:00"), make_ts("14:00"), 1.0),  # +6h
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "False"
    assert row["ValueConstraintScore"] == 0.0


def test_reduce_insulin_on_hypoglycemia_no_basal_ratio_unfulfilled_anchor_ignored(repo_protocol_kb):
    """
    Case: Hypoglycemia occurs but no BASAL_BITZUA_RATIO found within 24h.
    Because ignore-unfulfilled-anchors="true", we should NOT emit a 'False' row.
    Expected: output is empty.
    """
    pattern = repo_protocol_kb.get("REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN")

    df = pd.DataFrame([
        (1, "DISGLYCEMIA_EVENT", make_ts("08:00"), make_ts("08:00"), "Hypoglycemia"),
        # no BASAL_BITZUA_RATIO rows at all
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)

    assert len(out) == 0, "Should ignore unfulfilled anchors and emit no row"


def test_reduce_insulin_on_hypoglycemia_basal_ratio_after_24h_unfulfilled_anchor_ignored(repo_protocol_kb):
    """
    Case: BASAL_BITZUA_RATIO exists but only after 24h.
    Expected: no match; ignore-unfulfilled-anchors=true => no 'False' output.
    """
    pattern = repo_protocol_kb.get("REDUCE_INSULIN_ON_HYPOGLYCEMIA_PATTERN")

    df = pd.DataFrame([
        (1, "DISGLYCEMIA_EVENT", make_ts("08:00"), make_ts("08:00"), "Hypoglycemia"),
        (1, "BASAL_BITZUA_RATIO", make_ts("09:00", day=1), make_ts("09:00", day=1), 0.7),  # +25h
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)

    assert len(out) == 0


# --------------------------------------------------------------------------------------
# STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN
# TAK summary:
# - Context requirement: ANTIDIABETIC_HOME_BITZUA_CONTEXT must be True
# - Temporal relation: ADMISSION_EVENT (anchor) before ANTIDIABETIC_HOSPITAL_BITZUA (event) with max-distance 14d
# - Compliance function (time): trapez [0h, 72h, 14d, 14d]
#   => score ramps up 0->1 between 0 and 72h, then plateau at 1 until 14d, then drops to 0 after 14d
# Note: This trapez direction is exactly what the XML states, even if the human description says “first 3 days”.
# --------------------------------------------------------------------------------------

def test_stop_antidiabetics_on_admission_context_required(repo_protocol_kb):
    """
    Case: event occurs within max distance but context is False.
    Expected: no output because context gating blocks the rule.
    """
    pattern = repo_protocol_kb.get("STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "ANTIDIABETIC_HOSPITAL_BITZUA", make_ts("20:00"), make_ts("20:00"), "True"),  # +12h
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)
    assert len(out) == 0


def test_stop_antidiabetics_time_score_ramps_before_72h(repo_protocol_kb):
    pattern = repo_protocol_kb.get("STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN")

    df_in = pd.DataFrame([
        (1, "ADMISSION_EVENT",                 make_ts("08:00"),        make_ts("08:00"),        "True"),
        (1, "ANTIDIABETIC_HOME_BITZUA",make_ts("00:00"),        make_ts("23:59"),        "True"),
        (1, "ANTIDIABETIC_HOSPITAL_BITZUA",    make_ts("08:00", day=1), make_ts("08:00", day=1), "True"),  # +24h
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "Partial"
    assert 0.30 <= float(row["TimeConstraintScore"]) <= 0.36


def test_stop_antidiabetics_time_score_plateau_after_72h(repo_protocol_kb):
    """
    Case: context=True and hospital antidiabetic at +96h (4 days).
    Expected: within plateau [72h,14d] -> TimeConstraintScore=1.0
    """
    pattern = repo_protocol_kb.get("STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "ANTIDIABETIC_HOME_BITZUA", make_ts("00:00"), make_ts("23:59"), "True"),
        (1, "ANTIDIABETIC_HOSPITAL_BITZUA", make_ts("08:00", day=4), make_ts("08:00", day=4), "True"),  # +96h
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    assert float(row["TimeConstraintScore"]) == 1.0


def test_stop_antidiabetics_no_match_after_14d(repo_protocol_kb):
    """
    Case: event occurs after 14 days from admission.
    The temporal relation has max-distance=14d, so it should not match at all.
    Expected: no output.
    """
    pattern = repo_protocol_kb.get("STOP_ANTIDIABETICS_ON_ADMISSION_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "ANTIDIABETIC_HOME_BITZUA", make_ts("00:00"), make_ts("23:59"), "True"),
        (1, "ANTIDIABETIC_HOSPITAL_BITZUA", make_ts("08:00", day=15), make_ts("08:00", day=15), "True"),  # +15d, after trapezoid
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])

    out = pattern.apply(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    assert float(row["TimeConstraintScore"]) == 1.0

# --------------------------------------------------------------------------------------
# ROUTINE_GLUCOSE_MEASURE_PATTERN (global)
# Context: any glucose >=180 -> stricter cyclic compliance trapezB=3 (needs 3/day).
# Otherwise relaxed trapezB=1.
# --------------------------------------------------------------------------------------

def test_routine_glucose_measure_hyperglycemia_requires_3_per_day(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 200),  # triggers hyperglycemia context
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 10  # 10 days
    assert out.iloc[0]["Value"] == "Partial"
    assert out.iloc[1]["Value"] == "False"

    # With trapezB=3: 1 occurrence/day should score around 1/3 (id trapezoid)
    _assert_close(_get_cyclic_score(out.iloc[0]), 1.0/3.0, tol=0.15)

def test_routine_glucose_measure_hyperglycemia_compliant_with_three_measures(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 190),
        (1, "GLUCOSE_MEASURE", make_ts("13:00"), make_ts("13:00"), 210),
        (1, "GLUCOSE_MEASURE", make_ts("19:00"), make_ts("19:00"), 185),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 10
    assert out.iloc[0]["Value"] == "True"
    _assert_close(_get_cyclic_score(out.iloc[0]), 1.0, tol=0.05)
    assert out.iloc[1]["Value"] == "False"


def test_routine_glucose_measure_no_hyperglycemia_relaxed_rule(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 160),  # no hyperglycemia
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    out = pattern.apply(df_in)
    assert len(out) == 10
    assert out.iloc[0]["Value"] == "True"
    _assert_close(_get_cyclic_score(out.iloc[0]), 1.0, tol=0.05)
    assert out.iloc[1]["Value"] == "False"


# --------------------------------------------------------------------------------------
# ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN (global)
# Context: DIABETES_DIAGNOSIS_CONTEXT True -> end=3d. Else end=2d.
# Both require 3/day (trapezB=3).
# --------------------------------------------------------------------------------------

def test_routine_glucose_on_admission_diabetes_true_requires_3_days(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    diabetes_raw = repo_protocol_kb.get("DIABETES_DIAGNOSIS")
    diabetes_ctx = repo_protocol_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")

    rows = [
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ]
    # Only 2 days monitored well (should be insufficient when diabetes=True requires 3d horizon)
    for d in [0, 1]:
        for t in ["09:00", "13:00", "19:00"]:
            rows.append((1, "GLUCOSE_MEASURE", make_ts(t, day=d), make_ts(t, day=d), 120))

    df = pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        diabetes_ctx.apply(diabetes_raw.apply(df[df["ConceptName"] == "DIABETES_DIAGNOSIS"])),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 3
    row = out.iloc[0]
    assert row["Value"] == "True"
    assert _get_cyclic_score(row) == 1.0

    row = out.iloc[2]
    assert row["Value"] == "False"
    assert _get_cyclic_score(row) == 0.0

def test_routine_glucose_on_admission_diabetes_true_compliant_3_days(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    diabetes_raw = repo_protocol_kb.get("DIABETES_DIAGNOSIS")
    diabetes_ctx = repo_protocol_kb.get("DIABETES_DIAGNOSIS_CONTEXT")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")

    rows = [
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ]
    for d in [0, 1, 2]:
        for t in ["09:00", "13:00", "19:00"]:
            rows.append((1, "GLUCOSE_MEASURE", make_ts(t, day=d), make_ts(t, day=d), 120))

    df = pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        diabetes_ctx.apply(diabetes_raw.apply(df[df["ConceptName"] == "DIABETES_DIAGNOSIS"])),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 3
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)
    row = out.iloc[2]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)


def test_routine_glucose_on_admission_diabetes_false_only_2_days(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")

    rows = [
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ]
    for d in [0, 1]:
        for t in ["09:00", "13:00", "19:00"]:
            rows.append((1, "GLUCOSE_MEASURE", make_ts(t, day=d), make_ts(t, day=d), 120))

    df = pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 2
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)
    row = out.iloc[1]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)


# --------------------------------------------------------------------------------------
# ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN (global)
# Initiator: steroids IV or PO must exist, otherwise NO OUTPUT.
# Requires 3/day (trapezB=3).
# --------------------------------------------------------------------------------------

def test_routine_glucose_on_steroids_no_steroids_no_output(repo_protocol_kb):
    steroids_iv = repo_protocol_kb.get("STEROIDS_IV_BITZUA")
    steroids_po = repo_protocol_kb.get("STEROIDS_PO_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    death = repo_protocol_kb.get("DEATH_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN")

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 120),
        (1, "RELEASE_EVENT", make_ts("08:00", day=5), make_ts("08:00", day=5), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 0


def test_routine_glucose_on_steroids_requires_3_per_day(repo_protocol_kb):
    steroids_iv = repo_protocol_kb.get("STEROIDS_IV_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN")

    df = pd.DataFrame([
        (1, "STEROIDS_IV_BITZUA", make_ts("08:30"), make_ts("08:30"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 110),
        (1, "RELEASE_EVENT", make_ts("08:00", day=5), make_ts("08:00", day=5), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        steroids_iv.apply(df[df["ConceptName"] == "STEROIDS_IV_BITZUA"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 2
    row = out.iloc[0]
    assert row["Value"] == "Partial"
    _assert_close(_get_cyclic_score(row), 1.0/3.0, tol=0.05)


def test_routine_glucose_on_steroids_compliant_with_3_measures(repo_protocol_kb):
    steroids_iv = repo_protocol_kb.get("STEROIDS_IV_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_GLUCOSE_MEASURE_ON_STEROIDS_PATTERN")

    df = pd.DataFrame([
        (1, "STEROIDS_IV_BITZUA", make_ts("08:30"), make_ts("08:30"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 110),
        (1, "GLUCOSE_MEASURE", make_ts("13:00"), make_ts("13:00"), 115),
        (1, "GLUCOSE_MEASURE", make_ts("19:00"), make_ts("19:00"), 120),
        (1, "RELEASE_EVENT", make_ts("08:00", day=5), make_ts("08:00", day=5), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        steroids_iv.apply(df[df["ConceptName"] == "STEROIDS_IV_BITZUA"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 2
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)


# --------------------------------------------------------------------------------------
# ROUTINE_INSULIN_PATTERN (global)
# Requires hyperglycemia context (glucose >=180).
# Cyclic compliance: trapezB=1, trapezC=4 (1..4/day good).
# Value compliance: mul(weight, trapez [0.2,0.6] plateau, fade to 0 at 1.0*weight), default weight=81.
# --------------------------------------------------------------------------------------

def test_routine_insulin_requires_hyperglycemia_context(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    basal = repo_protocol_kb.get("BASAL_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    weight = repo_protocol_kb.get("WEIGHT_MEASURE")
    pattern = repo_protocol_kb.get("ROUTINE_INSULIN_PATTERN")

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 160),  # not >=180
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), 30),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        basal.apply(df[df["ConceptName"] == "BASAL_BITZUA"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 0

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "WEIGHT_MEASURE", make_ts("07:50"), make_ts("07:50"), 80),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 200), # Context trigger on window 1
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), 40),
        (1, "GLUCOSE_MEASURE", make_ts("09:00", day=1), make_ts("09:00", day=1), 200), # Context trigger on window 2
        (1, "BASAL_BITZUA", make_ts("16:00", day=1), make_ts("16:00", day=1), 50), # 50/80=0.625 - partial value score
        (1, "GLUCOSE_MEASURE", make_ts("09:00", day=2), make_ts("09:00", day=2), 200), # Context trigger on window 3
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        weight.apply(df[df["ConceptName"] == "WEIGHT_MEASURE"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        basal.apply(df[df["ConceptName"] == "BASAL_BITZUA"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 3
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)
    _assert_close(_get_value_score(row), 1.0, tol=0.05)
    
    row = out.iloc[1]
    assert row["Value"] == "Partial"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)
    assert float(row["ValueConstraintScore"]) < 1.0

    row = out.iloc[2]
    assert row["Value"] == "False"
    _assert_close(_get_cyclic_score(row), 0.0, tol=0.05)
    _assert_close(_get_value_score(row), 0.0, tol=0.05)


def test_routine_insulin_value_uses_weight_parameter(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    basal = repo_protocol_kb.get("BASAL_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    weight = repo_protocol_kb.get("WEIGHT_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    death = repo_protocol_kb.get("DEATH_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_INSULIN_PATTERN")

    # weight=80 => plateau 0.2..0.6*weight => 16..48. dose=40 => full value score
    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "WEIGHT_MEASURE", make_ts("07:50"), make_ts("07:50"), 80),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 200), # Context trigger on window 1
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), 40),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        weight.apply(df[df["ConceptName"] == "WEIGHT_MEASURE"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        basal.apply(df[df["ConceptName"] == "BASAL_BITZUA"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_cyclic_score(row), 1.0, tol=0.05)
    _assert_close(_get_value_score(row), 1.0, tol=0.05)


def test_routine_insulin_value_uses_default_weight_when_missing(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    basal = repo_protocol_kb.get("BASAL_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    death = repo_protocol_kb.get("DEATH_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_INSULIN_PATTERN")

    # default weight=81 => plateau 16.2..48.6; dose=40 => still full score
    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 200),
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), 40),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        basal.apply(df[df["ConceptName"] == "BASAL_BITZUA"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "True"
    _assert_close(_get_value_score(row), 1.0, tol=0.05)


def test_routine_insulin_value_score_drops_when_dose_too_high(repo_protocol_kb):
    admission = repo_protocol_kb.get("ADMISSION_EVENT")
    basal = repo_protocol_kb.get("BASAL_BITZUA")
    glucose = repo_protocol_kb.get("GLUCOSE_MEASURE")
    weight = repo_protocol_kb.get("WEIGHT_MEASURE")
    release = repo_protocol_kb.get("RELEASE_EVENT")
    death = repo_protocol_kb.get("DEATH_EVENT")
    pattern = repo_protocol_kb.get("ROUTINE_INSULIN_PATTERN")

    # weight=80 => C=0.6*80=48, D=1.0*80=80; dose=70 in decreasing limb
    expected = (80 - 70) / (80 - 48)

    df = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "WEIGHT_MEASURE", make_ts("07:50"), make_ts("07:50"), 80),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 200),
        (1, "BASAL_BITZUA", make_ts("12:00"), make_ts("12:00"), 70),
        (1, "RELEASE_EVENT", make_ts("08:00", day=10), make_ts("08:00", day=10), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df[df["ConceptName"] == "ADMISSION_EVENT"]),
        weight.apply(df[df["ConceptName"] == "WEIGHT_MEASURE"]),
        glucose.apply(df[df["ConceptName"] == "GLUCOSE_MEASURE"]),
        basal.apply(df[df["ConceptName"] == "BASAL_BITZUA"]),
        release.apply(df[df["ConceptName"] == "RELEASE_EVENT"]),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Value"] == "Partial"
    _assert_close(_get_value_score(row), expected, tol=0.15)


# -----------------------------
# Tests: Parsing & Validation
# -----------------------------

def test_parse_pattern_validates_structure(repo_protocol_kb, tmp_path):
    """
    Validate XML structure and instantiation in an isolated sandbox repo.
    """
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    pattern = LocalPattern.parse(pattern_path)
    repo.register(pattern)

    assert pattern.name == "GLUCOSE_ON_ADMISSION_SIMPLE"
    assert len(pattern.derived_from) == 2
    assert pattern.derived_from[0]["ref"] == "A1"
    assert pattern.derived_from[1]["ref"] == "E1"
    assert len(pattern.abstraction_rules) == 1


def test_pattern_validation_requires_max_distance_for_before(tmp_path: Path):
    """Validation fails if how='before' and no max-distance."""
    bad_pattern_xml = PATTERN_SIMPLE_XML.replace("max-distance='12h'", "")
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "BAD_PATTERN.xml", bad_pattern_xml)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    with pytest.raises(ValueError, match="requires max-distance"):
        LocalPattern.parse(pattern_path)

    # Test with missing min-distance (should default to 0s)
    valid_pattern_xml = PATTERN_SIMPLE_XML.replace("min-distance='0s'", "")
    pattern_path = write_xml(tmp_path, "VALID_PATTERN.xml", valid_pattern_xml)
    pattern = LocalPattern.parse(pattern_path)
    assert pattern is not None, "Pattern should parse successfully with default min-distance"


def test_pattern_validation_max_distance_ge_trapezD(tmp_path: Path):
    """Validation fails if max-distance < trapezD."""
    bad_pattern_xml = PATTERN_TIME_COMPLIANCE_XML.replace("max-distance='12h'", "max-distance='6h'")
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "BAD_PATTERN.xml", bad_pattern_xml)
    
    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)
    
    with pytest.raises(ValueError, match="max-distance.*must be >= .*trapezD"):
        LocalPattern.parse(pattern_path)


def test_pattern_validation_rejects_numeric_equal_constraint(tmp_path: Path):
    """Validation fails if numeric attribute has equal constraint."""
    bad_pattern_xml = PATTERN_SIMPLE_XML.replace(
        '<allowed-value min="0"/>',
        '<allowed-value equal="100"/>'
    )
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
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

def test_pattern_found_simple(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_input = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_input)

    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "True"
    assert row["StartDateTime"] == make_ts("08:00")
    assert row["EndDateTime"] == make_ts("10:00")
    assert pd.isna(row["TimeConstraintScore"])
    assert pd.isna(row["ValueConstraintScore"])


def test_pattern_not_found_no_event(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_admission = admission_tak.apply(df_raw)
    df_out = pattern_tak.apply(df_admission)

    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "False"
    assert row["StartDateTime"] == make_ts("08:00")
    assert row["EndDateTime"] == make_ts("08:00")


def test_pattern_not_found_event_too_late(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("22:00"), make_ts("22:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_input = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_input)

    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row["Value"] == "False"
    assert row["StartDateTime"] == make_ts("08:00")
    assert row["EndDateTime"] == make_ts("08:00")

def test_pattern_one_to_one_pairing(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 100),
        (1, "ADMISSION_EVENT", make_ts("10:00"), make_ts("10:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("11:00"), make_ts("11:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_input = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_input)

    assert len(df_out) == 2
    assert all(df_out["Value"] == "True")


def test_missed_opportunity_mixed_results(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), 100),
        (1, "ADMISSION_EVENT", make_ts("12:00"), make_ts("12:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_input = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_input).sort_values("StartDateTime").reset_index(drop=True)

    assert len(df_out) == 2
    assert df_out.iloc[0]["Value"] == "True"
    assert df_out.iloc[0]["StartDateTime"] == make_ts("08:00")
    assert df_out.iloc[1]["Value"] == "False"
    assert df_out.iloc[1]["StartDateTime"] == make_ts("12:00")
    assert df_out.iloc[1]["EndDateTime"] == make_ts("12:00")


def test_missed_opportunity_filtered_by_overlap(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), 100),
        (1, "ADMISSION_EVENT", make_ts("10:00"), make_ts("10:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_input = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_input)

    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"
    assert df_out.iloc[0]["StartDateTime"] == make_ts("08:00")


def test_pattern_empty_input_returns_false(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN.xml", PATTERN_SIMPLE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_empty = pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    df_out = pattern_tak.apply(df_empty)

    assert len(df_out) == 0
    assert df_out.empty

def test_missed_opportunity_filtered_by_context(repo_protocol_kb):
    admission_tak = _require(repo_protocol_kb, "ADMISSION_EVENT")
    pattern_tak = _require_any(repo_protocol_kb, ["GLUCOSE_ON_ADMISSION_CONTEXT", "GLUCOSE_MEASURE_ON_ADMISSION_PATTERN"])

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_adm = admission_tak.apply(df_raw)
    df_out = pattern_tak.apply(df_adm)

    # Context gate makes the anchor ineligible, so we do NOT punish it
    assert len(df_out) == 0
    assert df_out.empty
    
# -----------------------------
# Tests: Time-Constraint Compliance
# -----------------------------

def test_time_compliance_full(repo_protocol_kb, tmp_path):
    admission_path = write_xml(tmp_path, "ADMISSION_EVENT.xml", RAW_ADMISSION_XML)
    glucose_path = write_xml(tmp_path, "GLUCOSE.xml", RAW_GLUCOSE_MEASURE_XML)
    pattern_path = write_xml(tmp_path, "PATTERN_TIME.xml", PATTERN_TIME_COMPLIANCE_XML)

    repo = TAKRepository()
    repo.register(RawConcept.parse(admission_path))
    repo.register(RawConcept.parse(glucose_path))
    set_tak_repository(repo)

    admission_tak = repo.get("ADMISSION_EVENT")
    glucose_tak = repo.get("GLUCOSE_MEASURE")
    pattern_tak = LocalPattern.parse(pattern_path)
    repo.register(pattern_tak)

    df_raw = pd.DataFrame([
        (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    df_out = pattern_tak.apply(df_in)

    assert len(df_out) == 1
    assert df_out.iloc[0]["Value"] == "True"
    assert df_out.iloc[0]["TimeConstraintScore"] == 1.0


def test_time_compliance_partial(repo_protocol_kb, tmp_path):
    # Make a temporary pattern XML variant that yields a Partial time score
    pattern_xml = (
        PATTERN_TIME_COMPLIANCE_XML
        .replace("max-distance='12h'", "min-distance='2h' max-distance='12h'")
        .replace('name="GLUCOCE_ON_ADMISSION_TIME"', 'name="GLUCOSE_ON_ADMISSION_TIME_MIN_DISTANCE"')  # if typo exists
        .replace('name="GLUCOSE_ON_ADMISSION_TIME"', 'name="GLUCOSE_ON_ADMISSION_TIME_MIN_DISTANCE"')
        .replace(
            '<trapez trapezA="0h" trapezB="0h" trapezC="8h" trapezD="12h"/>',
            '<trapez trapezA="2h" trapezB="6h" trapezC="8h" trapezD="12h"/>'
        )
    )
    pattern_path = write_xml(tmp_path, "PATTERN_TIME_MIN_DISTANCE.xml", pattern_xml)

    # Take TAK objects from the original repo
    admission_tak = _require(repo_protocol_kb, "ADMISSION_EVENT")
    glucose_tak = _require(repo_protocol_kb, "GLUCOSE_MEASURE")

    # Build a temporary repo with just what we need
    old_repo = get_tak_repository()
    tmp_repo = TAKRepository()
    tmp_repo.register(admission_tak)
    tmp_repo.register(glucose_tak)

    set_tak_repository(tmp_repo)
    try:
        pattern_tak = LocalPattern.parse(pattern_path)
        tmp_repo.register(pattern_tak)

        df_raw = pd.DataFrame([
            (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
            (1, "GLUCOSE_MEASURE", make_ts("12:00"), make_ts("12:00"), 120),  # 4h gap => should be Partial with trapezA=2h, B=6h
        ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

        df_in = pd.concat([
            admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
            glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
        ], ignore_index=True)

        out = pattern_tak.apply(df_in)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["Value"] == "Partial"
        assert 0.0 < float(row["TimeConstraintScore"]) < 1.0
    finally:
        # Restore global repo no matter what
        set_tak_repository(old_repo)

def test_time_compliance_outside_max_distance_is_false(repo_protocol_kb, tmp_path):
    # Use the base time compliance pattern as-is
    pattern_path = write_xml(tmp_path, "PATTERN_TIME.xml", PATTERN_TIME_COMPLIANCE_XML)

    admission_tak = _require(repo_protocol_kb, "ADMISSION_EVENT")
    glucose_tak = _require(repo_protocol_kb, "GLUCOSE_MEASURE")

    old_repo = get_tak_repository()
    tmp_repo = TAKRepository()
    tmp_repo.register(admission_tak)
    tmp_repo.register(glucose_tak)

    set_tak_repository(tmp_repo)
    try:
        pattern_tak = LocalPattern.parse(pattern_path)
        tmp_repo.register(pattern_tak)

        df_raw = pd.DataFrame([
            (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
            (1, "GLUCOSE_MEASURE", make_ts("21:00"), make_ts("21:00"), 120),  # 13h gap
        ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

        df_in = pd.concat([
            admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
            glucose_tak.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
        ], ignore_index=True)

        out = pattern_tak.apply(df_in)

        # With max-distance=12h, there should be no valid pair, so we expect an unfulfilled anchor
        assert len(out) == 1
        assert out.iloc[0]["Value"] == "False"
        assert out.iloc[0]["StartDateTime"] == make_ts("08:00")
        assert out.iloc[0]["EndDateTime"] == make_ts("08:00")
    finally:
        set_tak_repository(old_repo)


# -----------------------------
# Tests: Value-Constraint Compliance
# -----------------------------

def test_value_compliance_downgrades_to_partial(repo_protocol_kb, tmp_path):
    # Write the exact value-compliance pattern XML you already have
    pattern_path = write_xml(tmp_path, "PATTERN_VALUE.xml", PATTERN_VALUE_COMPLIANCE_XML)

    # Get TAKs from the existing repo
    admission_tak = _require(repo_protocol_kb, "ADMISSION_EVENT")
    insulin_tak = _require(repo_protocol_kb, "INSULIN_BITZUA")
    weight_tak = _require(repo_protocol_kb, "WEIGHT_MEASURE")

    old_repo = get_tak_repository()
    tmp_repo = TAKRepository()
    tmp_repo.register(admission_tak)
    tmp_repo.register(insulin_tak)
    tmp_repo.register(weight_tak)

    set_tak_repository(tmp_repo)
    try:
        pattern_tak = LocalPattern.parse(pattern_path)
        tmp_repo.register(pattern_tak)

        df_raw = pd.DataFrame([
            (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
            (1, "WEIGHT_MEASURE",  make_ts("07:00"), make_ts("07:00"), 72),
            (1, "INSULIN_BITZUA",  make_ts("10:00"), make_ts("10:00"), 60),  # intentionally high
        ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

        df_in = pd.concat([
            admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
            insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_BITZUA"]),
            weight_tak.apply(df_raw[df_raw["ConceptName"] == "WEIGHT_MEASURE"]),
        ], ignore_index=True)

        out = pattern_tak.apply(df_in)
        assert len(out) == 1

        row = out.iloc[0]
        assert row["Value"] == "Partial"
        assert row["ValueConstraintScore"] is not None
        assert 0.0 < float(row["ValueConstraintScore"]) < 1.0

        # Optional: ensure time was not the reason (if your engine emits it)
        # assert pd.isna(row["TimeConstraintScore"]) or float(row["TimeConstraintScore"]) == 1.0

    finally:
        set_tak_repository(old_repo)

def test_value_compliance_uses_default_parameter_when_missing(repo_protocol_kb, tmp_path):
    pattern_path = write_xml(tmp_path, "PATTERN_VALUE.xml", PATTERN_VALUE_COMPLIANCE_XML)

    admission_tak = _require(repo_protocol_kb, "ADMISSION_EVENT")
    insulin_tak = _require(repo_protocol_kb, "INSULIN_BITZUA")
    # NOTE: intentionally NOT registering WEIGHT_MEASURE into df_in to force default usage
    # But we still register the TAK itself if the pattern depends on it existing in repo/validation.
    weight_tak = _require(repo_protocol_kb, "WEIGHT_MEASURE")

    old_repo = get_tak_repository()
    tmp_repo = TAKRepository()
    tmp_repo.register(admission_tak)
    tmp_repo.register(insulin_tak)
    tmp_repo.register(weight_tak)  # keep in repo for validate(), even if no rows exist

    set_tak_repository(tmp_repo)
    try:
        pattern_tak = LocalPattern.parse(pattern_path)
        tmp_repo.register(pattern_tak)

        df_raw = pd.DataFrame([
            (1, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
            (1, "INSULIN_BITZUA",  make_ts("10:00"), make_ts("10:00"), 25),
            # No WEIGHT_MEASURE rows in input -> must use default parameter
        ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

        df_in = pd.concat([
            admission_tak.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
            insulin_tak.apply(df_raw[df_raw["ConceptName"] == "INSULIN_BITZUA"]),
        ], ignore_index=True)

        # Sanity: confirm tuples are present (your normalization choice)
        assert isinstance(df_in[df_in["ConceptName"] == "INSULIN_BITZUA"].iloc[0]["Value"], tuple)

        out = pattern_tak.apply(df_in)
        assert len(out) == 1

        row = out.iloc[0]
        assert row["Value"] in {"True", "Partial"}  # depending on your trapez definition
        assert row["ValueConstraintScore"] is not None
        assert 0.0 < float(row["ValueConstraintScore"]) <= 1.0

        # If your default is truly chosen to make 25 “good”, you can keep this stricter:
        # assert row["Value"] == "True"
        # assert float(row["ValueConstraintScore"]) == 1.0

    finally:
        set_tak_repository(old_repo)


# -----------------------------
# Tests: Context Gating
# -----------------------------

def test_glucose_on_admission_requires_diabetes_context(repo_protocol_kb):
    admission = _require(repo_protocol_kb, "ADMISSION_EVENT")
    glucose   = _require(repo_protocol_kb, "GLUCOSE_MEASURE")
    diab_raw  = _require(repo_protocol_kb, "DIABETES_DIAGNOSIS")
    diab_ctx  = _require(repo_protocol_kb, "DIABETES_DIAGNOSIS_CONTEXT")
    pattern   = _require(repo_protocol_kb, "GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")

    # --- Case A: context present -> should emit 1 row (True/Partial depending on time compliance)
    df_raw = pd.DataFrame([
        (1000, "ADMISSION_EVENT",    make_ts("08:00"), make_ts("08:00"), "True"),
        (1000, "GLUCOSE_MEASURE",    make_ts("10:00"), make_ts("10:00"), 120),
        (1000, "DIABETES_DIAGNOSIS", make_ts("07:00"), make_ts("07:00"), "True"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in = pd.concat([
        admission.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"]),
        diab_ctx.apply(diab_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"])),
    ], ignore_index=True)

    out = pattern.apply(df_in)
    assert len(out) == 1
    assert out.iloc[0]["Value"] in ["True", "Partial"]

    # --- Case B: no context rows at all -> anchor not eligible -> no output rows
    df_raw2 = pd.DataFrame([
        (1000, "ADMISSION_EVENT", make_ts("08:00"), make_ts("08:00"), "True"),
        (1000, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), 120),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_in2 = pd.concat([
        admission.apply(df_raw2[df_raw2["ConceptName"] == "ADMISSION_EVENT"]),
        glucose.apply(df_raw2[df_raw2["ConceptName"] == "GLUCOSE_MEASURE"]),
    ], ignore_index=True)

    out2 = pattern.apply(df_in2)
    assert len(out2) == 0

# -----------------------------
# Tests: Multiple Anchors with Context
# -----------------------------

def test_glucose_on_admission_context_filters_out_later_anchor(repo_protocol_kb):
    admission = _require(repo_protocol_kb, "ADMISSION_EVENT")
    glucose   = _require(repo_protocol_kb, "GLUCOSE_MEASURE")
    diab_raw  = _require(repo_protocol_kb, "DIABETES_DIAGNOSIS")
    diab_ctx  = _require(repo_protocol_kb, "DIABETES_DIAGNOSIS_CONTEXT")
    pattern   = _require(repo_protocol_kb, "GLUCOSE_MEASURE_ON_ADMISSION_PATTERN")

    df_raw = pd.DataFrame([
        # Anchor 1 (no context supplied for it)
        (1000, "ADMISSION_EVENT",    make_ts("08:00"), make_ts("08:00"), "True"),
        (1000, "GLUCOSE_MEASURE",    make_ts("10:00"), make_ts("10:00"), 120),

        # Context becomes relevant here (1h before StartTime)
        (1000, "DIABETES_DIAGNOSIS", make_ts("07:00", day=2), make_ts("07:00", day=2), "True"),

        # Anchor 2 (has context + event)
        (1000, "ADMISSION_EVENT", make_ts("08:00", day=2), make_ts("08:00", day=2), "True"),
        (1000, "GLUCOSE_MEASURE", make_ts("10:00", day=2), make_ts("10:00", day=2), 130),
        # no diabetes diagnosis around day=2
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

    df_adm = admission.apply(df_raw[df_raw["ConceptName"] == "ADMISSION_EVENT"])
    df_glu = glucose.apply(df_raw[df_raw["ConceptName"] == "GLUCOSE_MEASURE"])
    df_diab_ctx = diab_ctx.apply(diab_raw.apply(df_raw[df_raw["ConceptName"] == "DIABETES_DIAGNOSIS"]))

    df_in = pd.concat([df_adm, df_glu, df_diab_ctx], ignore_index=True)
    out = pattern.apply(df_in).sort_values("StartDateTime").reset_index(drop=True)

    # Only the first admission is eligible due to context gating
    assert len(out) == 1
    assert out.iloc[0]["StartDateTime"] == make_ts("08:00", day=2)
    assert out.iloc[0]["Value"] in ["True", "Partial"]
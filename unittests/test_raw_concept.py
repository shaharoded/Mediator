"""
Comprehensive unit tests for RawConcept TAK.
"""
import pandas as pd
import pytest
from pathlib import Path

from core.tak.raw_concept import RawConcept, ParameterizedRawConcept
from core.tak.repository import set_tak_repository, TAKRepository
from unittests.test_utils import write_xml, make_ts  # FIXED: correct import path


# -----------------------------
# XML Fixtures (self-contained)
# -----------------------------

RAW_XML = """\
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

RAW_NUMERIC_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Glucose numeric measure</description>
  <attributes>
    <attribute name="GLUCOSE_LAB_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="50" max="400"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_NOMINAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ANTIDIABETIC_DRUGS_IV_BITZUA" concept-type="raw-nominal">
  <categories>Medications</categories>
  <description>IV antidiabetic treatment intensity</description>
  <attributes>
    <attribute name="ANTIDIABETIC_DRUGS_IV_DOSAGE" type="nominal">
      <nominal-allowed-values>
        <allowed-value value="Low"/>
        <allowed-value value="Medium"/>
        <allowed-value value="High"/>
      </nominal-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

RAW_BOOLEAN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission event</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
  </attributes>
</raw-concept>
"""

PARAM_RAW_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="BASE_GLUCOSE_MEASURE" tak="raw-concept" how='all' dynamic='true' idx="0" ref="P1" default="120"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

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
  <tuple-order>
    <attribute name="GLUCOSE_LAB_MEASURE"/>
  </tuple-order>
  <merge require-all="false"/>
</raw-concept>
"""
HIGH_GLUCOSE_IND_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="HIGH_GLUCOSE_IND" concept-type="raw-numeric">
    <categories>Context</categories>
    <description>Raw concept to indicate the transition to high glucose range in admission</description>
    <attributes>
        <attribute name="GLUCOSE_MEASURE" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="180"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>    
    """

LOW_GLUCOSE_IND_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="LOW_GLUCOSE_IND" concept-type="raw-numeric">
    <categories>Context</categories>
    <description>Raw concept to indicate the transition to low glucose range in admission</description>
    <attributes>
        <attribute name="GLUCOSE_MEASURE" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="20" max="70"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
    """

STEADY_GLUCOSE_MEASURE_HIGH_AFTER_FIRST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="STEADY_GLUCOSE_MEASURE_HIGH_AFTER_FIRST">
    <categories>Measurements</categories>
    <description>Raw concept that outputs only the glucose measurements after the first high glucose measurement (meaning it's repetative if high)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    
    <!-- Will use the closest instance to the beginning of the pattern -->
    <parameters>
        <parameter name="HIGH_GLUCOSE_IND" tak="raw-concept" how='before' dynamic="false" idx="0" ref="P1"/>
    </parameters>

    <functions>
        <function name="id">
            <value idx="0"/>
        </function>
    </functions>

</parameterized-raw-concept>    
    """

STEADY_GLUCOSE_MEASURE_LOW_AFTER_FIRST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="STEADY_GLUCOSE_MEASURE_LOW_AFTER_FIRST">
    <categories>Measurements</categories>
    <description>Raw concept that outputs only the glucose measurements after the first low glucose measurement (meaning it's repetative if low)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>

    <!-- Will use the closest instance to the beginning of the pattern -->
    <parameters>
        <parameter name="LOW_GLUCOSE_IND" tak="raw-concept" how='before' dynamic="false" idx="0" ref="P1"/>
    </parameters>

    <functions>
        <function name="id">
            <value idx="0"/>
        </function>
    </functions>

</parameterized-raw-concept>
    """

BASE_GLUCOSE_MEASURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASE_GLUCOSE_MEASURE" concept-type="raw-numeric">
  <categories>Measurements</categories>
  <description>Baseline glucose measure used as a parameter reference in ratio computations.</description>
  <attributes>
    <attribute name="BASE_GLUCOSE_MEASURE" type="numeric">
      <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
      </numeric-allowed-values>
    </attribute>
  </attributes>
</raw-concept>
"""

# Divides current GLUCOSE_MEASURE by the closest prior GLUCOSE_MEASURE within a
# configurable back-window.  Use .format(window="Xh") to instantiate.
GLUCOSE_RATIO_GOOD_BEFORE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="GLUCOSE_RATIO">
    <categories>Measurements</categories>
    <description>Glucose divided by the closest prior reading within the good-before window.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="GLUCOSE_MEASURE" tak="raw-concept" how="before" dynamic="true" idx="0" ref="P1" good-before="{window}"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

# Divides current GLUCOSE_MEASURE by the closest BASE_GLUCOSE_MEASURE reading within
# a configurable forward-window.  Use .format(window="Xh") to instantiate.
GLUCOSE_RATIO_GOOD_AFTER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="GLUCOSE_RATIO">
    <categories>Measurements</categories>
    <description>Glucose divided by the closest base reading within the good-after window.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="BASE_GLUCOSE_MEASURE" tak="raw-concept" how="all" dynamic="true" idx="0" ref="P1" good-after="{window}"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

# Invalid: good-after is incompatible with how='before' — must raise at parse time.
GLUCOSE_RATIO_GOOD_AFTER_HOW_BEFORE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="GLUCOSE_RATIO">
    <categories>Measurements</categories>
    <description>Invalid combination: good-after with how=before.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="GLUCOSE_MEASURE" tak="raw-concept" how="before" dynamic="true" idx="0" ref="P1" good-after="1h"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

# Invalid: good-before on a static (dynamic=false) parameter — must raise at parse time.
GLUCOSE_RATIO_GOOD_BEFORE_STATIC_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="GLUCOSE_RATIO">
    <categories>Measurements</categories>
    <description>Invalid combination: good-before with dynamic=false.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="GLUCOSE_MEASURE" tak="raw-concept" how="before" dynamic="false" idx="0" ref="P1" good-before="1h"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

# Mirrors the real STEADY_GLUCOSE_MEASURE_HIGH TAK: emits a reading only when the
# immediately preceding reading within 24 h was also >= 180.
STEADY_GLUCOSE_MEASURE_HIGH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="STEADY_GLUCOSE_MEASURE_HIGH">
    <categories>Measurements</categories>
    <description>Glucose reading emitted only when the immediately preceding reading within 24h was also &gt;= 180.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="GLUCOSE_MEASURE" tak="raw-concept" how="before" dynamic="true" idx="0" ref="P1" good-before="24h"/>
    </parameters>
    <functions>
        <function name="id_if_thresh_met">
            <value idx="0"/>
            <parameter ref="P1"/>
            <literal value="180"/>
            <literal value="ge"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

# Mirrors the real STEADY_GLUCOSE_MEASURE_LOW TAK: emits a reading only when the
# immediately preceding reading within 24 h was also <= 70.
STEADY_GLUCOSE_MEASURE_LOW_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="STEADY_GLUCOSE_MEASURE_LOW">
    <categories>Measurements</categories>
    <description>Glucose reading emitted only when the immediately preceding reading within 24h was also &lt;= 70.</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="GLUCOSE_MEASURE" tak="raw-concept" how="before" dynamic="true" idx="0" ref="P1" good-before="24h"/>
    </parameters>
    <functions>
        <function name="id_if_thresh_met">
            <value idx="0"/>
            <parameter ref="P1"/>
            <literal value="70"/>
            <literal value="le"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""


# -----------------------------
# DF builders (single patient)
# -----------------------------
def df_for_raw_ok():
    """Test exact-timestamp matching (no tolerance)."""
    rows = [
        (1, "BASAL_DOSAGE", make_ts("08:00"), make_ts("08:00"), 20),
        (1, "BASAL_ROUTE" , make_ts("08:00"), make_ts("08:00"), "SubCutaneous"),
        (1, "BASAL_DOSAGE", make_ts("09:00"), make_ts("09:00"), 19),
        (1, "BASAL_ROUTE" , make_ts("10:00"), make_ts("10:00"), "IntraVenous"),
        (1, "BASAL_DOSAGE", make_ts("10:00"), make_ts("10:00"), 30),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])


def df_for_raw_numeric():
    rows = [
        (1, "GLUCOSE_LAB_MEASURE", make_ts("06:00"), make_ts("06:00"), 45),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("07:00"), make_ts("07:00"), 120),
        (1, "GLUCOSE_LAB_MEASURE", make_ts("08:00"), make_ts("08:00"), 420),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])


def df_for_raw_nominal():
    rows = [
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("10:00"), make_ts("10:00"), "Low"),
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("12:00"), make_ts("12:00"), "Medium"),
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("14:00"), make_ts("14:00"), "INVALID"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])


def df_for_raw_boolean():
    rows = [
        (1, "ADMISSION", make_ts("03:00"), make_ts("03:00"), "anything"),
        (1, "ADMISSION", make_ts("04:00"), make_ts("04:00"), ""),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])


def df_for_good_window(*entries):
    """Build a minimal TAK-format DataFrame from (concept, hhmm, day, value) tuples."""
    rows = []
    for concept, hhmm, day, value in entries:
        ts = make_ts(hhmm, day=day)
        rows.append((1, concept, ts, ts, (value,), "raw-concept"))
    return pd.DataFrame(rows, columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])


def df_for_raw_same_timestamp():
    """Test merging when dosage and route have EXACT same timestamp."""
    rows = [
        (1, "BASAL_DOSAGE", make_ts("08:00"), make_ts("08:00"), 20),
        (1, "BASAL_ROUTE" , make_ts("08:00"), make_ts("08:00"), "SubCutaneous"),
        (1, "BASAL_DOSAGE", make_ts("09:00"), make_ts("09:00"), 25),
        (1, "BASAL_ROUTE" , make_ts("09:00"), make_ts("09:00"), "IntraVenous"),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])


# -----------------------------
# Tests
# -----------------------------
def test_parse_validate_raw(tmp_path: Path):
    xml_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_XML)
    tak = RawConcept.parse(xml_path)

    assert tak.name == "BASAL_BITZUA"
    assert tak.concept_type == "raw"
    assert tak.tuple_order == ("BASAL_DOSAGE", "BASAL_ROUTE")
    assert tak.merge_require_all is False

    attr_types = {a["name"]: a["type"] for a in tak.attributes}
    assert attr_types == {"BASAL_DOSAGE": "numeric", "BASAL_ROUTE": "nominal"}


def test_apply_raw_merge_require_all_false(tmp_path: Path):
    xml_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_ok()
    out = tak.apply(df)

    assert len(out) == 3
    assert list(out["ConceptName"].unique()) == [tak.name]

    tuples = list(out["Value"])
    assert tuples[0] == (20, "SubCutaneous")
    assert tuples[1] == (19, None)
    assert tuples[2] == (30, "IntraVenous")


def test_apply_raw_numeric(tmp_path: Path):
    xml_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_NUMERIC_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_numeric()
    out = tak.apply(df)
    assert len(out) == 1
    assert out.iloc[0]["Value"] == (120,)
    assert out.iloc[0]["ConceptName"] == tak.name


def test_apply_raw_nominal(tmp_path: Path):
    xml_path = write_xml(tmp_path, "ANTIDIABETIC_DRUGS_IV_BITZUA.xml", RAW_NOMINAL_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_nominal()
    out = tak.apply(df)

    assert len(out) == 2
    assert tuple(out["Value"]) == (("Low",), ("Medium",))
    assert all(out["ConceptName"] == tak.name)


def test_apply_raw_boolean(tmp_path: Path):
    xml_path = write_xml(tmp_path, "ADMISSION.xml", RAW_BOOLEAN_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_boolean()
    out = tak.apply(df)

    assert len(out) == 2
    assert tuple(out["Value"]) == (("True",), ("True",))
    assert all(out["ConceptName"] == tak.name)


def test_apply_raw_merge_same_timestamp(tmp_path: Path):
    """Test that dosage+route at exact same timestamp merge correctly."""
    xml_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_same_timestamp()
    out = tak.apply(df)

    assert len(out) == 2
    
    assert out.iloc[0]["Value"] == (20, "SubCutaneous")
    assert out.iloc[0]["StartDateTime"] == make_ts("08:00")
    
    assert out.iloc[1]["Value"] == (25, "IntraVenous")
    assert out.iloc[1]["StartDateTime"] == make_ts("09:00")
    
    print("\n✅ Exact-timestamp merge works correctly")


def test_parse_parameterized_raw_concept(tmp_path: Path):
    """Test parsing of parameterized-raw-concept XML."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", PARAM_RAW_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)
    assert tak.name == "M-SHR_MEASURE"
    assert tak.derived_from == "GLUCOSE_MEASURE"
    assert len(tak.parameters) == 1
    assert len(tak.functions) == 1


def test_apply_parameterized_raw_concept(tmp_path: Path):
    """Test apply logic for parameterized-raw-concept."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", PARAM_RAW_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)
    # Simulate input: two glucose measurements, one as parameter
    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 1
    # Should compute 100 / 50 = 2.0
    assert out.iloc[0]["Value"][0] == 2.0
    assert out.iloc[0]["ConceptName"] == "M-SHR_MEASURE"


def test_apply_parameterized_raw_concept_with_default_param(tmp_path: Path):
    """Test apply logic for parameterized-raw-concept with missing param (uses default)."""
    # Ensure the parameter has a default value in the XML
    param_raw_xml_default = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="BASE_GLUCOSE_MEASURE" tak="raw-concept" idx="0" how='before' dynamic='false' ref="P1" default="25"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml_default)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)
    # Simulate input: only main value, no param row
    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 1
    # Should compute 100 / 25 = 4.0
    assert out.iloc[0]["Value"][0] == 4.0
    assert out.iloc[0]["ConceptName"] == "M-SHR_MEASURE"


def test_apply_parameterized_raw_concept_with_param_row(tmp_path: Path):
    """Test apply logic for parameterized-raw-concept with param row present (should use row, not default)."""
    param_raw_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="BASE_GLUCOSE_MEASURE" tak="raw-concept" idx="0" how='before' dynamic='false' ref="P1" default="25"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)
    # Simulate input: main value and param row (param row should be used, not default)
    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 1
    # Should compute 100 / 50 = 2.0 (uses param row, not default)
    assert out.iloc[0]["Value"][0] == 2.0
    assert out.iloc[0]["ConceptName"] == "M-SHR_MEASURE"


def test_parameterized_raw_concept_output_shape(tmp_path: Path):
    """Test that ParameterizedRawConcept output shape matches parent raw concept output (excluding parameter rows)."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", PARAM_RAW_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    parent = RawConcept.parse(glucose_path)
    repo.register(parent)
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)
    # Use scalar for parent input, tuple for parameterized
    parent_input = pd.DataFrame([
        (1, "GLUCOSE_LAB_MEASURE", make_ts("08:00"), make_ts("08:00"), 100),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])
    parent_out = parent.apply(parent_input).reset_index(drop=True)
    df_in = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])
    tak_out = tak.apply(df_in).reset_index(drop=True)
    assert tak_out.shape[0] == parent_out.shape[0]
    for col in ["PatientId", "StartDateTime", "EndDateTime"]:
        assert all(parent_out[col].values == tak_out[col].values)


def test_apply_parameterized_raw_concept_with_how_flag(tmp_path: Path):
    """Test apply logic for parameterized-raw-concept with 'how' flag and different parameter TAKs."""
    # Define the parameterized-raw-concept XML
    param_raw_xml_how = """\
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Raw concept to manage the measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="{param_name}" tak="raw-concept" idx="0" ref="P1" how="{how}" dynamic="{dynamic}"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
"""

    # Register GLUCOSE_MEASURE and BASE_GLUCOSE_MEASURE TAKs
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    base_glucose_path = write_xml(tmp_path, "BASE_GLUCOSE_MEASURE.xml", RAW_NUMERIC_XML.replace("GLUCOSE_MEASURE", "BASE_GLUCOSE_MEASURE"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(base_glucose_path))

    # Case 1: Parameter is GLUCOSE_MEASURE, before the instance
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml_how.format(param_name="GLUCOSE_MEASURE", how="before", dynamic="true"))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # Case 1: Parameter row exists and is before the derived-from row
    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 1 # Expecting 1 row as first row is used as parameter
    assert out.iloc[0]["Value"][0] == 2.0  # 100 / 50
    assert out.iloc[0]["ConceptName"] == "M-SHR_MEASURE"

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (200,), "raw-concept"),

    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 2
    # Row 1 (08:00): 100 / 50 (closest before is 07:00) = 2.0
    assert out.iloc[0]["Value"][0] == 2.0
    # Row 2 (10:00): 200 / 100 (closest before is 08:00) = 2.0
    assert out.iloc[1]["Value"][0] == 2.0
    assert out.iloc[1]["ConceptName"] == "M-SHR_MEASURE"

    # Clean up previous registration to avoid duplicate name error
    if "M-SHR_MEASURE" in repo.taks:
        del repo.taks["M-SHR_MEASURE"]

    # Case 2: Parameter is BASE_GLUCOSE_MEASURE
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml_how.format(param_name="BASE_GLUCOSE_MEASURE", how="all", dynamic="true"))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("11:00"), make_ts("11:00"), (50,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (200,), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 2
    # Row 1 (08:00): 100 / 50 (closest is 11:00) = 2.0
    assert out.iloc[0]["Value"][0] == 2.0
    # Row 2 (10:00): 200 / 50 (closest is 11:00) = 4.0
    assert out.iloc[1]["Value"][0] == 4.0
    assert out.iloc[0]["ConceptName"] == "M-SHR_MEASURE"

    # Clean up previous registration
    if "M-SHR_MEASURE" in repo.taks:
        del repo.taks["M-SHR_MEASURE"]

    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml_how.format(param_name="BASE_GLUCOSE_MEASURE", how="before", dynamic="true"))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("11:00"), make_ts("11:00"), (50,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (200,), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    out = tak.apply(df)
    assert len(out) == 0  # No valid 'before' parameter available

    # Clean up previous registration
    if "M-SHR_MEASURE" in repo.taks:
        del repo.taks["M-SHR_MEASURE"]

    # Case 4: dynamic="false" (Static Baseline)
    # Parameter is BASE_GLUCOSE_MEASURE, how="before"
    # Logic: Resolve parameter ONCE based on first row's time (08:00).
    # Param at 07:00 (50) is valid. Param at 09:00 (80) is ignored even for the 10:00 row.
    param_path = write_xml(tmp_path, "M-SHR_MEASURE.xml", param_raw_xml_how.format(param_name="BASE_GLUCOSE_MEASURE", how="before", dynamic="false"))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (100,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (50,), "raw-concept"),
        (1, "BASE_GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (80,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (200,), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    
    out = tak.apply(df)
    assert len(out) == 2
    
    # Row 1 (08:00): Ref time 08:00. Closest before is 07:00 (50). Result: 100/50 = 2.0
    assert out.iloc[0]["Value"][0] == 2.0
    
    # Row 2 (10:00): Ref time is STILL 08:00 (baseline). Closest before 08:00 is 07:00 (50).
    # Even though 09:00 (80) exists and is before 10:00, it is ignored because dynamic=false.
    # Result: 200/50 = 4.0. (If dynamic=true, it would be 200/80 = 2.5)
    assert out.iloc[1]["Value"][0] == 4.0

def test_steady_glucose_high_pivot_emits_only_after_first_indicator(tmp_path: Path):
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    high_ind_path = write_xml(tmp_path, "HIGH_GLUCOSE_IND.xml", HIGH_GLUCOSE_IND_XML)
    param_path = write_xml(tmp_path, "STEADY_HIGH_AFTER_FIRST.xml", STEADY_GLUCOSE_MEASURE_HIGH_AFTER_FIRST_XML)

    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(high_ind_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("06:00"), make_ts("06:00"), (100,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (110,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("07:30"), make_ts("07:30"), (210,), "raw-concept"),
        (1, "HIGH_GLUCOSE_IND", make_ts("08:00"), make_ts("08:00"), ("True",), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (200,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (150,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    out = tak.apply(df)
    assert len(out) == 2
    assert list(out["StartDateTime"]) == [make_ts("09:00"), make_ts("10:00")]

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("06:00"), make_ts("06:00"), (100,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("08:00"), make_ts("08:00"), (190,), "raw-concept"),
        (1, "HIGH_GLUCOSE_IND", make_ts("08:00"), make_ts("08:00"), ("True",), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("09:00"), make_ts("09:00"), (200,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("10:00"), make_ts("10:00"), (150,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    out = tak.apply(df)
    assert len(out) == 2
    assert list(out["StartDateTime"]) == [make_ts("09:00"), make_ts("10:00")]


def test_steady_glucose_high_no_indicator_emits_nothing(tmp_path: Path):
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    high_ind_path = write_xml(tmp_path, "HIGH_GLUCOSE_IND.xml", HIGH_GLUCOSE_IND_XML)
    param_path = write_xml(tmp_path, "STEADY_HIGH_AFTER_FIRST.xml", STEADY_GLUCOSE_MEASURE_HIGH_AFTER_FIRST_XML)

    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(high_ind_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = pd.DataFrame([
        (1, "GLUCOSE_MEASURE", make_ts("06:00"), make_ts("06:00"), (100,), "raw-concept"),
        (1, "GLUCOSE_MEASURE", make_ts("07:00"), make_ts("07:00"), (110,), "raw-concept"),
    ], columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

    out = tak.apply(df)
    assert out.empty


def test_apply_creatinine_rel_serum_measure(tmp_path: Path):
    """Test CREATININE_REL_SERUM_MEASURE parameterized-raw-concept with multiple creatinine values."""
    # Write the raw and parameterized concept XMLs
    creatinine_raw_path = write_xml(tmp_path, "CREATININE_SERUM_MEASURE.xml", '''<?xml version="1.0" encoding="UTF-8"?>\n<raw-concept name="CREATININE_SERUM_MEASURE" concept-type="raw-numeric">\n    <categories>Measurements</categories>\n    <description>Raw concept to manage the measurement of CREATININE by Serum using mg/dL units</description>\n    <attributes>\n        <attribute name="CREATININE_SERUM_MEASURE" type="numeric">\n            <numeric-allowed-values>\n                <allowed-value min="0.1" max="20"/>\n            </numeric-allowed-values>\n        </attribute>\n    </attributes>\n</raw-concept>''')
    creatinine_rel_path = write_xml(tmp_path, "CREATININE_REL_SERUM_MEASURE.xml", '''<?xml version="1.0" encoding="UTF-8"?>\n<parameterized-raw-concept name="CREATININE_REL_SERUM_MEASURE">\n    <categories>Measurements</categories>\n    <description>Raw concept to monitor the creatinine trend within admission, if 2 or more exist</description>\n    <derived-from name="CREATININE_SERUM_MEASURE" tak="raw-concept"/>\n    <parameters>\n        <parameter name="CREATININE_SERUM_MEASURE" tak="raw-concept" idx="0" how='before' dynamic="false" ref="P1"/>\n    </parameters>\n    <functions>\n        <function name="subtract">\n            <value idx="0"/>\n            <parameter ref="P1"/>\n        </function>\n    </functions>\n</parameterized-raw-concept>''')
    # Register in repo
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(creatinine_raw_path))
    tak = ParameterizedRawConcept.parse(creatinine_rel_path)
    repo.register(tak)
    # Simulate input: multiple creatinine measures for a patient
    def make_ts(dtstr):
        return pd.Timestamp(dtstr)
    df = pd.DataFrame([
        (1, "CREATININE_SERUM_MEASURE", make_ts("2022-01-01 08:00"), make_ts("2022-01-01 08:00"), (0.8,), "raw-concept"),
        (1, "CREATININE_SERUM_MEASURE", make_ts("2022-01-02 09:00"), make_ts("2022-01-02 09:00"), (1.0,), "raw-concept"),
        (1, "CREATININE_SERUM_MEASURE", make_ts("2022-01-03 10:00"), make_ts("2022-01-03 10:00"), (0.7,), "raw-concept"),
    ], columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
    out = tak.apply(df)
    # Should produce one row for each value after the first (so 2 rows)
    assert len(out) == 2, f"Expected 2 output rows, got {len(out)}. Output: {out}"
    # Check values: (1.0-0.8)=0.2, (0.7-0.8)=-0.1
    assert abs(out.iloc[0]["Value"][0] - 0.2) < 1e-6, f"First diff wrong: {out.iloc[0]['Value'][0]}"
    assert abs(out.iloc[1]["Value"][0] + 0.1) < 1e-6, f"Second diff wrong: {out.iloc[1]['Value'][0]}"
    assert all(out["ConceptName"] == "CREATININE_REL_SERUM_MEASURE")
    print("\n✅ CREATININE_REL_SERUM_MEASURE produces output as expected")


# ------------------------------------------------------------------
# good-before / good-after window tests
# ------------------------------------------------------------------

def test_good_before_within_window_uses_recent_prior(tmp_path):
    """good-before="2h": only prior readings within 2 h are eligible as parameter."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path   = write_xml(tmp_path, "GLUCOSE_RATIO.xml", GLUCOSE_RATIO_GOOD_BEFORE_XML.format(window="2h"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # Readings at 07:00 (100), 07:30 (80), 10:00 (200).
    # For the 10:00 row, good-before="2h" means parameter must be >= 08:00.
    # Both 07:00 and 07:30 fall outside → P1=None → row is skipped.
    # 07:30 has 07:00 prior which is within its own 2 h window → emits 80/100=0.8.
    df = df_for_good_window(
        ("GLUCOSE_MEASURE", "07:00", 0, 100),
        ("GLUCOSE_MEASURE", "07:30", 0, 80),
        ("GLUCOSE_MEASURE", "10:00", 0, 200),
    )
    out = tak.apply(df)
    assert len(out) == 1
    assert abs(out.iloc[0]["Value"][0] - 0.8) < 1e-6
    assert out.iloc[0]["StartDateTime"] == make_ts("07:30")


def test_good_before_all_priors_outside_window_skips_row(tmp_path):
    """good-before="30m": when all prior readings are older than 30 min, the row is skipped."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path   = write_xml(tmp_path, "GLUCOSE_RATIO.xml", GLUCOSE_RATIO_GOOD_BEFORE_XML.format(window="30m"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # 07:00 (100) is 3 h before 10:00 — outside the 30-min window → P1=None → both rows skipped.
    df = df_for_good_window(
        ("GLUCOSE_MEASURE", "07:00", 0, 100),
        ("GLUCOSE_MEASURE", "10:00", 0, 200),
    )
    out = tak.apply(df)
    assert out.empty


def test_good_before_reading_just_inside_window_is_used(tmp_path):
    """good-before="3h": a reading exactly at the window boundary is included."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path   = write_xml(tmp_path, "GLUCOSE_RATIO.xml", GLUCOSE_RATIO_GOOD_BEFORE_XML.format(window="3h"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # 07:00 is exactly 3 h before 10:00 — sits on the boundary (>=) and should qualify.
    df = df_for_good_window(
        ("GLUCOSE_MEASURE", "07:00", 0, 100),
        ("GLUCOSE_MEASURE", "10:00", 0, 200),
    )
    out = tak.apply(df)
    assert len(out) == 1
    assert abs(out.iloc[0]["Value"][0] - 2.0) < 1e-6   # 200 / 100


def test_good_after_within_window_uses_near_future_reading(tmp_path):
    """good-after="4h": a BASE reading 3 h ahead is included; one 5 h ahead is excluded."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    base_path    = write_xml(tmp_path, "BASE_GLUCOSE_MEASURE.xml", BASE_GLUCOSE_MEASURE_XML)
    param_path   = write_xml(tmp_path, "GLUCOSE_RATIO.xml", GLUCOSE_RATIO_GOOD_AFTER_XML.format(window="4h"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(base_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # GLUCOSE_MEASURE at 10:00 (100); BASE readings at 13:00 (50) and 16:00 (80).
    # good-after="4h" → parameter must be <= 14:00.  13:00 qualifies, 16:00 does not.
    df = df_for_good_window(
        ("GLUCOSE_MEASURE",      "10:00", 0, 100),
        ("BASE_GLUCOSE_MEASURE", "13:00", 0, 50),
        ("BASE_GLUCOSE_MEASURE", "16:00", 0, 80),
    )
    out = tak.apply(df)
    assert len(out) == 1
    assert abs(out.iloc[0]["Value"][0] - 2.0) < 1e-6   # 100 / 50


def test_good_after_all_readings_outside_window_skips_row(tmp_path):
    """good-after="2h": when the only BASE reading is beyond the window, the row is skipped."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    base_path    = write_xml(tmp_path, "BASE_GLUCOSE_MEASURE.xml", BASE_GLUCOSE_MEASURE_XML)
    param_path   = write_xml(tmp_path, "GLUCOSE_RATIO.xml", GLUCOSE_RATIO_GOOD_AFTER_XML.format(window="2h"))
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    repo.register(RawConcept.parse(base_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    # BASE reading at 15:00 is 5 h after 10:00 — outside a 2-h good-after window.
    df = df_for_good_window(
        ("GLUCOSE_MEASURE",      "10:00", 0, 100),
        ("BASE_GLUCOSE_MEASURE", "15:00", 0, 80),
    )
    out = tak.apply(df)
    assert out.empty


def test_good_after_with_how_before_raises_at_parse(tmp_path):
    """good-after combined with how='before' must raise a ValueError at parse time."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    with pytest.raises(ValueError, match="good-after.*how='before'"):
        ParameterizedRawConcept.parse(write_xml(tmp_path, "BAD.xml", GLUCOSE_RATIO_GOOD_AFTER_HOW_BEFORE_XML))


def test_good_before_with_dynamic_false_raises_at_parse(tmp_path):
    """good-before on a static parameter (dynamic=false) must raise a ValueError at parse time."""
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    with pytest.raises(ValueError, match="good-before/good-after.*dynamic=true"):
        ParameterizedRawConcept.parse(write_xml(tmp_path, "BAD.xml", GLUCOSE_RATIO_GOOD_BEFORE_STATIC_XML))


def test_id_if_thresh_met_gate_consecutive_high_readings(tmp_path):
    """
    Full integration test for STEADY_GLUCOSE_MEASURE_HIGH logic:
    a row is emitted only when the immediately preceding GLUCOSE_MEASURE
    within 24 h was also >= 180.
    """
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path   = write_xml(tmp_path, "STEADY_GLUCOSE_MEASURE_HIGH.xml", STEADY_GLUCOSE_MEASURE_HIGH_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = df_for_good_window(
        ("GLUCOSE_MEASURE", "06:00", 0, 150),  # no prior → skip
        ("GLUCOSE_MEASURE", "10:00", 0, 200),  # prior 06:00 = 150 < 180 → gate drops → skip
        ("GLUCOSE_MEASURE", "14:00", 0, 220),  # prior 10:00 = 200 >= 180 → emit
        ("GLUCOSE_MEASURE", "18:00", 0, 190),  # prior 14:00 = 220 >= 180 → emit
        ("GLUCOSE_MEASURE", "08:00", 2, 185),  # day 3: all priors > 24 h ago → skip
    )
    out = tak.apply(df)

    assert len(out) == 2
    assert out.iloc[0]["StartDateTime"] == make_ts("14:00", day=0)
    assert out.iloc[0]["Value"][0] == 220
    assert out.iloc[1]["StartDateTime"] == make_ts("18:00", day=0)
    assert out.iloc[1]["Value"][0] == 190


def test_id_if_thresh_met_gate_consecutive_low_readings(tmp_path):
    """
    Full integration test for STEADY_GLUCOSE_MEASURE_LOW logic:
    a row is emitted only when the immediately preceding GLUCOSE_MEASURE
    within 24 h was also <= 70.
    """
    glucose_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_GLUCOSE_XML)
    param_path   = write_xml(tmp_path, "STEADY_GLUCOSE_MEASURE_LOW.xml", STEADY_GLUCOSE_MEASURE_LOW_XML)
    repo = TAKRepository()
    set_tak_repository(repo)
    repo.register(RawConcept.parse(glucose_path))
    tak = ParameterizedRawConcept.parse(param_path)
    repo.register(tak)

    df = df_for_good_window(
        ("GLUCOSE_MEASURE", "06:00", 0, 100),  # no prior → skip
        ("GLUCOSE_MEASURE", "10:00", 0,  60),  # prior 06:00 = 100 > 70 → gate drops → skip
        ("GLUCOSE_MEASURE", "14:00", 0,  65),  # prior 10:00 = 60 <= 70 → emit
        ("GLUCOSE_MEASURE", "18:00", 0,  55),  # prior 14:00 = 65 <= 70 → emit
        ("GLUCOSE_MEASURE", "08:00", 2,  68),  # day 3: all priors > 24 h ago → skip
    )
    out = tak.apply(df)

    assert len(out) == 2
    assert out.iloc[0]["StartDateTime"] == make_ts("14:00", day=0)
    assert out.iloc[0]["Value"][0] == 65
    assert out.iloc[1]["StartDateTime"] == make_ts("18:00", day=0)
    assert out.iloc[1]["Value"][0] == 55
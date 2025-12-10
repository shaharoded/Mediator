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
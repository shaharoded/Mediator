"""
Comprehensive unit tests for RawConcept TAK.
"""
import pandas as pd
import pytest
from pathlib import Path

from core.tak.raw_concept import RawConcept
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
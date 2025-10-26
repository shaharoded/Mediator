# unittests/test_raw_concept.py
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Import the class under test (adjust if your package name differs)
from core.tak.raw_concept import RawConcept
from core.tak.utils import parse_duration  # must exist per your implementation


# -----------------------------
# Helpers: Write XMLs to disk
# -----------------------------
def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


# -----------------------------
# XML Fixtures (schema-aligned)
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

  <merge tolerance="15m" require-all="false"/>
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
def make_ts(hhmm: str, day: int = 0):
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)

def df_for_raw_ok():
    # Windows:
    #  - Window 1 (08:00..08:05): dosage+route -> tuple emitted
    #  - Window 2 (09:00..09:01): dosage only -> partial allowed (require-all=False) so emitted with None
    rows = [
        # full tuple
        (1, "BASAL_DOSAGE", make_ts("08:00"), make_ts("08:00"), 20),
        (1, "BASAL_ROUTE" , make_ts("08:05"), make_ts("08:05"), "SubCutaneous"),
        # partial (route missing)
        (1, "BASAL_DOSAGE", make_ts("09:00"), make_ts("09:00"), 18),
        (1, "BASAL_DOSAGE", make_ts("09:01"), make_ts("09:01"), 19),
        # gap > 15m → new window
        (1, "BASAL_ROUTE" , make_ts("10:30"), make_ts("10:30"), "IntraVenous"),
        (1, "BASAL_DOSAGE", make_ts("10:35"), make_ts("10:35"), 30),
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

def df_for_raw_numeric():
    rows = [
        (1, "GLUCOSE_LAB_MEASURE", make_ts("06:00"), make_ts("06:00"), 45),   # filtered (too low)
        (1, "GLUCOSE_LAB_MEASURE", make_ts("07:00"), make_ts("07:00"), 120),  # kept
        (1, "GLUCOSE_LAB_MEASURE", make_ts("08:00"), make_ts("08:00"), 420),  # filtered (too high)
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

def df_for_raw_nominal():
    rows = [
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("10:00"), make_ts("10:00"), "Low"),
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("12:00"), make_ts("12:00"), "Medium"),
        (1, "ANTIDIABETIC_DRUGS_IV_DOSAGE", make_ts("14:00"), make_ts("14:00"), "INVALID"),  # filtered
    ]
    return pd.DataFrame(rows, columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value"])

def df_for_raw_boolean():
    rows = [
        (1, "ADMISSION", make_ts("03:00"), make_ts("03:00"), "anything"),
        (1, "ADMISSION", make_ts("04:00"), make_ts("04:00"), ""),  # still counts
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
    assert tak.merge_tolerance == parse_duration("15m")
    assert tak.merge_require_all is False

    # attributes parsed
    attr_types = {a["name"]: a["type"] for a in tak.attributes}
    assert attr_types == {"BASAL_DOSAGE": "numeric", "BASAL_ROUTE": "nominal"}

    # nominal allowed
    route_attr = next(a for a in tak.attributes if a["name"] == "BASAL_ROUTE")
    assert set(route_attr["allowed"]) == {"SubCutaneous", "IntraVenous"}

    # numeric min/max
    dose_attr = next(a for a in tak.attributes if a["name"] == "BASAL_DOSAGE")
    assert dose_attr["min"] == 0 and dose_attr["max"] == 100


def test_apply_raw_merge_require_all_false(caplog, tmp_path: Path):
    xml_path = write_xml(tmp_path, "BASAL_BITZUA.xml", RAW_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_ok()
    out = tak.apply(df)

    # Expect 3 merged windows:
    # 1) (20, "SubCutaneous")
    # 2) (19, None)  -- partial allowed
    # 3) (30, "IntraVenous")
    assert len(out) == 3
    assert list(out["ConceptName"].unique()) == [tak.name]
    assert all(v["AbstractionType"] == tak.family for _, v in out.iterrows())

    tuples = list(out["Value"])
    assert tuples[0] == (20, "SubCutaneous")
    assert tuples[1] == (19, None)
    # order within window 3 might produce (30, "IntraVenous") based on last values
    assert tuples[2] == (30, "IntraVenous")

    # Should log at least one info/warning about window sizes vs tuple_order
    assert any("RAW" in rec.msg for rec in caplog.records)


def test_apply_raw_numeric(tmp_path: Path):
    xml_path = write_xml(tmp_path, "GLUCOSE_MEASURE.xml", RAW_NUMERIC_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_numeric()
    out = tak.apply(df)
    # Only the middle value (120) should remain
    assert len(out) == 1
    assert out.iloc[0]["Value"] == (120,)
    assert out.iloc[0]["ConceptName"] == tak.name


def test_apply_raw_nominal(tmp_path: Path):
    xml_path = write_xml(tmp_path, "ANTIDIABETIC_DRUGS_IV_BITZUA.xml", RAW_NOMINAL_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_nominal()
    out = tak.apply(df)

    # The "INVALID" row should be filtered out; remaining wrapped as 1-tuples
    assert len(out) == 2
    assert tuple(out["Value"]) == (("Low",), ("Medium",))
    assert all(out["ConceptName"] == tak.name)


def test_apply_raw_boolean(tmp_path: Path):
    xml_path = write_xml(tmp_path, "ADMISSION.xml", RAW_BOOLEAN_XML)
    tak = RawConcept.parse(xml_path)

    df = df_for_raw_boolean()
    out = tak.apply(df)

    # Both rows should become ("True",) — STRING not Python bool
    assert len(out) == 2
    assert tuple(out["Value"]) == (("True",), ("True",))
    assert all(out["ConceptName"] == tak.name)
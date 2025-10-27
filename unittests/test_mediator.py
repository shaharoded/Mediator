"""
Unit tests for Mediator orchestration logic.

Tests cover:
1. TAK repository building (parsing, validation, dependency order)
2. Patient data extraction (input queries for different TAK families)
3. Output writing (DB insertion)
4. Error handling (malformed TAKs, missing dependencies)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from core.mediator import Mediator
from backend.dataaccess import DataAccess


def make_ts(hhmm: str, day: int = 0) -> datetime:
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


@pytest.fixture
def temp_kb(tmp_path: Path) -> Path:
    """Create temporary knowledge base structure."""
    kb_path = tmp_path / "knowledge-base"
    (kb_path / "raw-concepts").mkdir(parents=True)
    (kb_path / "events").mkdir(parents=True)
    (kb_path / "states").mkdir(parents=True)
    (kb_path / "trends").mkdir(parents=True)
    (kb_path / "contexts").mkdir(parents=True)
    return kb_path


@pytest.fixture
def temp_db(tmp_path: Path) -> DataAccess:
    """Create temporary test database."""
    db_path = tmp_path / "test.db"
    da = DataAccess(db_path=str(db_path))
    da.create_db(drop=True)
    return da


@pytest.fixture
def sample_kb(temp_kb: Path) -> Path:
    """Populate KB with sample TAKs."""
    # Raw concept
    (temp_kb / "raw-concepts" / "GLUCOSE.xml").write_text("""
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
""")
    
    # State
    (temp_kb / "states" / "GLUCOSE_STATE.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_STATE">
  <categories>Measurements</categories>
  <description>Glucose state</description>
  <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
  <persistence good-after="24h" interpolate="false" max-skip="0"/>
  <discretization-rules>
    <attribute idx="0">
      <rule value="Low" min="0" max="70"/>
      <rule value="Normal" min="70" max="180"/>
      <rule value="High" min="180"/>
    </attribute>
  </discretization-rules>
</state>
""")
    
    return temp_kb


def test_build_repository_success(temp_db, sample_kb):
    """Test successful TAK repository build."""
    mediator = Mediator(sample_kb, temp_db)
    repo = mediator.build_repository()
    
    assert len(repo.taks) == 2
    assert "GLUCOSE_MEASURE" in repo.taks
    assert "GLUCOSE_STATE" in repo.taks
    assert len(mediator.raw_concepts) == 1
    assert len(mediator.states) == 1


def test_build_repository_missing_dependency(temp_kb, temp_db):
    """Test repository build fails when parent TAK is missing."""
    # State without parent raw-concept
    (temp_kb / "states" / "ORPHAN_STATE.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<state name="ORPHAN_STATE">
  <categories>Test</categories>
  <description>Orphan state</description>
  <derived-from name="NONEXISTENT_RAW" tak="raw-concept"/>
  <persistence good-after="1h" interpolate="false" max-skip="0"/>
</state>
""")
    
    mediator = Mediator(temp_kb, temp_db)
    with pytest.raises(RuntimeError, match="validation failed"):
        mediator.build_repository()


def test_get_patient_ids(temp_db, sample_kb):
    """Test patient ID extraction."""
    # Insert test data
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (2, "GLUCOSE_LAB", "2024-01-01 09:00:00", "2024-01-01 09:00:00", "150")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    patient_ids = mediator.get_patient_ids()
    
    assert patient_ids == [1, 2]


def test_get_input_for_raw_concept(temp_db, sample_kb):
    """Test input extraction for RawConcept TAK."""
    # Insert test data
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    raw_concept = mediator.raw_concepts[0]
    df = mediator.get_input_for_tak(1, raw_concept)
    
    assert len(df) == 1
    assert df.iloc[0]["ConceptName"] == "GLUCOSE_LAB"
    assert df.iloc[0]["Value"] == "120"


def test_write_output(temp_db, sample_kb):
    """Test output writing to OutputPatientData."""
    mediator = Mediator(sample_kb, temp_db)
    
    df_output = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 12:00:00",
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    rows_written = mediator.write_output(df_output)
    assert rows_written == 1
    
    # Verify DB write
    rows = temp_db.fetch_records(
        "SELECT ConceptName, Value FROM OutputPatientData WHERE PatientId = ?",
        (1,)
    )
    assert len(rows) == 1
    assert rows[0] == ("GLUCOSE_STATE", "Normal")


def test_process_patient_end_to_end(temp_db, sample_kb):
    """Test complete patient processing pipeline."""
    # Insert raw input
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 12:00:00", "2024-01-01 12:00:00", "200")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    # Process patient
    stats = mediator._process_patient_sync(1)
    
    # Check stats
    assert "GLUCOSE_MEASURE" in stats
    assert "GLUCOSE_STATE" in stats
    assert stats["GLUCOSE_STATE"] > 0
    
    # Verify DB output
    rows = temp_db.fetch_records(
        "SELECT ConceptName, Value FROM OutputPatientData WHERE PatientId = ? ORDER BY StartDateTime",
        (1,)
    )
    assert len(rows) >= 1  # At least one state interval
    assert all(r[0] == "GLUCOSE_STATE" for r in rows)

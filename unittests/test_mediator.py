"""
Comprehensive unit tests for Mediator orchestration logic.

Tests cover:
1. TAK repository building (parsing, validation, dependency order)
2. Patient data extraction (input queries for different TAK families)
3. Global clippers (START/END boundary trimming)
4. Patient subsetting (process specific patients vs all)
5. Full caching (raw-concepts cached, no redundant computation)
6. Output writing (DB insertion with deduplication)
7. Multi-patient async processing
8. Error handling (malformed TAKs, missing dependencies)
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio

from core.mediator import Mediator
from backend.dataaccess import DataAccess


def make_ts(hhmm: str, day: int = 0) -> datetime:
    """Build timestamp: 2024-01-01 + day offset + HH:MM."""
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def temp_kb(tmp_path: Path) -> Path:
    """Create temporary knowledge base structure."""
    kb_path = tmp_path / "knowledge-base"
    (kb_path / "raw-concepts").mkdir(parents=True)
    (kb_path / "events").mkdir(parents=True)
    (kb_path / "states").mkdir(parents=True)
    (kb_path / "trends").mkdir(parents=True)
    (kb_path / "contexts").mkdir(parents=True)
    
    # Create global_clippers.json
    (kb_path / "global_clippers.json").write_text("""{
    "global_clippers": [
        {"name": "ADMISSION", "how": "START"},
        {"name": "DEATH", "how": "END"}
    ]
}""")
    
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
    """Populate KB with sample TAKs (raw-concept → event → state → trend)."""
    
    # Raw concept: GLUCOSE_MEASURE
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
    
    # Raw concept: ADMISSION (global clipper)
    (temp_kb / "raw-concepts" / "ADMISSION.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="ADMISSION" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Admission event</description>
  <attributes>
    <attribute name="ADMISSION" type="boolean"/>
  </attributes>
</raw-concept>
""")
    
    # Raw concept: DEATH (global clipper)
    (temp_kb / "raw-concepts" / "DEATH.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="DEATH" concept-type="raw-boolean">
  <categories>Admin</categories>
  <description>Death event</description>
  <attributes>
    <attribute name="DEATH" type="boolean"/>
  </attributes>
</raw-concept>
""")
    
    # Event: HYPERGLYCEMIA_EVENT
    (temp_kb / "events" / "HYPERGLYCEMIA.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<event name="HYPERGLYCEMIA_EVENT">
  <categories>Events</categories>
  <description>Hyperglycemia event</description>
  <derived-from>
    <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
  </derived-from>
  <abstraction-rules>
    <rule value="Hyperglycemia" operator="or">
      <attribute name="GLUCOSE_MEASURE" idx="0">
        <allowed-value min="250"/>
      </attribute>
    </rule>
  </abstraction-rules>
</event>
""")
    
    # State: GLUCOSE_STATE
    (temp_kb / "states" / "GLUCOSE_STATE.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_STATE">
  <categories>Measurements</categories>
  <description>Glucose state</description>
  <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
  <persistence good-after="6h" interpolate="false" max-skip="0"/>
  <discretization-rules>
    <attribute idx="0">
      <rule value="Low" min="0" max="70"/>
      <rule value="Normal" min="70" max="180"/>
      <rule value="High" min="180"/>
    </attribute>
  </discretization-rules>
</state>
""")
    
    # Trend: GLUCOSE_TREND
    (temp_kb / "trends" / "GLUCOSE_TREND.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<trend name="GLUCOSE_TREND">
  <categories>Measurements</categories>
  <description>Glucose trend</description>
  <derived-from name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" significant-variation="40"/>
  <time-steady value="6h"/>
  <persistence good-after="6h"/>
</trend>
""")
    
    return temp_kb


# -----------------------------
# Tests: Repository Building
# -----------------------------

def test_build_repository_success(temp_db, sample_kb):
    """Test successful TAK repository build."""
    mediator = Mediator(sample_kb, temp_db)
    repo = mediator.build_repository()
    
    assert len(repo.taks) == 6  # 3 raw + 1 event + 1 state + 1 trend
    assert "GLUCOSE_MEASURE" in repo.taks
    assert "ADMISSION" in repo.taks
    assert "DEATH" in repo.taks
    assert "HYPERGLYCEMIA_EVENT" in repo.taks
    assert "GLUCOSE_STATE" in repo.taks
    assert "GLUCOSE_TREND" in repo.taks
    
    # Check execution order
    assert len(mediator.raw_concepts) == 3
    assert len(mediator.events) == 1
    assert len(mediator.states) == 1
    assert len(mediator.trends) == 1


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
    
    mediator = Mediator(sample_kb, temp_db)
    with pytest.raises(RuntimeError, match="validation failed"):
        mediator.build_repository()


def test_load_global_clippers(temp_db, sample_kb):
    """Test global clippers loaded from JSON."""
    mediator = Mediator(sample_kb, temp_db)
    
    assert len(mediator.global_clippers) == 2
    assert mediator.global_clippers["ADMISSION"] == "START"
    assert mediator.global_clippers["DEATH"] == "END"


# -----------------------------
# Tests: Patient ID Extraction
# -----------------------------

def test_get_patient_ids_all(temp_db, sample_kb):
    """Test patient ID extraction (all patients)."""
    # Insert test data for 3 patients
    for pid in [1, 2, 3]:
        temp_db.execute_query(
            "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
            (pid, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
        )
    
    mediator = Mediator(sample_kb, temp_db)
    patient_ids = mediator.get_patient_ids()
    
    assert patient_ids == [1, 2, 3]


def test_get_patient_ids_subset(temp_db, sample_kb):
    """Test patient ID extraction with subset filter."""
    # Insert test data for 3 patients
    for pid in [1, 2, 3]:
        temp_db.execute_query(
            "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
            (pid, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
        )
    
    mediator = Mediator(sample_kb, temp_db)
    patient_ids = mediator.get_patient_ids(patient_subset=[1, 3, 99])  # 99 doesn't exist
    
    assert patient_ids == [1, 3]  # Only existing patients returned


# -----------------------------
# Tests: Input Data Extraction
# -----------------------------

def test_get_input_for_raw_concept(temp_db, sample_kb):
    """Test input extraction for RawConcept TAK."""
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
    
    raw_concept = mediator.raw_concepts[0]  # GLUCOSE_MEASURE
    df = mediator.get_input_for_raw_concept(1, raw_concept)
    
    assert len(df) == 2
    assert all(df["ConceptName"] == "GLUCOSE_LAB")
    assert list(df["Value"]) == ["120", "200"]


def test_get_input_for_tak_caching(temp_db, sample_kb):
    """Test that get_input_for_tak uses cache (no redundant queries)."""
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    raw_concept = mediator.raw_concepts[0]
    tak_outputs = {}
    
    # First call: should query DB
    df1 = mediator.get_input_for_tak(1, raw_concept, tak_outputs)
    assert len(df1) == 1
    assert raw_concept.name in tak_outputs  # Cached
    
    # Second call: should use cache (no query)
    df2 = mediator.get_input_for_tak(1, raw_concept, tak_outputs)
    assert len(df2) == 1
    assert df1.equals(df2)  # Same DataFrame


# -----------------------------
# Tests: Output Writing
# -----------------------------

def test_write_output(temp_db, sample_kb):
    """Test output writing to OutputPatientData."""
    mediator = Mediator(sample_kb, temp_db)
    
    df_output = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 14:00:00",
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    rows_written = mediator.write_output(df_output)
    assert rows_written == 1
    
    # Verify DB write
    rows = temp_db.fetch_records(
        "SELECT ConceptName, Value, AbstractionType FROM OutputPatientData WHERE PatientId = ?",
        (1,)
    )
    assert len(rows) == 1
    assert rows[0] == ("GLUCOSE_STATE", "Normal", "state")


def test_write_output_deduplication(temp_db, sample_kb):
    """Test INSERT OR IGNORE prevents duplicate writes."""
    mediator = Mediator(sample_kb, temp_db)
    
    df_output = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 14:00:00",
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    # Write twice
    rows1 = mediator.write_output(df_output)
    rows2 = mediator.write_output(df_output)
    
    assert rows1 == 1
    assert rows2 == 0  # Duplicate ignored
    
    # Verify only 1 row in DB
    rows = temp_db.fetch_records("SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?", (1,))
    assert rows[0][0] == 1


# -----------------------------
# Tests: Global Clippers
# -----------------------------

def test_global_clippers_start_trim(temp_db, sample_kb):
    """Test START clipper (ADMISSION) trims interval start."""
    # Insert ADMISSION at 08:00
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "ADMISSION", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "True")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    clipper_df = mediator._get_global_clipper_times(1)
    
    assert clipper_df is not None
    assert len(clipper_df) == 1
    assert clipper_df.iloc[0]["ConceptName"] == "ADMISSION"
    
    # Test clipping: interval [07:00 → 10:00] should become [08:00:01 → 10:00]
    df_before = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": pd.to_datetime("2024-01-01 07:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 10:00:00"),
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    df_after = mediator._apply_global_clippers(df_before, clipper_df)
    
    assert len(df_after) == 1
    assert df_after.iloc[0]["StartDateTime"] == pd.to_datetime("2024-01-01 08:00:01")
    assert df_after.iloc[0]["EndDateTime"] == pd.to_datetime("2024-01-01 10:00:00")


def test_global_clippers_end_trim(temp_db, sample_kb):
    """Test END clipper (DEATH) trims interval end."""
    # Insert DEATH at 18:00
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "DEATH", "2024-01-01 18:00:00", "2024-01-01 18:00:00", "True")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    clipper_df = mediator._get_global_clipper_times(1)
    
    assert clipper_df is not None
    assert len(clipper_df) == 1
    assert clipper_df.iloc[0]["ConceptName"] == "DEATH"
    
    # Test clipping: interval [15:00 → 20:00] should become [15:00 → 17:59:59]
    df_before = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": pd.to_datetime("2024-01-01 15:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 20:00:00"),
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    df_after = mediator._apply_global_clippers(df_before, clipper_df)
    
    assert len(df_after) == 1
    assert df_after.iloc[0]["StartDateTime"] == pd.to_datetime("2024-01-01 15:00:00")
    assert df_after.iloc[0]["EndDateTime"] == pd.to_datetime("2024-01-01 17:59:59")


def test_global_clippers_invalid_interval_dropped(temp_db, sample_kb):
    """Test that intervals with StartDateTime >= EndDateTime after clipping are dropped."""
    # Insert ADMISSION and DEATH very close together
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "ADMISSION", "2024-01-01 10:00:00", "2024-01-01 10:00:00", "True")
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "DEATH", "2024-01-01 10:00:30", "2024-01-01 10:00:30", "True")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    clipper_df = mediator._get_global_clipper_times(1)
    
    # Interval that would flip after clipping
    df_before = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "GLUCOSE_STATE",
            "StartDateTime": pd.to_datetime("2024-01-01 09:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 11:00:00"),
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    df_after = mediator._apply_global_clippers(df_before, clipper_df)
    
    # Interval should be dropped (StartDateTime 10:00:01 >= EndDateTime 10:00:29)
    assert len(df_after) == 0


# -----------------------------
# Tests: Patient Processing
# -----------------------------

def test_process_patient_single(temp_db, sample_kb):
    """Test single patient processing (raw → event → state → trend)."""
    # Insert glucose measurements
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "100")
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 12:00:00", "2024-01-01 12:00:00", "300")
    )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    stats = mediator._process_patient_sync(1)
    
    # Check stats (raw-concepts cached, not written)
    assert "GLUCOSE_MEASURE" in stats
    assert stats["GLUCOSE_MEASURE"] == 2  # Cached count
    
    # Check outputs (event, state, trend written)
    assert "HYPERGLYCEMIA_EVENT" in stats
    assert "GLUCOSE_STATE" in stats
    assert "GLUCOSE_TREND" in stats
    
    # Verify DB writes
    rows = temp_db.fetch_records(
        "SELECT ConceptName, AbstractionType FROM OutputPatientData WHERE PatientId = ? ORDER BY ConceptName",
        (1,)
    )
    assert len(rows) >= 3  # At least 1 event + 1 state + 1 trend
    concept_names = {r[0] for r in rows}
    assert "HYPERGLYCEMIA_EVENT" in concept_names
    assert "GLUCOSE_STATE" in concept_names
    assert "GLUCOSE_TREND" in concept_names


def test_process_patient_with_global_clippers(temp_db, sample_kb):
    """Test patient processing with global clippers applied."""
    # Insert ADMISSION and DEATH
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "ADMISSION", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "True")
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "DEATH", "2024-01-01 18:00:00", "2024-01-01 18:00:00", "True")
    )
    
    # Insert glucose measurement (should be clipped)
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 07:00:00", "2024-01-01 07:00:00", "120")  # Before admission
    )
    temp_db.execute_query(
        "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
        (1, "GLUCOSE_LAB", "2024-01-01 10:00:00", "2024-01-01 10:00:00", "150")  # Within window
    )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    stats = mediator._process_patient_sync(1)
    
    # Verify states were clipped (StartDateTime > ADMISSION, EndDateTime < DEATH)
    rows = temp_db.fetch_records(
        "SELECT StartDateTime, EndDateTime FROM OutputPatientData WHERE PatientId = ? AND ConceptName = ?",
        (1, "GLUCOSE_STATE")
    )
    
    for start, end in rows:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        assert start_dt > pd.to_datetime("2024-01-01 08:00:00")  # After ADMISSION
        assert end_dt < pd.to_datetime("2024-01-01 18:00:00")    # Before DEATH


def test_process_multiple_patients_async(temp_db, sample_kb):
    """Test async multi-patient processing."""
    # Insert data for 3 patients
    for pid in [1, 2, 3]:
        temp_db.execute_query(
            "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
            (pid, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
        )
    
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    # Process all patients async
    patient_stats = asyncio.run(mediator.process_all_patients_async(max_concurrent=2))
    
    assert len(patient_stats) == 3
    assert all(pid in patient_stats for pid in [1, 2, 3])
    assert all("GLUCOSE_MEASURE" in patient_stats[pid] for pid in [1, 2, 3])
    assert all("GLUCOSE_STATE" in patient_stats[pid] for pid in [1, 2, 3])
    
    # Verify no errors
    assert all("error" not in patient_stats[pid] for pid in [1, 2, 3])


def test_run_full_pipeline_with_subset(temp_db, sample_kb):
    """Test full run() pipeline with patient subset."""
    # Insert data for 3 patients
    for pid in [1, 2, 3]:
        temp_db.execute_query(
            "INSERT INTO InputPatientData (PatientId, ConceptName, StartDateTime, EndDateTime, Value) VALUES (?, ?, ?, ?, ?)",
            (pid, "GLUCOSE_LAB", "2024-01-01 08:00:00", "2024-01-01 08:00:00", "120")
        )
    
    mediator = Mediator(sample_kb, temp_db)
    
    # Run with subset [1, 3]
    patient_stats = mediator.run(max_concurrent=1, patient_subset=[1, 3])
    
    assert len(patient_stats) == 2
    assert 1 in patient_stats
    assert 3 in patient_stats
    assert 2 not in patient_stats  # Not processed
    
    # Verify DB contains only patients 1 and 3
    rows = temp_db.fetch_records(
        "SELECT DISTINCT PatientId FROM OutputPatientData ORDER BY PatientId",
        ()
    )
    assert [r[0] for r in rows] == [1, 3]


# -----------------------------
# Tests: Error Handling
# -----------------------------

def test_process_patient_handles_errors(temp_db, sample_kb):
    """Test graceful error handling for patient processing."""
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    # Process nonexistent patient (should not crash)
    stats = mediator._process_patient_sync(999)
    
    # No error should be returned (just empty stats or error key)
    assert isinstance(stats, dict)
    # Should have minimal stats (no crashes)


def test_empty_input_returns_empty_output(temp_db, sample_kb):
    """Test that empty input data returns empty output."""
    mediator = Mediator(sample_kb, temp_db)
    mediator.build_repository()
    
    # Process patient with no data
    stats = mediator._process_patient_sync(1)
    
    # Should return empty stats (no TAKs produced output)
    assert all(v == 0 or v == {} for v in stats.values())
    
    # Verify no DB writes
    rows = temp_db.fetch_records("SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?", (1,))
    assert rows[0][0] == 0

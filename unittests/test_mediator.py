"""
Integration tests for Mediator orchestration logic (using production KB + DB).

Tests cover:
1. TAK repository building (real knowledge base validation)
2. Patient data extraction (real DB queries)
3. Global clippers (real InputPatientData clippers)
4. Patient subsetting (process first 10 patients)
5. Full caching (verify raw-concepts cached correctly)
6. Output writing (real DB insertion with deduplication)
7. Multi-patient async processing (small subset)
8. Error handling (graceful handling of missing patients)

NOTE: These are integration tests that use the production database and knowledge base.
They process a small subset of patients (default: first 10) to keep tests fast.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio

from core.mediator import Mediator
from backend.dataaccess import DataAccess
from backend.config import DB_PATH
from core.config import TAK_FOLDER


def make_ts(hhmm: str, day: int = 0) -> datetime:
    """Build timestamp: 2024-01-01 + day offset + HH:MM."""
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def prod_kb() -> Path:
    """Use production knowledge base (no temp creation)."""
    kb_path = Path(TAK_FOLDER)
    if not kb_path.exists():
        pytest.skip(f"Production KB not found: {kb_path}")
    return kb_path


@pytest.fixture
def prod_db() -> DataAccess:
    """
    Use production DB (no temp creation).
    DB creation is already tested in test_dataaccess.py.
    Tests will use the real backend/data/mediator.db.
    """
    db_path = Path(DB_PATH)
    if not db_path.exists():
        pytest.skip(f"Production DB not found: {db_path}. Run: python backend/dataaccess.py --create_db --load_csv <file>")
    return DataAccess(db_path=str(db_path))


@pytest.fixture
def small_patient_subset(prod_db) -> list:
    """Get first 10 patients from production DB for testing."""
    query = "SELECT DISTINCT PatientId FROM InputPatientData ORDER BY PatientId LIMIT 10;"
    rows = prod_db.fetch_records(query, ())
    patient_ids = [int(r[0]) for r in rows]
    if not patient_ids:
        pytest.skip("No patients found in InputPatientData. Load data first.")
    return patient_ids


# -----------------------------
# Tests: Repository Building
# -----------------------------

def test_build_repository_success(prod_db, prod_kb):
    """Test successful TAK repository build using production KB."""
    mediator = Mediator(prod_kb, prod_db)
    repo = mediator.build_repository()
    
    # Verify repository contains TAKs
    assert len(repo.taks) > 0, "Repository should contain at least one TAK"
    
    # Check TAK families are populated
    assert len(mediator.raw_concepts) > 0, "Should have raw-concepts"
    
    # Check global clippers loaded
    assert len(mediator.global_clippers) > 0, "Should have global clippers"
    
    print(f"\n✅ Repository built: {len(repo.taks)} TAKs")
    print(f"   - Raw concepts: {len(mediator.raw_concepts)}")
    print(f"   - Events:       {len(mediator.events)}")
    print(f"   - States:       {len(mediator.states)}")
    print(f"   - Trends:       {len(mediator.trends)}")
    print(f"   - Contexts:     {len(mediator.contexts)}")
    print(f"   - Patterns:     {len(mediator.patterns)}")


def test_load_global_clippers(prod_db, prod_kb):
    """Test global clippers loaded from production global_clippers.json."""
    mediator = Mediator(prod_kb, prod_db)
    
    assert len(mediator.global_clippers) > 0, "Should have global clippers"
    
    # Check structure (should be {name: 'START'|'END'})
    for name, how in mediator.global_clippers.items():
        assert how in ("START", "END"), f"Invalid clipper 'how' value: {how}"
    
    print(f"\n✅ Global clippers: {list(mediator.global_clippers.keys())}")


# -----------------------------
# Tests: Patient ID Extraction
# -----------------------------

def test_get_patient_ids_all(prod_db, prod_kb):
    """Test patient ID extraction (all patients in DB)."""
    mediator = Mediator(prod_kb, prod_db)
    patient_ids = mediator.get_patient_ids()
    
    assert len(patient_ids) > 0, "Should have at least one patient"
    assert all(isinstance(pid, int) for pid in patient_ids), "All patient IDs should be integers"
    
    print(f"\n✅ Total patients in DB: {len(patient_ids)}")


def test_get_patient_ids_subset(prod_db, prod_kb, small_patient_subset):
    """Test patient ID extraction with subset filter."""
    mediator = Mediator(prod_kb, prod_db)
    
    # Request first 5 patients + a nonexistent one
    requested = small_patient_subset[:5] + [999999]
    patient_ids = mediator.get_patient_ids(patient_subset=requested)
    
    # Should return only existing patients
    assert len(patient_ids) == 5, "Should return only existing patients"
    assert 999999 not in patient_ids, "Nonexistent patient should be filtered out"
    
    print(f"\n✅ Patient subset: {patient_ids}")


# -----------------------------
# Tests: Input Data Extraction
# -----------------------------

def test_get_input_for_raw_concept(prod_db, prod_kb, small_patient_subset):
    """Test input extraction for RawConcept TAK (using production data)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.raw_concepts:
        pytest.skip("No raw-concepts in KB")
    
    # Use first raw-concept and first patient
    raw_concept = mediator.raw_concepts[0]
    patient_id = small_patient_subset[0]
    
    df = mediator.get_input_for_raw_concept(patient_id, raw_concept)
    
    # Verify schema
    assert all(col in df.columns for col in ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
    
    print(f"\n✅ Extracted {len(df)} rows for {raw_concept.name}, patient {patient_id}")


def test_get_input_for_tak_caching(prod_db, prod_kb, small_patient_subset):
    """Test that get_input_for_tak uses cache (no redundant queries)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.raw_concepts:
        pytest.skip("No raw-concepts in KB")
    
    raw_concept = mediator.raw_concepts[0]
    patient_id = small_patient_subset[0]
    tak_outputs = {}
    
    # First call: should query DB
    df1 = mediator.get_input_for_tak(patient_id, raw_concept, tak_outputs)
    assert raw_concept.name in tak_outputs, "Should be cached after first call"
    
    # Second call: should use cache
    df2 = mediator.get_input_for_tak(patient_id, raw_concept, tak_outputs)
    assert df1.equals(df2), "Cached DataFrame should match original"
    
    print(f"\n✅ Caching works: {raw_concept.name} cached {len(df1)} rows")


# -----------------------------
# Tests: Output Writing
# -----------------------------

def test_write_output(prod_db, prod_kb, small_patient_subset):
    """Test output writing to OutputPatientData."""
    mediator = Mediator(prod_kb, prod_db)
    patient_id = small_patient_subset[0]
    
    # Create test output (use a unique test concept name to avoid polluting production data)
    df_output = pd.DataFrame([
        {
            "PatientId": patient_id,
            "ConceptName": "TEST_MEDIATOR_OUTPUT",
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 14:00:00",
            "Value": "TestValue",
            "AbstractionType": "test"
        }
    ])
    
    rows_written = mediator.write_output(df_output)
    assert rows_written == 1
    
    # Verify DB write
    rows = prod_db.fetch_records(
        "SELECT ConceptName, Value FROM OutputPatientData WHERE PatientId = ? AND ConceptName = ?",
        (patient_id, "TEST_MEDIATOR_OUTPUT")
    )
    assert len(rows) == 1
    assert rows[0] == ("TEST_MEDIATOR_OUTPUT", "TestValue")
    
    # Cleanup: delete test row
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE ConceptName = ?", ("TEST_MEDIATOR_OUTPUT",))
    
    print(f"\n✅ Write test passed (patient {patient_id})")


def test_write_output_deduplication(prod_db, prod_kb, small_patient_subset):
    """Test INSERT OR IGNORE prevents duplicate writes."""
    mediator = Mediator(prod_kb, prod_db)
    patient_id = small_patient_subset[0]
    
    df_output = pd.DataFrame([
        {
            "PatientId": patient_id,
            "ConceptName": "TEST_DEDUP",
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 14:00:00",
            "Value": "TestValue",
            "AbstractionType": "test"
        }
    ])
    
    # Write twice
    rows1 = mediator.write_output(df_output)
    rows2 = mediator.write_output(df_output)
    
    assert rows1 == 1
    assert rows2 == 0, "Duplicate should be ignored"
    
    # Cleanup
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE ConceptName = ?", ("TEST_DEDUP",))
    
    print(f"\n✅ Deduplication works (patient {patient_id})")


# -----------------------------
# Tests: Global Clippers
# -----------------------------

def test_global_clippers_query(prod_db, prod_kb, small_patient_subset):
    """Test that global clippers can be queried from InputPatientData."""
    mediator = Mediator(prod_kb, prod_db)
    patient_id = small_patient_subset[0]
    
    clipper_df = mediator._get_global_clipper_times(patient_id)
    
    # May be None if patient has no clipper events
    if clipper_df is not None:
        assert "ConceptName" in clipper_df.columns
        assert "StartDateTime" in clipper_df.columns
        print(f"\n✅ Found {len(clipper_df)} clippers for patient {patient_id}")
    else:
        print(f"\n⚠️  No clippers found for patient {patient_id} (this is OK)")


def test_global_clippers_trim_logic(prod_db, prod_kb):
    """Test that _apply_global_clippers correctly trims intervals."""
    mediator = Mediator(prod_kb, prod_db)
    
    # Simulate clipper data (START clipper at 08:00)
    clipper_df = pd.DataFrame([
        {"ConceptName": "ADMISSION", "StartDateTime": pd.to_datetime("2024-01-01 08:00:00")}
    ])
    
    # Simulate state interval that starts before clipper
    df_before = pd.DataFrame([
        {
            "PatientId": 1,
            "ConceptName": "TEST_STATE",
            "StartDateTime": pd.to_datetime("2024-01-01 07:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 10:00:00"),
            "Value": "Normal",
            "AbstractionType": "state"
        }
    ])
    
    df_after = mediator._apply_global_clippers(df_before, clipper_df)
    
    # StartDateTime should be trimmed to 08:00:01
    assert len(df_after) == 1
    assert df_after.iloc[0]["StartDateTime"] == pd.to_datetime("2024-01-01 08:00:01")
    assert df_after.iloc[0]["EndDateTime"] == pd.to_datetime("2024-01-01 10:00:00")
    
    print("\n✅ Global clipper trimming works correctly")


# -----------------------------
# Tests: Patient Processing
# -----------------------------

def test_process_patient_single(prod_db, prod_kb, small_patient_subset):
    """Test single patient processing (read from production DB, write to OutputPatientData)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_id = small_patient_subset[0]
    
    # Clear output for this patient before test
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    
    # Process patient
    stats = mediator._process_patient_sync(patient_id)
    
    # Check that stats were returned (some TAKs should have produced output)
    assert isinstance(stats, dict)
    assert "error" not in stats, f"Patient processing failed: {stats.get('error')}"
    
    # Verify DB writes (should have at least some output)
    rows = prod_db.fetch_records(
        "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
        (patient_id,)
    )
    total_rows = rows[0][0]
    
    print(f"\n✅ Patient {patient_id} processed:")
    print(f"   - Stats: {stats}")
    print(f"   - Total output rows: {total_rows}")
    
    # Cleanup: delete test output
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))


def test_process_multiple_patients_async(prod_db, prod_kb, small_patient_subset):
    """Test async multi-patient processing (small subset)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    # Use first 3 patients only
    test_patients = small_patient_subset[:3]
    
    # Clear output for test patients
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Process async
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=test_patients
        )
    )
    
    assert len(patient_stats) == len(test_patients)
    assert all(pid in patient_stats for pid in test_patients)
    
    # Verify no errors
    errors = [pid for pid, stats in patient_stats.items() if "error" in stats]
    assert len(errors) == 0, f"Errors for patients: {errors}"
    
    print(f"\n✅ Processed {len(test_patients)} patients async:")
    for pid, stats in patient_stats.items():
        total = sum(v for k, v in stats.items() if k != "error" and isinstance(v, int))
        print(f"   - Patient {pid}: {total} rows written")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


def test_run_full_pipeline_with_subset(prod_db, prod_kb, small_patient_subset):
    """Test full run() pipeline with patient subset (integration test)."""
    mediator = Mediator(prod_kb, prod_db)
    
    # Use first 5 patients
    test_patients = small_patient_subset[:5]
    
    # Clear output for test patients
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Run pipeline
    patient_stats = mediator.run(max_concurrent=4, patient_subset=test_patients)
    
    assert len(patient_stats) == len(test_patients)
    
    # Verify DB writes
    for pid in test_patients:
        rows = prod_db.fetch_records(
            "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
            (pid,)
        )
        row_count = rows[0][0]
        print(f"   - Patient {pid}: {row_count} output rows")
    
    print(f"\n✅ Full pipeline run successful ({len(test_patients)} patients)")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


# -----------------------------
# Tests: Error Handling
# -----------------------------

def test_process_patient_handles_errors(prod_db, prod_kb):
    """Test graceful error handling for nonexistent patient."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    # Process nonexistent patient (should not crash)
    stats = mediator._process_patient_sync(999999)
    
    # Should return empty stats (no crashes)
    assert isinstance(stats, dict)
    print(f"\n✅ Nonexistent patient handled gracefully: {stats}")


def test_empty_subset_returns_empty(prod_db, prod_kb):
    """Test that empty patient subset returns empty results."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=[]
        )
    )
    
    assert len(patient_stats) == 0
    print("\n✅ Empty subset handled correctly")


# -----------------------------
# Performance Benchmark (Optional)
# -----------------------------

def test_benchmark_10_patients(prod_db, prod_kb, small_patient_subset):
    """Benchmark: process 10 patients and report timing."""
    import time
    
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    test_patients = small_patient_subset[:10]
    
    # Clear output
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Time processing
    start = time.time()
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=test_patients
        )
    )
    elapsed = time.time() - start
    
    total_rows = sum(
        sum(v for k, v in stats.items() if k != "error" and isinstance(v, int))
        for stats in patient_stats.values()
    )
    
    print(f"\n✅ Benchmark (10 patients):")
    print(f"   - Total time: {elapsed:.2f}s")
    print(f"   - Rows written: {total_rows}")
    print(f"   - Throughput: {total_rows / elapsed:.1f} rows/sec")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


# -----------------------------
# Tests: Pattern Output (NEW)
# -----------------------------

def test_pattern_output_split(prod_db, prod_kb, small_patient_subset):
    """Test that Pattern outputs are correctly split into OutputPatientData + PatientQAScores."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    patient_id = small_patient_subset[0]
    
    # Clear output for test patient
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))
    
    # Process patient
    stats = mediator._process_patient_sync(patient_id)
    
    # Check that at least one pattern was processed
    pattern_stats = {k: v for k, v in stats.items() if k in [p.name for p in mediator.patterns]}
    
    if not pattern_stats:
        pytest.skip(f"No pattern data for patient {patient_id}")
    
    # Verify OutputPatientData writes (only rows with valid timestamps)
    output_rows = prod_db.fetch_records(
        """SELECT ConceptName, StartDateTime, EndDateTime, Value 
           FROM OutputPatientData 
           WHERE PatientId = ? AND AbstractionType = 'local-pattern'""",
        (patient_id,)
    )
    
    # Verify PatientQAScores writes (all rows including "False" patterns)
    qa_rows = prod_db.fetch_records(
        """SELECT PatternName, ComplianceType, ComplianceScore 
           FROM PatientQAScores 
           WHERE PatientId = ?""",
        (patient_id,)
    )
    
    print(f"\n✅ Pattern output split test (patient {patient_id}):")
    print(f"   - OutputPatientData rows: {len(output_rows)}")
    print(f"   - PatientQAScores rows: {len(qa_rows)}")
    
    if output_rows:
        print(f"   - Example output: {output_rows[0]}")
    if qa_rows:
        print(f"   - Example QA score: {qa_rows[0]}")
    
    # Verify structure
    for row in qa_rows:
        pattern_name, compliance_type, score = row
        assert compliance_type in ("TimeConstraint", "ValueConstraint"), f"Invalid compliance type: {compliance_type}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    # Cleanup
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))


def test_pattern_false_instances_only_in_qa_scores(prod_db, prod_kb, small_patient_subset):
    """Test that 'False' patterns (no match) only appear in PatientQAScores, not OutputPatientData."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    patient_id = small_patient_subset[0]
    
    # Clear output
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))
    
    # Process patient
    mediator._process_patient_sync(patient_id)
    
    # Check OutputPatientData: should have NO rows with NaT timestamps
    output_rows = prod_db.fetch_records(
        """SELECT ConceptName, StartDateTime 
           FROM OutputPatientData 
           WHERE PatientId = ? AND AbstractionType = 'local-pattern'""",
        (patient_id,)
    )
    
    for concept_name, start_dt in output_rows:
        assert start_dt is not None, f"Pattern {concept_name} has NULL StartDateTime in OutputPatientData (should be filtered out)"
    
    # Check PatientQAScores: may have rows with NULL StartDateTime (for "False" patterns)
    qa_rows = prod_db.fetch_records(
        """SELECT PatternName, StartDateTime, ComplianceScore 
           FROM PatientQAScores 
           WHERE PatientId = ? AND StartDateTime IS NULL""",
        (patient_id,)
    )
    
    if qa_rows:
        print(f"\n✅ Found {len(qa_rows)} 'False' pattern instances (NULL timestamps in QA scores):")
        for pattern_name, start_dt, score in qa_rows[:3]:  # Show first 3
            print(f"   - {pattern_name}: score={score}")
    
    # Cleanup
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))


def test_pattern_compliance_scores_unpivoted(prod_db, prod_kb, small_patient_subset):
    """Test that Pattern compliance scores are correctly unpivoted (2 rows per pattern instance)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    patient_id = small_patient_subset[0]
    
    # Clear output
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))
    
    # Process patient
    mediator._process_patient_sync(patient_id)
    
    # Get QA scores grouped by (PatternName, StartDateTime)
    qa_rows = prod_db.fetch_records(
        """SELECT PatternName, StartDateTime, ComplianceType, ComplianceScore 
           FROM PatientQAScores 
           WHERE PatientId = ?
           ORDER BY PatternName, StartDateTime, ComplianceType""",
        (patient_id,)
    )
    
    if not qa_rows:
        pytest.skip(f"No QA scores for patient {patient_id}")
    
    # Group by (PatternName, StartDateTime) and check that each group has 1-2 rows
    from itertools import groupby
    
    grouped = {}
    for key, group in groupby(qa_rows, key=lambda r: (r[0], r[1])):
        grouped[key] = list(group)
    
    print(f"\n✅ QA score unpivoting test (patient {patient_id}):")
    print(f"   - Total unique pattern instances: {len(grouped)}")
    
    for (pattern_name, start_dt), rows in list(grouped.items())[:3]:  # Show first 3
        compliance_types = [r[2] for r in rows]
        scores = [r[3] for r in rows]
        print(f"   - {pattern_name} @ {start_dt}: {len(rows)} compliance scores")
        print(f"     Types: {compliance_types}, Scores: {scores}")
        
        # Each instance should have 1-2 rows (TimeConstraint and/or ValueConstraint)
        assert len(rows) <= 2, f"Too many rows for pattern instance: {len(rows)}"
        
        # Check that compliance types are unique
        assert len(set(compliance_types)) == len(compliance_types), "Duplicate compliance types found"
    
    # Cleanup
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))


def test_pattern_deduplication(prod_db, prod_kb, small_patient_subset):
    """Test that duplicate pattern outputs are handled by INSERT OR IGNORE."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    patient_id = small_patient_subset[0]
    
    # Clear output
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))
    
    # Process patient TWICE
    stats1 = mediator._process_patient_sync(patient_id)
    stats2 = mediator._process_patient_sync(patient_id)
    
    # Second run should have 0 writes (all duplicates ignored)
    pattern_stats = {k: v for k, v in stats2.items() if k in [p.name for p in mediator.patterns]}
    
    if pattern_stats:
        for pattern_name, stat_dict in pattern_stats.items():
            if isinstance(stat_dict, dict):
                output_rows = stat_dict.get("output_rows", 0)
                qa_rows = stat_dict.get("qa_scores", 0)
                print(f"   - {pattern_name}: {output_rows} output rows, {qa_rows} QA rows (second run)")
                
                # Both should be 0 (duplicates ignored)
                assert output_rows == 0, f"Duplicate output rows written for {pattern_name}"
                assert qa_rows == 0, f"Duplicate QA scores written for {pattern_name}"
    
    print(f"\n✅ Pattern deduplication works (patient {patient_id})")
    
    # Cleanup
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (patient_id,))


# -----------------------------
# Tests: Patient Processing
# -----------------------------

def test_process_patient_single(prod_db, prod_kb, small_patient_subset):
    """Test single patient processing (read from production DB, write to OutputPatientData)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_id = small_patient_subset[0]
    
    # Clear output for this patient before test
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    
    # Process patient
    stats = mediator._process_patient_sync(patient_id)
    
    # Check that stats were returned (some TAKs should have produced output)
    assert isinstance(stats, dict)
    assert "error" not in stats, f"Patient processing failed: {stats.get('error')}"
    
    # Verify DB writes (should have at least some output)
    rows = prod_db.fetch_records(
        "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
        (patient_id,)
    )
    total_rows = rows[0][0]
    
    print(f"\n✅ Patient {patient_id} processed:")
    print(f"   - Stats: {stats}")
    print(f"   - Total output rows: {total_rows}")
    
    # Cleanup: delete test output
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))


def test_process_multiple_patients_async(prod_db, prod_kb, small_patient_subset):
    """Test async multi-patient processing (small subset)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    # Use first 3 patients only
    test_patients = small_patient_subset[:3]
    
    # Clear output for test patients
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Process async
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=test_patients
        )
    )
    
    assert len(patient_stats) == len(test_patients)
    assert all(pid in patient_stats for pid in test_patients)
    
    # Verify no errors
    errors = [pid for pid, stats in patient_stats.items() if "error" in stats]
    assert len(errors) == 0, f"Errors for patients: {errors}"
    
    print(f"\n✅ Processed {len(test_patients)} patients async:")
    for pid, stats in patient_stats.items():
        total = sum(v for k, v in stats.items() if k != "error" and isinstance(v, int))
        print(f"   - Patient {pid}: {total} rows written")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


def test_run_full_pipeline_with_subset(prod_db, prod_kb, small_patient_subset):
    """Test full run() pipeline with patient subset (integration test)."""
    mediator = Mediator(prod_kb, prod_db)
    
    # Use first 5 patients
    test_patients = small_patient_subset[:5]
    
    # Clear output for test patients
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Run pipeline
    patient_stats = mediator.run(max_concurrent=4, patient_subset=test_patients)
    
    assert len(patient_stats) == len(test_patients)
    
    # Verify DB writes
    for pid in test_patients:
        rows = prod_db.fetch_records(
            "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
            (pid,)
        )
        row_count = rows[0][0]
        print(f"   - Patient {pid}: {row_count} output rows")
    
    print(f"\n✅ Full pipeline run successful ({len(test_patients)} patients)")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


# -----------------------------
# Tests: Error Handling
# -----------------------------

def test_process_patient_handles_errors(prod_db, prod_kb):
    """Test graceful error handling for nonexistent patient."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    # Process nonexistent patient (should not crash)
    stats = mediator._process_patient_sync(999999)
    
    # Should return empty stats (no crashes)
    assert isinstance(stats, dict)
    print(f"\n✅ Nonexistent patient handled gracefully: {stats}")


def test_empty_subset_returns_empty(prod_db, prod_kb):
    """Test that empty patient subset returns empty results."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=[]
        )
    )
    
    assert len(patient_stats) == 0
    print("\n✅ Empty subset handled correctly")


# -----------------------------
# Performance Benchmark (Optional)
# -----------------------------

def test_benchmark_10_patients(prod_db, prod_kb, small_patient_subset):
    """Benchmark: process 10 patients and report timing."""
    import time
    
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    test_patients = small_patient_subset[:10]
    
    # Clear output
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    # Time processing
    start = time.time()
    patient_stats = asyncio.run(
        mediator.process_all_patients_async(
            max_concurrent=4,
            patient_subset=test_patients
        )
    )
    elapsed = time.time() - start
    
    total_rows = sum(
        sum(v for k, v in stats.items() if k != "error" and isinstance(v, int))
        for stats in patient_stats.values()
    )
    
    print(f"\n✅ Benchmark (10 patients):")
    print(f"   - Total time: {elapsed:.2f}s")
    print(f"   - Rows written: {total_rows}")
    print(f"   - Throughput: {total_rows / elapsed:.1f} rows/sec")
    
    # Cleanup
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))

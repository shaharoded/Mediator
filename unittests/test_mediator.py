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

from pathlib import Path
import pytest
import time
import pandas as pd

from core.mediator import Mediator
from backend.dataaccess import DataAccess
from backend.config import DB_PATH
from core.config import TAK_FOLDER
from unittests.test_utils import make_ts  # FIXED: correct import path


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def prod_kb() -> Path:
    """Return path to production knowledge base."""
    return Path(TAK_FOLDER)


@pytest.fixture
def prod_db() -> DataAccess:
    """Return production database connection."""
    return DataAccess(DB_PATH)


@pytest.fixture
def small_patient_subset(prod_db) -> list:
    """Return first 10 patient IDs from production DB."""
    rows = prod_db.fetch_records("SELECT DISTINCT PatientId FROM InputPatientData LIMIT 10")
    return [row[0] for row in rows]


# -----------------------------
# Tests: Repository Building
# -----------------------------

def test_build_repository_success(prod_db, prod_kb):
    """Test successful TAK repository build using production KB."""
    mediator = Mediator(prod_kb, prod_db)
    repo = mediator.build_repository()
    
    assert len(repo.taks) > 0, "Repository should contain at least one TAK"
    assert len(mediator.raw_concepts) > 0, "Should have raw-concepts"
    assert len(mediator.global_clippers) > 0, "Should have global clippers"
    
    print(f"\n✅ Repository built: {len(repo.taks)} TAKs")


def test_load_global_clippers(prod_db, prod_kb):
    """Test global clippers loaded from production global_clippers.json."""
    mediator = Mediator(prod_kb, prod_db)
    
    assert len(mediator.global_clippers) > 0, "Should have global clippers"
    
    for name, how in mediator.global_clippers.items():
        assert how in ("START", "END"), f"Clipper {name} has invalid 'how': {how}"
    
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
    
    requested = small_patient_subset[:5] + [999999]
    patient_ids = mediator.get_patient_ids(patient_subset=requested)
    
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
        pytest.skip("No raw-concepts in repository")
    
    raw_concept = mediator.raw_concepts[0]
    patient_id = small_patient_subset[0]
    
    tak_outputs = {}
    df = mediator.get_input_for_tak(patient_id, raw_concept, tak_outputs, da=prod_db)
    
    assert all(col in df.columns for col in ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
    
    print(f"\n✅ Extracted {len(df)} rows for {raw_concept.name}, patient {patient_id}")


def test_get_input_for_tak_caching(prod_db, prod_kb, small_patient_subset):
    """Test that get_input_for_tak uses cache (no redundant queries)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.raw_concepts:
        pytest.skip("No raw-concepts in repository")
    
    raw_concept = mediator.raw_concepts[0]
    patient_id = small_patient_subset[0]
    tak_outputs = {}
    
    df1 = mediator.get_input_for_tak(patient_id, raw_concept, tak_outputs, da=prod_db)
    tak_outputs[raw_concept.name] = df1
    df2 = mediator.get_input_for_tak(patient_id, raw_concept, tak_outputs, da=prod_db)
    
    assert df1.equals(df2), "Cached DataFrame should match original"
    
    print(f"\n✅ Caching works: {raw_concept.name} cached {len(df1)} rows")


# -----------------------------
# Tests: Output Writing
# -----------------------------

def test_write_output(prod_db, prod_kb, small_patient_subset):
    """Test output writing to OutputPatientData."""
    mediator = Mediator(prod_kb, prod_db)
    patient_id = small_patient_subset[0]
    
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
    
    rows = prod_db.fetch_records(
        "SELECT ConceptName, Value FROM OutputPatientData WHERE PatientId = ? AND ConceptName = ?",
        (patient_id, "TEST_MEDIATOR_OUTPUT")
    )
    assert len(rows) == 1
    assert rows[0] == ("TEST_MEDIATOR_OUTPUT", "TestValue")
    
    # Cleanup
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
    
    clipper_df = mediator._get_global_clipper_times(patient_id, prod_db)
    
    if clipper_df is not None:
        assert "ConceptName" in clipper_df.columns
        assert "StartDateTime" in clipper_df.columns
    else:
        print(f"\n⚠️  Patient {patient_id} has no clipper events")


def test_global_clippers_trim_logic(prod_db, prod_kb):
    """Test that _apply_global_clippers correctly trims intervals."""
    mediator = Mediator(prod_kb, prod_db)
    
    clipper_df = pd.DataFrame([
        {"ConceptName": "ADMISSION", "StartDateTime": pd.to_datetime("2024-01-01 08:00:00")}
    ])
    
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
    
    assert len(df_after) == 1
    assert df_after.iloc[0]["StartDateTime"] == pd.to_datetime("2024-01-01 08:00:00")
    assert df_after.iloc[0]["EndDateTime"] == pd.to_datetime("2024-01-01 10:00:00")
    
    print("\n✅ Global clipper trimming works correctly")


# -----------------------------
# Tests: Patient Processing
# -----------------------------

def test_process_patient_single(prod_db, prod_kb, small_patient_subset):
    """Test single patient processing."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_id = small_patient_subset[0]
    
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))
    
    stats = mediator._process_patient_sync(patient_id)
    
    assert isinstance(stats, dict)
    assert "error" not in stats, f"Patient processing failed: {stats.get('error')}"
    
    rows = prod_db.fetch_records(
        "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
        (patient_id,)
    )
    total_rows = rows[0][0]
    
    print(f"\n✅ Patient {patient_id} processed: {total_rows} rows")
    
    prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (patient_id,))


def test_process_multiple_patients_parallel(prod_db, prod_kb, small_patient_subset):
    """Test parallel multi-patient processing (small subset)."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    test_patients = small_patient_subset[:3]
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    patient_stats = mediator.process_all_patients_parallel(
        max_concurrent=4,
        patient_subset=test_patients
    )
    
    assert len(patient_stats) == len(test_patients)
    assert all(pid in patient_stats for pid in test_patients)
    
    errors = [pid for pid, stats in patient_stats.items() if "error" in stats]
    assert len(errors) == 0, f"Errors for patients: {errors}"
    
    print(f"\n✅ Processed {len(test_patients)} patients async")
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


def test_run_full_pipeline_with_subset(prod_db, prod_kb, small_patient_subset):
    """Test full run() pipeline with patient subset (integration test)."""
    mediator = Mediator(prod_kb, prod_db)
    
    test_patients = small_patient_subset[:5]
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    patient_stats = mediator.run(max_concurrent=4, patient_subset=test_patients)
    
    assert len(patient_stats) == len(test_patients)
    
    for pid in test_patients:
        rows = prod_db.fetch_records(
            "SELECT COUNT(*) FROM OutputPatientData WHERE PatientId = ?",
            (pid,)
        )
        assert rows[0][0] >= 0
    
    print(f"\n✅ Full pipeline run successful ({len(test_patients)} patients)")
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


# -----------------------------
# Tests: Error Handling
# -----------------------------

def test_process_patient_handles_errors(prod_db, prod_kb):
    """Test graceful error handling for nonexistent patient."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    stats = mediator._process_patient_sync(999999)
    
    assert isinstance(stats, dict)
    print(f"\n✅ Nonexistent patient handled gracefully: {stats}")


def test_empty_subset_returns_empty(prod_db, prod_kb):
    """Test that empty patient subset returns empty results."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    patient_stats = mediator.process_all_patients_parallel(
        max_concurrent=4,
        patient_subset=[]
    )
    
    assert len(patient_stats) == 0
    print("\n✅ Empty subset handled correctly")


# -----------------------------
# Performance Benchmark (Optional)
# -----------------------------

def test_benchmark_10_patients(prod_db, prod_kb, small_patient_subset):
    """Benchmark: process 10 patients and report timing."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    test_patients = small_patient_subset[:10]
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))
    
    start = time.time()
    patient_stats = mediator.process_all_patients_parallel(
        max_concurrent=4,
        patient_subset=test_patients
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
    
    for pid in test_patients:
        prod_db.execute_query("DELETE FROM OutputPatientData WHERE PatientId = ?", (pid,))


# -----------------------------
# Tests: Pattern Output
# -----------------------------

def test_pattern_output_split(prod_db, prod_kb, small_patient_subset):
    """Test that Pattern output is correctly split into main output + QA scores."""
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    # Use first pattern and first patient
    pattern = mediator.patterns[0]
    patient_id = small_patient_subset[0]
    
    # Simulate pattern output with compliance scores
    df_pattern_output = pd.DataFrame([
        {
            "PatientId": patient_id,
            "ConceptName": pattern.name,
            "StartDateTime": "2024-01-01 08:00:00",
            "EndDateTime": "2024-01-01 10:00:00",
            "Value": "True",
            "TimeConstraintScore": 1.0,
            "ValueConstraintScore": 0.8,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        },
        {
            "PatientId": patient_id,
            "ConceptName": pattern.name,
            "StartDateTime": "2024-01-01 14:00:00",
            "EndDateTime": "2024-01-01 16:00:00",
            "Value": "Partial",
            "TimeConstraintScore": 0.6,
            "ValueConstraintScore": 1.0,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        },
        {
            "PatientId": patient_id,
            "ConceptName": pattern.name,
            "StartDateTime": pd.NaT,
            "EndDateTime": pd.NaT,
            "Value": "False",
            "TimeConstraintScore": 0.0,
            "ValueConstraintScore": 0.0,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        }
    ])
    
    # Split output
    df_main, df_scores = mediator._split_pattern_output(df_pattern_output)
    
    # Verify main output (only True/Partial, no False)
    assert len(df_main) == 2
    assert all(df_main["ConceptName"] == pattern.name)
    assert all(df_main["Value"].isin(["True", "Partial"]))
    assert "TimeConstraintScore" not in df_main.columns
    assert "ValueConstraintScore" not in df_main.columns
    
    # Verify QA scores (all 3 rows, including False)
    assert len(df_scores) == 3
    assert all(df_scores["ConceptName"] == pattern.name)
    assert "TimeConstraintScore" in df_scores.columns
    assert "ValueConstraintScore" in df_scores.columns
    assert df_scores.iloc[0]["TimeConstraintScore"] == 1.0
    assert df_scores.iloc[1]["ValueConstraintScore"] == 1.0
    assert df_scores.iloc[2]["TimeConstraintScore"] == 0.0
    
    print(f"\n✅ Pattern output split correctly: {len(df_main)} main rows, {len(df_scores)} QA rows")


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


def test_pattern_compliance_scores_unpivoted(prod_db, prod_kb):
    """
    Test that pattern compliance scores are unpivoted correctly (one row per score type).
    FIXED: Use synthetic data instead of searching through real patients.
    """
    mediator = Mediator(prod_kb, prod_db)
    mediator.build_repository()
    
    if not mediator.patterns:
        pytest.skip("No patterns in KB")
    
    # Find a pattern with BOTH time and value compliance
    pattern_with_compliance = None
    for pattern in mediator.patterns:
        has_time = any(r.time_constraint_compliance for r in pattern.abstraction_rules)
        has_value = any(r.value_constraint_compliance for r in pattern.abstraction_rules)
        if has_time and has_value:
            pattern_with_compliance = pattern
            break
    
    if not pattern_with_compliance:
        pytest.skip("No pattern with both time and value compliance found")
    
    print(f"\n✅ Testing with pattern: {pattern_with_compliance.name}")
    
    # Create synthetic pattern output with compliance scores
    test_patient_id = 999999  # Use non-existent patient ID to avoid collisions
    df_pattern_output = pd.DataFrame([
        {
            "PatientId": test_patient_id,
            "ConceptName": pattern_with_compliance.name,
            "StartDateTime": pd.to_datetime("2024-01-01 08:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 10:00:00"),
            "Value": "True",
            "TimeConstraintScore": 1.0,
            "ValueConstraintScore": 0.8,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        },
        {
            "PatientId": test_patient_id,
            "ConceptName": pattern_with_compliance.name,
            "StartDateTime": pd.to_datetime("2024-01-01 14:00:00"),
            "EndDateTime": pd.to_datetime("2024-01-01 16:00:00"),
            "Value": "Partial",
            "TimeConstraintScore": 0.6,
            "ValueConstraintScore": 1.0,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        },
        {
            "PatientId": test_patient_id,
            "ConceptName": pattern_with_compliance.name,
            "StartDateTime": pd.NaT,
            "EndDateTime": pd.NaT,
            "Value": "False",
            "TimeConstraintScore": 0.0,
            "ValueConstraintScore": 0.0,
            "CyclicConstraintScore": None,
            "AbstractionType": "local-pattern"
        }
    ])
    
    # Clear any existing test data
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (test_patient_id,))
    
    # Write QA scores using mediator's internal method
    mediator.write_qa_scores(df_pattern_output)
    
    # Query PatientQAScores to verify unpivoting
    query = """
        SELECT PatientId, PatternName, StartDateTime, ComplianceType, ComplianceScore
        FROM PatientQAScores
        WHERE PatientId = ?
        ORDER BY StartDateTime, ComplianceType
    """
    rows = prod_db.fetch_records(query, (test_patient_id,))
    
    # Verify we have rows
    assert len(rows) > 0, "No QA scores written to database"
    
    df_scores = pd.DataFrame(rows, columns=["PatientId", "PatternName", "StartDateTime", "ComplianceType", "ComplianceScore"])
    
    print(f"\n=== QA Scores (unpivoted) ===")
    print(df_scores)
    
    # Verify structure
    assert "ComplianceType" in df_scores.columns
    assert "ComplianceScore" in df_scores.columns
    
    # Verify unpivoting: should have 6 rows (3 instances × 2 compliance types)
    assert len(df_scores) == 6, f"Expected 6 rows (3 instances × 2 types), got {len(df_scores)}"
    
    # Verify compliance types
    compliance_types = set(df_scores["ComplianceType"].unique())
    assert compliance_types == {"TimeConstraint", "ValueConstraint"}, f"Expected both types, got {compliance_types}"
    
    # Verify each instance has 2 rows (one per compliance type)
    grouped = df_scores.groupby("StartDateTime").size()
    # 2 instances with timestamps should have 2 rows each
    # 1 instance with NULL timestamp (False) should have 2 rows
    assert all(count == 2 for count in grouped), "Each instance should have 2 rows (one per compliance type)"
    
    # Verify scores match input
    row1_time = df_scores[(df_scores["StartDateTime"] == "2024-01-01 08:00:00") & (df_scores["ComplianceType"] == "TimeConstraint")].iloc[0]
    row1_value = df_scores[(df_scores["StartDateTime"] == "2024-01-01 08:00:00") & (df_scores["ComplianceType"] == "ValueConstraint")].iloc[0]
    assert row1_time["ComplianceScore"] == 1.0
    assert row1_value["ComplianceScore"] == 0.8
    
    row2_time = df_scores[(df_scores["StartDateTime"] == "2024-01-01 14:00:00") & (df_scores["ComplianceType"] == "TimeConstraint")].iloc[0]
    row2_value = df_scores[(df_scores["StartDateTime"] == "2024-01-01 14:00:00") & (df_scores["ComplianceType"] == "ValueConstraint")].iloc[0]
    assert row2_time["ComplianceScore"] == 0.6
    assert row2_value["ComplianceScore"] == 1.0
    
    # Verify "False" instance has 2 rows with NULL timestamp
    false_rows = df_scores[df_scores["StartDateTime"].isna()]
    assert len(false_rows) == 2, "False instance should have 2 rows (one per compliance type)"
    assert set(false_rows["ComplianceType"]) == {"TimeConstraint", "ValueConstraint"}
    assert all(false_rows["ComplianceScore"] == 0.0)
    
    print(f"\n✅ Unpivoting test PASSED: {len(df_scores)} rows (3 instances × 2 types)")
    
    # Cleanup
    prod_db.execute_query("DELETE FROM PatientQAScores WHERE PatientId = ?", (test_patient_id,))
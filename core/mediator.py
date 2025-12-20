"""
TO-DO:
 - Currently all patterns have the False rows insersion, but some patterns function like events (infection) and don't need it. Let's have a block for that.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
import logging
import traceback
import os
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .tak.tak import TAK
from .tak.repository import TAKRepository, set_tak_repository
from .tak.raw_concept import RawConcept, ParameterizedRawConcept
from .tak.event import Event
from .tak.state import State
from .tak.trend import Trend
from .tak.context import Context
from .tak.pattern import Pattern, LocalPattern, GlobalPattern
from .config import TAK_FOLDER

# Add parent directory to path for backend imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.dataaccess import DataAccess 
from backend.config import (
    DB_PATH,
    GET_DATA_BY_PATIENT_CONCEPTS_QUERY,
    INSERT_ABSTRACTED_MEASUREMENT_QUERY,
    INSERT_QA_SCORE_QUERY
)

logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None, console_level: int = logging.WARNING):
    """
    Configure logging with separate console and file handlers.
    
    Args:
        log_file: Path to log file (if None, only console logging)
        console_level: Minimum level for console output (default: WARNING)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (WARNING and above by default)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (INFO and above)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

def _process_patient_worker(
    patient_id: int,
    kb_path_str: str,
    db_path_str: str,
    global_clippers: Dict[str, str],
    write_queue
) -> Dict[str, Union[int, str]]:
    """
    Standalone worker function for processing a single patient in a separate process.
    Must be at module level for pickling by ProcessPoolExecutor.
    
    Args:
        patient_id: Patient ID to process
        kb_path_str: Path to knowledge base (as string for pickling)
        db_path_str: Path to database (as string for pickling)
        global_clippers: Global clipper configuration
        write_queue: Queue to handle concurrent writes.
    
    Returns:
        Dict with stats: {tak_name: rows_written} or {"error": error_message}
    """
    try:
        # Suppress prints in worker processes (only main process should print)
        import io
        import contextlib
        
        # Reconstruct Mediator in this process
        from pathlib import Path
        from backend.dataaccess import DataAccess
        
        kb_path = Path(kb_path_str)
        db_path = Path(db_path_str)
        
        # Suppress stdout during DataAccess init (WAL mode message)
        with contextlib.redirect_stdout(io.StringIO()):
            da = DataAccess(db_path=str(db_path))
            mediator = Mediator(knowledge_base_path=kb_path, data_access=da)
            
            # Override global_clippers (avoid re-loading from file)
            mediator.global_clippers = global_clippers
            
            # Build repository silently in this process (each process needs its own TAK instances)
            mediator.build_repository(silent=True)
        
        # Process patient (logging still works, just no stdout prints)
        result = mediator._process_patient_sync(patient_id, return_cache=False, write_queue=write_queue)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@dataclass
class WriteBatch:
    table: str  # "output" or "qa"
    rows: List[Tuple]

@dataclass
class FlushRequest:
    """
    Control message for the single-writer process.
    When received, the writer flushes pending buffers and acks via reply_queue.
    """
    reply_queue: Optional[Any] = None

def _writer_worker(db_path: str, write_queue):
    """
    Single-writer process for SQLite.

    Purpose:
    - Enforce "exactly one writer" to SQLite to avoid 'database is locked' errors
      when multiple patient workers run in parallel.

    How it works:
    - Patient worker processes NEVER write to the DB directly.
      They push WriteBatch objects into `write_queue`.
    - This worker pulls batches from the queue, buffers them in memory, and periodically
      commits them to the database in large executemany() calls.

    Shutdown:
    - Stops when it receives a sentinel (None) from the queue
      and the queue is empty.
    - Performs a final flush before exit.
    """
    da = DataAccess(db_path=db_path)
    conn = da.conn
    cursor = da.cursor

    # Optional: increase WAL autocheckpoint to avoid huge WAL files
    conn.execute("PRAGMA wal_autocheckpoint=2000;")

    with open(INSERT_ABSTRACTED_MEASUREMENT_QUERY, "r") as f:
        q_output = f.read()
    with open(INSERT_QA_SCORE_QUERY, "r") as f:
        q_qa = f.read()

    last_commit = time.time()
    commit_every_seconds = 1.0
    commit_every_batches = 50
    batch_counter = 0

    pending_output = []
    pending_qa = []

    def flush():
        """
        Flush buffered rows to SQLite.

        pending_output and pending_qa accumulate rows from multiple queue batches.
        flush() writes the accumulated rows using executemany() and commits once.

        Called:
        - When enough batches were accumulated (commit_every_batches)
        - Or enough time passed (commit_every_seconds)
        - And once at shutdown for a final commit.
        """
        nonlocal pending_output, pending_qa, batch_counter, last_commit
        if pending_output:
            cursor.executemany(q_output, pending_output)
            pending_output = []
        if pending_qa:
            cursor.executemany(q_qa, pending_qa)
            pending_qa = []
        conn.commit()
        batch_counter = 0
        last_commit = time.time()

    while True:
        item = write_queue.get()
        if item is None:
            break

        # Explicit flush command (used by tests, can also be used by callers)
        if isinstance(item, FlushRequest):
            flush()
            if item.reply_queue is not None:
                item.reply_queue.put(True)
            continue

        batch: WriteBatch = item
        if not batch.rows:
            continue

        if batch.table == "output":
            pending_output.extend(batch.rows)
        elif batch.table == "qa":
            pending_qa.extend(batch.rows)
        else:
            raise ValueError(f"Unknown table: {batch.table}")

        batch_counter += 1

        # Flush on thresholds
        now = time.time()
        if batch_counter >= commit_every_batches or (now - last_commit) >= commit_every_seconds:
            flush()

    # Final flush
    flush()
    conn.close()


class Mediator:
    """
    Orchestrates the KBTA pipeline:
    1. Load & validate TAK repository (raw-concepts → events → states → trends → contexts → patterns)
    2. For each patient, for each TAK: extract filtered input → apply TAK → write output to DB
    3. Supports process parallelism for patient-level concurrency.
    
    Design: Full caching — all TAK outputs cached in memory during patient processing (~500KB per patient).
    """
    
    def __init__(
        self, 
        knowledge_base_path: Path, 
        data_access: Optional[DataAccess] = None
    ):
        """
        Initialize Mediator with knowledge base and DB access.
        
        Args:
            knowledge_base_path: Path to knowledge-base folder (contains raw-concepts/, events/, states/, trends/, contexts/)
            data_access: DataAccess instance for DB operations
        """
        self.kb_path = Path(knowledge_base_path)
        self.da = data_access
        self.db_path = data_access.db_path  # Store DB path for per-thread connections
        self.repo: Optional[TAKRepository] = None
        
        # TAK execution order (dependencies resolved), populated in build_repository()
        # Defines the execution order for applying TAKs to patients, which TAKRepository does not track (key - value store only)
        self.raw_concepts: List[RawConcept] = []
        self.events: List[Event] = []
        self.states: List[State] = []
        self.trends: List[Trend] = []
        self.contexts: List[Context] = []
        self.patterns: List[Pattern] = []
        
        # Load global clippers (START/END events that clip all abstractions)
        self.global_clippers = self._load_global_clippers()
        
        logger.info(f"Mediator initialized with KB path: {knowledge_base_path}")
    
    def _load_global_clippers(self) -> Dict[str, str]:
        """
        Load global clippers from knowledge-base/global_clippers.json.
        Returns: {clipper_name: 'START' | 'END'}
        """
        clipper_path = self.kb_path / "global_clippers.json"
        if not clipper_path.exists():
            logger.warning(f"Global clippers file not found: {clipper_path}")
            return {}
        
        try:
            with open(clipper_path, 'r') as f:
                data = json.load(f)
            
            clippers = {}
            for c in data.get("global_clippers", []):
                name = c["name"]
                how = c["how"].upper()
                if how not in ("START", "END"):
                    raise ValueError(f"Invalid clipper 'how': {how} (must be START or END)")
                clippers[name] = how
            
            logger.info(f"Loaded {len(clippers)} global clippers: {list(clippers.keys())}")
            return clippers
            
        except Exception as e:
            logger.error(f"Failed to load global clippers: {e}")
            return {}
    
    def build_repository(self, silent: bool = False) -> TAKRepository:
        """
        Load and validate all TAKs from knowledge base in dependency order.
        Raises on parsing/validation errors.
        
        Args:
            silent: If True, suppress all output (for worker processes)
        
        Returns:
            TAKRepository instance (also stored in self.repo and set as global)
        """
        if not silent:
            print("\n" + "="*80)
            print("PHASE 1: Building TAK Repository")
            print("="*80)
        
        repo = TAKRepository()
        
        # Set global repository BEFORE parsing any TAKs
        # (TAK.validate() needs access to repo to check parent dependencies)
        set_tak_repository(repo)
        
        # Load TAKs in dependency order with progress tracking
        # Must ensure dependencies are loaded before dependents, so order is important
        phases = [
            ("Raw Concepts", self.kb_path / "raw-concepts", RawConcept, self.raw_concepts),
            ("Parameterized Raw Concepts", self.kb_path / "parameterized-raw-concepts", ParameterizedRawConcept, self.raw_concepts),
            ("Events", self.kb_path / "events", Event, self.events),
            ("States", self.kb_path / "states", State, self.states),
            ("Trends", self.kb_path / "trends", Trend, self.trends),
            ("Contexts", self.kb_path / "contexts", Context, self.contexts),
            ("Local Patterns", self.kb_path / "local patterns", LocalPattern, self.patterns),
            ("Global Patterns", self.kb_path / "global patterns", GlobalPattern, self.patterns)
        ]
        
        total_files = sum(len(list(path.glob("*.xml"))) for _, path, _, _ in phases if path.exists())
        
        # Use tqdm only if not silent
        pbar = tqdm(total=total_files, desc="Loading TAKs", unit="file", disable=silent)
        with pbar:
            for phase_name, phase_path, tak_class, storage_list in phases:
                if not phase_path.exists():
                    logger.warning(f"[{phase_name}] Folder not found: {phase_path}")
                    continue
                
                xml_files = sorted(phase_path.glob("*.xml"))
                for xml_path in xml_files:
                    try:
                        # Parse TAK (structural validation happens here)
                        tak = tak_class.parse(xml_path)
                        
                        # Register in repository (detects duplicates)
                        repo.register(tak)
                        storage_list.append(tak)
                        
                        pbar.set_postfix_str(f"{phase_name}: {tak.name}")
                        pbar.update(1)
                        
                    except Exception as e:
                        pbar.close()
                        raise RuntimeError(f"Failed to parse {xml_path.name}: {e}") from e
        
        # Run business-logic validation on all TAKs
        if not silent:
            print("\n[Validation] Running business-logic checks on TAK repository...")
        try:
            repo.finalize_repository()
        except Exception as e:
            raise RuntimeError(f"TAK repository finalization failed: {e}") from e
        
        # Summary (only if not silent)
        if not silent:
            # Count unique TAKs from repository (avoids duplicates on re-run)
            raw_concepts_count = sum(1 for t in repo.taks.values() if isinstance(t, RawConcept))
            events_count = sum(1 for t in repo.taks.values() if isinstance(t, Event))
            states_count = sum(1 for t in repo.taks.values() if isinstance(t, State))
            trends_count = sum(1 for t in repo.taks.values() if isinstance(t, Trend))
            contexts_count = sum(1 for t in repo.taks.values() if isinstance(t, Context))
            patterns_count = sum(1 for t in repo.taks.values() if isinstance(t, Pattern))
            
            print("\n" + "="*80)
            print("✅ TAK Repository Built Successfully")
            print("="*80)
            print(f"  Raw Concepts: {raw_concepts_count}")
            print(f"  Events:       {events_count}")
            print(f"  States:       {states_count}")
            print(f"  Trends:       {trends_count}")
            print(f"  Contexts:     {contexts_count}")
            print(f"  Patterns:     {patterns_count}")
            print(f"  TOTAL TAKs:   {len(repo.taks)}")
            print("="*80 + "\n")
        
        self.repo = repo
        return repo
    
    def get_patient_ids(self, patient_subset: Optional[List[int]] = None) -> List[int]:
        """
        Retrieve patient IDs from InputPatientData.
        
        Args:
            patient_subset: Optional list of patient IDs to filter. If None, returns all patients.
                           If empty list [], returns empty list (no patients).
        
        Returns:
            List of patient IDs (sorted)
        """
        # Check if patient_subset is explicitly empty (not just falsy)
        if patient_subset is not None and len(patient_subset) == 0:
            return []
        
        if patient_subset:
            # Validate that requested patients exist in DB
            placeholders = ','.join('?' * len(patient_subset))
            query = f"SELECT DISTINCT PatientId FROM InputPatientData WHERE PatientId IN ({placeholders}) ORDER BY PatientId;"
            rows = self.da.fetch_records(query, tuple(patient_subset))
            found_ids = [int(r[0]) for r in rows]
            
            # Warn about missing patients
            missing = set(patient_subset) - set(found_ids)
            if missing:
                logger.warning(f"Requested patients not found in DB: {sorted(missing)}")
            
            return found_ids
        else:
            # Return all patients
            query = "SELECT DISTINCT PatientId FROM InputPatientData ORDER BY PatientId;"
            rows = self.da.fetch_records(query, ())
            return [int(r[0]) for r in rows]
    
    def __get_input_for_raw_concept(self, patient_id: int, raw_concept: RawConcept, da: Optional[DataAccess] = None) -> pd.DataFrame:
        """
        Query InputPatientData for a RawConcept's data by its attributes.
        Thread-safe: uses provided DataAccess instance if given, otherwise uses self.da.
        
        Returns DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
        """
        # Use provided connection or default
        db = da if da is not None else self.da
        
        concept_names = [a["name"] for a in raw_concept.attributes]
        table = "InputPatientData"
        
        # Load query template and replace placeholders
        with open(GET_DATA_BY_PATIENT_CONCEPTS_QUERY, 'r') as f:
            query_template = f.read()
        
        query = query_template.replace("{table}", table)
        placeholders = ','.join('?' * len(concept_names))
        query = query.replace("{CONCEPT_PLACEHOLDERS}", placeholders)
        
        params = (patient_id, *concept_names)
        rows = db.fetch_records(query, params)
        
        if not rows:
            return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
        
        df = pd.DataFrame(rows, columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"])
        df["AbstractionType"] = "raw-concept"
        return df
    
    def get_input_for_tak(self, patient_id: int, tak: TAK, tak_outputs: Dict[str, pd.DataFrame], da: DataAccess) -> pd.DataFrame:
        """
        Resolve TAK input dependencies (READ-ONLY cache access).
        
        BASE CASE (RawConcept):
            - Check cache first (if TAK was already applied)
            - If not cached: query InputPatientData (caller will cache after TAK.apply())
        
        RECURSIVE CASE (Event/State/Trend/Context/Pattern):
            - All dependencies MUST be in cache (guaranteed by topological execution order)
            - Concatenate cached DataFrames
        
        Args:
            patient_id: Patient ID
            tak: TAK instance to prepare input for
            tak_outputs: Cache of {tak_name: DataFrame} (OUTPUT of tak.apply())
            da: DataAccess instance (thread-safe)
        
        Returns:
            DataFrame ready for tak.apply()
        """
        # BASE CASE: RawConcept
        if isinstance(tak, RawConcept) and not isinstance(tak, ParameterizedRawConcept):
            # Check cache (if TAK was already applied)
            if tak.name in tak_outputs:
                return tak_outputs[tak.name]
            
            # Not cached → query InputPatientData (caller will cache AFTER apply())
            return self.__get_input_for_raw_concept(patient_id, tak, da)
        
        # RECURSIVE CASE: All other TAKs (cache-only lookups)
        
        # Helper to get dependency from cache (with error handling)
        def get_cached_dependency(dep_name: str) -> pd.DataFrame:
            if dep_name not in tak_outputs:
                raise RuntimeError(
                    f"TAK '{tak.name}' dependency '{dep_name}' not in cache for patient {patient_id}. "
                    f"This indicates incorrect execution order."
                )
            return tak_outputs[dep_name].copy()
        
        # CASE: ParameterizedRawConcept → derived_from + parameters
        if isinstance(tak, ParameterizedRawConcept):
            dfs = [get_cached_dependency(tak.derived_from)] +\
                  [get_cached_dependency(spec["name"]) for spec in tak.parameters]
            dfs = [df for df in dfs if not df.empty]
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
            return pd.concat(dfs, ignore_index=True)
        
        # CASE: Event → derived_from is list of RawConcepts
        if isinstance(tak, Event):
            dfs = [get_cached_dependency(df_spec["name"]) for df_spec in tak.derived_from]
            # Filter out empty DataFrames before concat
            dfs = [df for df in dfs if not df.empty]
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
            return pd.concat(dfs, ignore_index=True)
        
        # CASE: State → single derived_from (RawConcept or Event)
        if isinstance(tak, State):
            return get_cached_dependency(tak.derived_from)
        
        # CASE: Trend → single derived_from (RawConcept only)
        if isinstance(tak, Trend):
            return get_cached_dependency(tak.derived_from)
        
        # CASE: Context → derived_from (RawConcepts) + clippers (any TAK type)
        if isinstance(tak, Context):
            dfs = [get_cached_dependency(spec["name"]) for spec in tak.derived_from + tak.clippers]
            dfs = [df for df in dfs if not df.empty]
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
            return pd.concat(dfs, ignore_index=True)
        
        # CASE: Pattern → derived_from + parameters
        if isinstance(tak, Pattern):
            dfs = [get_cached_dependency(spec["name"]) for spec in tak.derived_from + tak.parameters]
            dfs = [df for df in dfs if not df.empty]
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
            return pd.concat(dfs, ignore_index=True)
        
        # Fallback (should never reach here)
        raise RuntimeError(f"Unknown TAK type: {tak.family}")
    
    def _output_df_to_rows(self, df: pd.DataFrame) -> List[Tuple]:
        """
        DB writer helper:
        DataFrame -> rows for insersion to OutputPatientData table

        Args:
            df: The output df from apply method
        """
        if df.empty:
            return []
        return [
            (
                int(row["PatientId"]),
                str(row["ConceptName"]),
                str(row["StartDateTime"]),
                str(row["EndDateTime"]),
                str(row["Value"]),
                str(row["AbstractionType"]),
            )
            for _, row in df.iterrows()
        ]

    def _qa_df_to_rows(self, df_scores: pd.DataFrame) -> List[Tuple]:
        """
        DB writer helper:
        DataFrame -> rows for insersion to PatientQAScores table

        Args:
           df_scores: The output df from melting the QA scores from Pattern's apply 
        """
        if df_scores.empty:
            return []

        df_melted = df_scores.melt(
            id_vars=["PatientId", "ConceptName", "StartDateTime", "EndDateTime"],
            value_vars=["TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore"],
            var_name="ComplianceType",
            value_name="ComplianceScore",
        ).dropna(subset=["ComplianceScore"])

        if df_melted.empty:
            return []

        df_melted["ComplianceType"] = df_melted["ComplianceType"].map({
            "TimeConstraintScore": "TimeConstraint",
            "ValueConstraintScore": "ValueConstraint",
            "CyclicConstraintScore": "CyclicConstraint",
        })

        return [
            (
                int(row["PatientId"]),
                str(row["ConceptName"]),
                str(row["StartDateTime"]) if pd.notna(row["StartDateTime"]) else None,
                str(row["EndDateTime"]) if pd.notna(row["EndDateTime"]) else None,
                str(row["ComplianceType"]),
                float(row["ComplianceScore"]),
            )
            for _, row in df_melted.iterrows()
        ]
    
    def _write_rows(self, table: str, rows: List[Tuple], write_queue) -> int:
        """
        Queue-only writer.

        Args:
            table: Table name, which table to write to
            rows: Output from _output_df_to_rows/ _qa_df_to_rows
            write_queue: The batch queue
        """
        if not rows:
            return 0
        if write_queue is None:
            raise RuntimeError("write_queue must be provided (closed system: queue-only writes).")

        write_queue.put(WriteBatch(table=table, rows=rows))
        return len(rows)
    
    def write_output(self, df: pd.DataFrame, write_queue=None) -> int:
        """
        Write TAK output to OutputPatientData.
        Thread-safe: uses provided DataAccess instance if given.
        
        Args:
            df: DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
            write_queue: Queue to manage writing operations
        
        Returns:
            Number of rows actually inserted (excluding duplicates)
        """
        if df.empty:
            return 0
        
        # Convert to tuples for executemany
        rows = self._output_df_to_rows(df)
        return self._write_rows("output", rows, write_queue)
    
    def write_qa_scores(self, df_scores: pd.DataFrame, write_queue=None) -> int:
        """
        Write Pattern compliance scores to PatientQAScores table.
        Thread-safe: uses provided DataAccess instance if given.
        """
        if df_scores.empty:
            return 0
        
        # Convert to tuples for executemany
        rows = self._qa_df_to_rows(df_scores)
        return self._write_rows("qa", rows, write_queue)

    def _get_global_clipper_times(self, patient_id: int, da: DataAccess) -> Optional[pd.DataFrame]:
        """
        Query InputPatientData for global clipper events (once per patient, before TAK processing).
        Returns DataFrame with columns: ConceptName, StartDateTime, or None if no clippers found.
        """
        if not self.global_clippers:
            return None
        
        clipper_names = list(self.global_clippers.keys())
        table = "InputPatientData"
        
        # Load query template and replace placeholders
        with open(GET_DATA_BY_PATIENT_CONCEPTS_QUERY, 'r') as f:
            query_template = f.read()
        
        query = query_template.replace("{table}", table)
        placeholders = ','.join('?' * len(clipper_names))
        query = query.replace("{CONCEPT_PLACEHOLDERS}", placeholders)
        
        params = (patient_id, *clipper_names)
        rows = da.fetch_records(query, params)
        
        if not rows:
            return None
        
        # Return DataFrame with ConceptName and StartDateTime (clippers are point events)
        df = pd.DataFrame(rows, columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"])
        return df[["ConceptName", "StartDateTime", "EndDateTime"]]
    
    def _apply_global_clippers(self, df: pd.DataFrame, clipper_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Apply global clippers to abstraction output.
        Clips events to be strictly within the start and end clippers.
        Preserving events that start exactly at the start clippers and events that end exactly on the end clippers, and are 1s long.
        """
        if df.empty or clipper_df is None or clipper_df.empty:
            return df
        
        # Working on a copy
        df = df.copy()
        
        # 1. Get Wall Times
        start_clippers = clipper_df[clipper_df["ConceptName"].isin(
            [name for name, how in self.global_clippers.items() if how == "START"]
        )]
        end_clippers = clipper_df[clipper_df["ConceptName"].isin(
            [name for name, how in self.global_clippers.items() if how == "END"]
        )]
        
        # 2. Apply Start Clipping
        if not start_clippers.empty:
            min_start = start_clippers["StartDateTime"].min()
            
            # Clip Start: strictly >= Admission
            # If Event starts at min_start (T), T < T is False -> No Clip.
            mask = df["StartDateTime"] < min_start
            if mask.any():
                df.loc[mask, "StartDateTime"] = min_start

        # 3. Apply End Clipping
        if not end_clippers.empty:
            max_end = end_clippers["EndDateTime"].max()
            
            # Clip End: strictly <= Release
            # If Event ends at max_end (T), T > T is False -> No Clip.
            mask = df["EndDateTime"] > max_end
            if mask.any():
                df.loc[mask, "EndDateTime"] = max_end

        # 4. Drop Invalid Intervals (Duration <= 0)
        # Events that were fully outside will now have Start >= End and be dropped.
        # Events that were [T, T+1] and Wall=T will remain [T, T+1] and be kept.
        valid_mask = df["StartDateTime"] < df["EndDateTime"]
        
        dropped_count = (~valid_mask).sum()
        if dropped_count > 0:
            df = df[valid_mask].copy()

        return df.sort_values("StartDateTime").reset_index(drop=True)
    
    def _split_pattern_output(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split Pattern output into main output (for OutputPatientData) and QA scores (for PatientQAScores).
        
        Main output: Rows with valid timestamps (True/Partial patterns)
        QA scores: All rows (including False patterns with NaT timestamps)
        
        Args:
            df: Pattern output with columns [PatientId, ConceptName, StartDateTime, EndDateTime, 
                                            Value, TimeConstraintScore, ValueConstraintScore, CyclicConstraintScore, AbstractionType]
        
        Returns:
            Tuple of (main_output_df, qa_scores_df)
        """
        if df.empty:
            return (
                pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"]),
                pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore"])
            )
        
        # We filter Generic "no pattern found" rows (Start=NaT, End=NaT)
        # 2. Specific "missed opportunity" rows (Start=Valid, End=Valid) remain
        df_main = df[df["StartDateTime"].notna() & (df["EndDateTime"].notna())][["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"]].copy()
        
        # QA scores: all rows (including "False" with NaT)
        # Extract compliance score columns
        df_scores = df[["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "TimeConstraintScore", "ValueConstraintScore", "CyclicConstraintScore"]].copy()
        
        # Fill NaT with sentinel date for DB insertion (approx. 2262-04-11 23:47:16)
        sentinel_date = pd.Timestamp.max
    
        df_scores.loc[:, 'StartDateTime'] = df_scores['StartDateTime'].fillna(sentinel_date)
        df_scores.loc[:, 'EndDateTime'] = df_scores['EndDateTime'].fillna(sentinel_date)
        
        return df_main, df_scores
    
    def _save_failed_patients_json(self, failed_patient_ids: List[int], out_path: Path, meta: Optional[dict] = None) -> None:
        payload = {
            "failed_patient_ids": sorted(set(int(x) for x in failed_patient_ids)),
            "meta": meta or {},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _process_patient_sync(self, 
                              patient_id: int,
                              return_cache: bool = False,
                              write_queue=None
                              ) -> Union[Dict[str, Union[int, str]], Tuple[Dict[str, Union[int, str]], Dict[str, pd.DataFrame]]]:
        """
        Process single patient synchronously with full TAK dependency chain.
        
        Args:
            patient_id: Patient ID to process
            return_cache: If True, return (stats, tak_outputs_cache) for debugging
            write_queue: Control concurrent writing
        
        Returns:
            If return_cache=False: Dict of TAK name -> output row count
            If return_cache=True: Tuple of (stats, tak_outputs_cache)
        """
        if write_queue is None:
            raise RuntimeError("write_queue must be provided. Direct DB writes are disabled in this pipeline to allow parallelism.")
        # Create new DB connection for this thread (SQLite thread safety)
        thread_da = DataAccess(db_path=self.db_path)  # always per-process/per-thread read connection
        
        logger.info(f"[Patient {patient_id}] Processing start")
        stats = {}
        tak_outputs = {}  # Cache ALL TAK outputs for this patient_id {tak.name: tak.apply() df}
        
        try:
            # Phase 0: Query global clipper times ONCE per patient
            clipper_df = self._get_global_clipper_times(patient_id, thread_da)
            execution_order = getattr(self.repo, 'execution_order', list(self.repo.taks.keys()))

            # Process TAKs in topological order
            for tak_name in execution_order:
                tak = self.repo.get(tak_name)
                try:
                    # --- Step 1: Get input data (READ-ONLY cache access) ---
                    df_input = self.get_input_for_tak(patient_id, tak, tak_outputs, thread_da)
                    # Will not skip on empty inputs - TAK.apply() will know how to return them.
                    
                    # --- Step 2: Apply TAK ---
                    df_output = tak.apply(df_input)

                    # --- Step 3: Write to DB + cache output ---
                    if isinstance(tak, Pattern):
                        # Patterns: split into main output + QA scores
                        df_output_main, df_output_scores = self._split_pattern_output(df_output)
                        
                        # Apply global clippers (only to main output, not QA scores with NaT)
                        df_output_main = self._apply_global_clippers(df_output_main, clipper_df)
                        
                        # Update cache with main output (for downstream patterns)
                        tak_outputs[tak_name] = df_output_main
                        
                        rows_written = len(df_output_main)
                        self.write_output(df_output_main, write_queue)
                        self.write_qa_scores(df_output_scores, write_queue)
                        
                        # Add to stats the number of temporal rows written
                        stats[tak_name] = rows_written

                    else:
                        # All other TAKs: apply global clippers before caching/writing
                        df_output = self._apply_global_clippers(df_output, clipper_df)
                        
                        tak_outputs[tak_name] = df_output
                        rows_written = len(df_output)
                        
                        # All other TAKs: write to OutputPatientData (skip RawConcepts)
                        if not isinstance(tak, RawConcept):
                            self.write_output(df_output, write_queue)
                        
                        stats[tak_name] = rows_written
                    
                except Exception as e:
                    error_msg = f"ERROR: {str(e)}"
                    logger.error(f"[Patient {patient_id}][{tak_name}] {error_msg}")
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
                    stats[tak_name] = error_msg
                    
                    # Store empty DataFrame in cache on error
                    tak_outputs[tak_name] = pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
            
            logger.info(f"[Patient {patient_id}] Processing complete | stats={stats}")
            return (stats, tak_outputs) if return_cache else stats
            
        except Exception as e:
            logger.error(f"[Patient {patient_id}] Critical error: {e}", exc_info=True)
            return ({"error": str(e)}, tak_outputs) if return_cache else {"error": str(e)}
        finally:
            # Close thread-local connection
            thread_da.conn.close()

    def process_all_patients_parallel(
        self,
        max_concurrent: int = 0,
        patient_subset: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process all patients through TAK pipeline with multiprocessing.
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes (0 = all cores)
            patient_subset: Optional list of patient IDs to process. If None, processes all patients.
        
        Returns:
            Dict mapping patient_id → {tak_name: rows_written}
        """
        # Use all available cores if max_concurrent is <= 0
        if max_concurrent <= 0:
            max_concurrent = os.cpu_count() or 4
        
        patient_ids = self.get_patient_ids(patient_subset=patient_subset)
        
        if not patient_ids:
            print("[Warning] No patients found (check patient_subset or InputPatientData).")
            return {}
        
        print("\n" + "="*80)
        print(f"PHASE 2: Processing {len(patient_ids)} Patients (max_workers={max_concurrent})")
        if patient_subset:
            print(f"         Patient Subset: {patient_ids}")
        print("="*80 + "\n")

        # Use a consistent MP context (important on Windows, and generally safer)
        ctx = mp.get_context("spawn")
        
        # Add start writer arg
        manager = ctx.Manager()
        write_queue = manager.Queue(maxsize=200)  # picklable proxy queue (Windows spawn safe) with backpressure limit
        writer_proc = ctx.Process(
            target=_writer_worker, 
            args=(str(self.db_path), write_queue),
            daemon=False
        )
        writer_proc.start()
        
        # Submit workers with write_queue included
        patient_stats = {}
        errors = 0
        failed_patient_ids: List[int] = []
        total_rows = 0
        with ProcessPoolExecutor(max_workers=max_concurrent, mp_context=ctx) as executor:
            future_to_pid = {}
            for pid in patient_ids:
                fut = executor.submit(
                    _process_patient_worker,
                    pid,
                    str(self.kb_path),
                    str(self.db_path),
                    self.global_clippers,
                    write_queue,   # IMPORTANT
                )
                future_to_pid[fut] = pid

            for fut in tqdm(as_completed(future_to_pid), total=len(future_to_pid), desc="Processing patients", unit="patient"):
                pid = future_to_pid[fut]
                result = fut.result()
                patient_stats[pid] = result

                # total rows: count only ints (rows written per TAK)
                if isinstance(result, dict) and "error" not in result:
                    total_rows += sum(v for v in result.values() if isinstance(v, int))

        # Stop writer cleanly
        write_queue.put(None)   # sentinel
        writer_proc.join()
        manager.shutdown()

        # Determine failed patients (either fatal error or any TAK-level ERROR)
        for pid, stats in patient_stats.items():
            if not isinstance(stats, dict):
                failed_patient_ids.append(pid)
                continue

            if "error" in stats:
                failed_patient_ids.append(pid)
                continue

            if any(isinstance(v, str) and v.startswith("ERROR") for v in stats.values()):
                failed_patient_ids.append(pid)

        errors = len(set(failed_patient_ids))

        failed_json_path = Path(self.db_path).parent / "failed_patients.json"
        self._save_failed_patients_json(
            failed_patient_ids,
            failed_json_path,
            meta={
                "db_path": str(self.db_path),
                "kb_path": str(self.kb_path),
                "n_patients_requested": len(patient_ids),
                "n_patients_failed": len(set(failed_patient_ids)),
            }
        )

        print("\n" + "="*80)
        print("✅ Patient Processing Complete")
        print("="*80)
        print(f"  Patients processed: {len(patient_ids)}")
        print(f"  Total rows written: {total_rows}")
        print(f"  Errors:             {errors}")
        if failed_patient_ids:
            print(f"[Info] Failed patients saved to: {failed_json_path}")
        print("="*80 + "\n")

        return patient_stats
    
    def run(
        self,
        max_concurrent: int = 4,
        patient_subset: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Main entry point: Build repository → Process patients → Write outputs.
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes (0 = all cores)
            patient_subset: Optional list of patient IDs to process
        """
        # Phase 1: Build TAK repository
        self.build_repository()
        
        # Phase 2: Process patients (multiprocessing)
        patient_stats = self.process_all_patients_parallel(
            max_concurrent=max_concurrent,
            patient_subset=patient_subset
        )
        
        return patient_stats
    

def _cli_main(argv):
    """CLI entry point for Mediator pipeline."""
    parser = argparse.ArgumentParser(
        prog="mediator",
        description="Run Mediator KBTA pipeline on patient data",
    )

    parser.add_argument(
        "--kb",
        type=Path,
        default=TAK_FOLDER,
        help=f"Path to knowledge base folder (default: {TAK_FOLDER})"
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to database file (default: {DB_PATH})"
    )

    parser.add_argument(
        "--patients",
        type=str,
        help="Comma-separated patient IDs (e.g., '1,2,3'). If omitted, processes all patients."
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent patient processes (0 = all cores). Default: 4"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for file logs (default: INFO)"
    )

    args = parser.parse_args(argv)

    # Setup log file
    log_file = Path(args.db).parent / "mediator_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Validate inputs
    if not args.kb.exists():
        print(f"[Error] Knowledge base folder not found: {args.kb}")
        sys.exit(1)

    if not args.db.exists():
        print(f"[Error] Database file not found: {args.db}")
        print("Run: python -m backend.dataaccess --create_db --load_csv <file>")
        sys.exit(1)

    patient_subset = None
    if args.patients:
        try:
            patient_subset = [int(p.strip()) for p in args.patients.split(",") if p.strip()]
        except ValueError as e:
            print(f"[Error] Invalid patient ID format: {e}")
            sys.exit(1)

    # Initialize and run
    da = DataAccess(db_path=str(args.db))
    mediator = Mediator(knowledge_base_path=args.kb, data_access=da)

    patient_stats = mediator.run(
        max_concurrent=args.max_concurrent,
        patient_subset=patient_subset
    )

    errors = sum(1 for s in patient_stats.values() if isinstance(s, dict) and "error" in s)

    print("\n" + "=" * 80)
    print("Pipeline Complete")
    print("=" * 80)
    print(f"Patients processed: {len(patient_stats)}")
    print(f"Errors:             {errors}")
    print(f"Log file:           {log_file}")
    print("=" * 80)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    _cli_main(sys.argv[1:])
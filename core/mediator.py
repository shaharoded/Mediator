"""
TO-DO:
 - Need to add min-distance to "before" on pattern schema, that ignores events that are too close.
 - I need a flag to patterns that I didn't find. For example, had hyper, no following measure, so no score? I need a logic that for every hyper event must satisfy with measure after within reasonable time.
 - What happens on local pattern if I define something like glucose every 12 hours? Will the anchor catch the closest option? What will happen if I define a context to it as well? Might "skip" smaller pattern-intervals just because the larger one also fits, which might distory the QA.
 - We have empty QA row when no pattern is found, but what about when we only find patterns some of the days? How to say "there should have been an instance here"? Maybe global pattern?
 - For relative insulin dosage, need to allow 'default' to be optional in parameterized raw concept, for data leakage, limit parameter to before the event, if doesn't exist. parameters in compliance must demand default, but for anything temporal - not must.
 - check if trapezoid works backwards as well, meaning if I have min-distance=2h, and trapezoidA=0h, will it give score 0 for events 0-2h before anchor, or will it give score 1? Need to make sure it gives score 0.
 - Currently if I'm looking for "before" pattern, and I have the same pattern but as "overlap", this instance is not captured, meaning also no QA. need to think.
    - select='last' doesn't make sense with time constraint QA. need to think or block option.
 - define that max-distance=0 for 'before' will also capture 'overlap', so that if context window overlaps with event, it is included. As long as anchor.StartTime < event.StartTime, we can treat "before" as inclusive of overlap.
 - define Overlap(Pattern) to use for complex context. Should check if 2+ contexts (or any other concept) overlap and if so will return their overlap window (should include +- good before/after?).
 """

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import asyncio
import json
import logging
import traceback
import sys
import argparse

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .tak.tak import TAK
from .tak.repository import TAKRepository, set_tak_repository
from .tak.raw_concept import RawConcept, ParameterizedRawConcept
from .tak.event import Event
from .tak.state import State
from .tak.trend import Trend
from .tak.context import Context
from .tak.pattern import Pattern, LocalPattern
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


class Mediator:
    """
    Orchestrates the KBTA pipeline:
    1. Load & validate TAK repository (raw-concepts → events → states → trends → contexts → patterns)
    2. For each patient, for each TAK: extract filtered input → apply TAK → write output to DB
    3. Supports async parallelism for patient-level processing
    
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
    
    def build_repository(self) -> TAKRepository:
        """
        Load and validate all TAKs from knowledge base in dependency order.
        Raises on parsing/validation errors.
        
        Returns:
            TAKRepository instance (also stored in self.repo and set as global)
        """
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
            ("Patterns", self.kb_path / "patterns", LocalPattern, self.patterns)
        ]
        
        total_files = sum(len(list(path.glob("*.xml"))) for _, path, _, _ in phases if path.exists())
        
        with tqdm(total=total_files, desc="Loading TAKs", unit="file") as pbar:
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
        print("\n[Validation] Running business-logic checks on TAK repository...")
        try:
            repo.finalize_repository()
        except Exception as e:
            raise RuntimeError(f"TAK repository finalization failed: {e}") from e
        
        # Summary
        # Count unique TAKs from repository (avopids dupolicates on re-run)
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
    
    def write_output(self, df: pd.DataFrame, da: Optional[DataAccess] = None) -> int:
        """
        Write TAK output to OutputPatientData.
        Thread-safe: uses provided DataAccess instance if given.
        
        Args:
            df: DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
            da: Optional DataAccess instance (for thread safety)
        
        Returns:
            Number of rows actually inserted (excluding duplicates)
        """
        if df.empty:
            return 0
        
        # Use provided connection or default
        db = da if da is not None else self.da
        
        # Convert to tuples for executemany
        rows = [
            (
                int(row["PatientId"]),
                str(row["ConceptName"]),
                str(row["StartDateTime"]),
                str(row["EndDateTime"]),
                str(row["Value"]),
                str(row["AbstractionType"])
            )
            for _, row in df.iterrows()
        ]
        
        # Use INSERT OR IGNORE to skip duplicates
        return db.insert_many(INSERT_ABSTRACTED_MEASUREMENT_QUERY, rows, batch_size=1000)
    
    def write_qa_scores(self, df_scores: pd.DataFrame, da: Optional[DataAccess] = None) -> int:
        """
        Write Pattern compliance scores to PatientQAScores table.
        Thread-safe: uses provided DataAccess instance if given.
        """
        if df_scores.empty:
            return 0
        
        # Use provided connection or default
        db = da if da is not None else self.da
        
        # OPTIMIZED: Vectorized unpivot using pd.melt
        df_melted = df_scores.melt(
            id_vars=["PatientId", "ConceptName", "StartDateTime", "EndDateTime"],
            value_vars=["TimeConstraintScore", "ValueConstraintScore"],
            var_name="ComplianceType",
            value_name="ComplianceScore"
        )
        
        df_melted = df_melted.dropna(subset=["ComplianceScore"])
        
        if df_melted.empty:
            return 0
        
        df_melted["ComplianceType"] = df_melted["ComplianceType"].map({
            "TimeConstraintScore": "TimeConstraint",
            "ValueConstraintScore": "ValueConstraint"
        })
        
        rows = [
            (
                int(row["PatientId"]),
                str(row["ConceptName"]),
                str(row["StartDateTime"]) if pd.notna(row["StartDateTime"]) else None,
                str(row["EndDateTime"]) if pd.notna(row["EndDateTime"]) else None,
                str(row["ComplianceType"]),
                float(row["ComplianceScore"])
            )
            for _, row in df_melted.iterrows()
        ]
        
        return db.insert_many(INSERT_QA_SCORE_QUERY, rows, batch_size=1000)

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
        return df[["ConceptName", "StartDateTime"]]
    
    def _apply_global_clippers(self, df: pd.DataFrame, clipper_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Apply global clippers to abstraction output by trimming interval boundaries.
        - START clippers: if row.StartDateTime <= clipper_time, set StartDateTime = clipper_time + 1s
        - END clippers: if row.EndDateTime >= clipper_time, set EndDateTime = clipper_time - 1s
        - Drop rows where StartDateTime >= EndDateTime after clipping (invalid intervals)
        
        EXCEPTION: Do not clip:
          1. TAKs listed in global_clippers.json (prevents self-clipping)
          2. Raw concepts (foundational data should not be clipped)
        
        Uses vectorized operations for efficient processing.
        
        Args:
            df: Abstraction output DataFrame
            clipper_df: DataFrame with columns [ConceptName, StartDateTime] from InputPatientData
        
        Returns:
            Clipped DataFrame (with adjusted boundaries)
        """
        if df.empty or clipper_df is None or clipper_df.empty:
            return df
        
        # Exclude clipper TAKs AND raw concepts from being clipped
        clipper_names = set(self.global_clippers.keys())
        
        # Split into clippable vs non-clippable
        # Non-clippable: clippers + all raw concepts
        is_raw_concept = df["AbstractionType"] == "raw-concept"
        is_clipper = df["ConceptName"].isin(clipper_names)
        df_non_clippable = df[is_raw_concept | is_clipper]
        df_clippable = df[~(is_raw_concept | is_clipper)]
        
        # If nothing to clip, return as-is
        if df_clippable.empty:
            return df
        
        # Only clip non-clipper, non-raw-concept TAKs
        df = df_clippable.copy()
        
        # OPTIMIZATION 2: Extract clipper times as numpy arrays (vectorized comparisons)
        start_clippers = clipper_df[clipper_df["ConceptName"].isin(
            [name for name, how in self.global_clippers.items() if how == "START"]
        )]
        
        end_clippers = clipper_df[clipper_df["ConceptName"].isin(
            [name for name, how in self.global_clippers.items() if how == "END"]
        )]
        
        # OPTIMIZATION 3: Compute boundaries ONCE (not per row)
        if not start_clippers.empty:
            min_start_time = start_clippers["StartDateTime"].min()
            # Adjust all rows where StartDateTime <= min_start_time
            mask = df["StartDateTime"] <= min_start_time
            if mask.any():
                df.loc[mask, "StartDateTime"] = min_start_time + pd.Timedelta(seconds=1)
        
        if not end_clippers.empty:
            max_end_time = end_clippers["StartDateTime"].max()
            # Vectorized: adjust all rows where EndDateTime >= max_end_time
            mask = df["EndDateTime"] >= max_end_time
            if mask.any():
                df.loc[mask, "EndDateTime"] = max_end_time - pd.Timedelta(seconds=1)
        
        # OPTIMIZATION 4: Drop invalid intervals (StartDateTime >= EndDateTime) in one pass
        valid_mask = df["StartDateTime"] < df["EndDateTime"]
        dropped_count = (~valid_mask).sum()
        if dropped_count > 0:
            # Log which concepts were affected
            invalid_rows = df[~valid_mask]
            affected_concepts = invalid_rows["ConceptName"].value_counts().to_dict()
            concepts_str = ", ".join(f"{concept}: {count}" for concept, count in affected_concepts.items())
            logger.info(
                f"[Global Clippers] Dropped {dropped_count} invalid intervals "
                f"(StartDateTime >= EndDateTime after clipping) | Affected concepts: {concepts_str}"
            )
        
        # Concatenate non-clippable rows back
        if not df_non_clippable.empty:
            df_clipped = df[valid_mask]
            return pd.concat([df_non_clippable, df_clipped], ignore_index=True).sort_values("StartDateTime").reset_index(drop=True)
        
        return df[valid_mask]
    
    def _split_pattern_output(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split Pattern output into main output (for OutputPatientData) and QA scores (for PatientQAScores).
        
        Main output: Rows with valid timestamps (True/Partial patterns)
        QA scores: All rows (including False patterns with NaT timestamps)
        
        Args:
            df: Pattern output with columns [PatientId, ConceptName, StartDateTime, EndDateTime, 
                                            Value, TimeConstraintScore, ValueConstraintScore, AbstractionType]
        
        Returns:
            Tuple of (main_output_df, qa_scores_df)
        """
        if df.empty:
            return (
                pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"]),
                pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "TimeConstraintScore", "ValueConstraintScore"])
            )
        
        # Main output: only rows with valid timestamps (not NaT)
        # This filters out "False" patterns (which have NaT timestamps)
        df_main = df[df["StartDateTime"].notna()][["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"]].copy()
        
        # QA scores: all rows (including "False" with NaT)
        # Extract compliance score columns
        df_scores = df[["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "TimeConstraintScore", "ValueConstraintScore"]].copy()
        
        # Fill NaT with sentinel date for DB insertion (approx. 2262-04-11 23:47:16)
        sentinel_date = pd.Timestamp.max
    
        df_scores.loc[:, 'StartDateTime'] = df_scores['StartDateTime'].fillna(sentinel_date)
        df_scores.loc[:, 'EndDateTime'] = df_scores['EndDateTime'].fillna(sentinel_date)
        
        return df_main, df_scores

    def _process_patient_sync(self, 
                              patient_id: int,
                              return_cache: bool = False
                              ) -> Union[Dict[str, Union[int, str]], Tuple[Dict[str, Union[int, str]], Dict[str, pd.DataFrame]]]:
        """
        Process single patient synchronously with full TAK dependency chain.
        
        Args:
            patient_id: Patient ID to process
            return_cache: If True, return (stats, tak_outputs_cache) for debugging
        
        Returns:
            If return_cache=False: Dict of TAK name -> output row count
            If return_cache=True: Tuple of (stats, tak_outputs_cache)
        """
        # Create new DB connection for this thread (SQLite thread safety)
        thread_da = DataAccess(db_path=self.db_path)
        
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
                        self.write_output(df_output_main, thread_da)
                        self.write_qa_scores(df_output_scores, thread_da)
                        
                        # Add to stats the number of temporal rows written
                        stats[tak_name] = rows_written

                    else:
                        # All other TAKs: apply global clippers before caching/writing
                        df_output = self._apply_global_clippers(df_output, clipper_df)
                        
                        tak_outputs[tak_name] = df_output
                        rows_written = len(df_output)
                        
                        # All other TAKs: write to OutputPatientData (skip RawConcepts)
                        if not isinstance(tak, RawConcept):
                            self.write_output(df_output, thread_da)
                        
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

    async def process_patient_async(self, patient_id: int, semaphore: asyncio.Semaphore) -> Dict[str, int]:
        """
        Process a single patient through the entire TAK pipeline (async).
        
        Args:
            patient_id: Patient ID
            semaphore: Asyncio semaphore for concurrency control
        
        Returns:
            Dict with stats: {tak_name: rows_written}
        """
        async with semaphore:
            # Run synchronous TAK processing in executor (blocking operations)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_patient_sync, patient_id)
    
    async def process_all_patients_async(
        self,
        max_concurrent: int = 4,
        patient_subset: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process all patients through TAK pipeline with async parallelism.
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes
            patient_subset: Optional list of patient IDs to process. If None, processes all patients.
        
        Returns:
            Dict mapping patient_id → {tak_name: rows_written}
        """
        patient_ids = self.get_patient_ids(patient_subset=patient_subset)
        
        if not patient_ids:
            print("[Warning] No patients found (check patient_subset or InputPatientData).")
            return {}
        
        print("\n" + "="*80)
        print(f"PHASE 2: Processing {len(patient_ids)} Patients (max_concurrent={max_concurrent})")
        if patient_subset:
            print(f"         Patient Subset: {patient_ids}")
        print("="*80 + "\n")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [
            self.process_patient_async(pid, semaphore)
            for pid in patient_ids
        ]
        
        # Process with progress bar
        results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing patients", unit="patient"):
            result = await coro
            results.append(result)
        
        # Map results back to patient IDs
        patient_stats = dict(zip(patient_ids, results))
        
        # Summary (filter out error values properly)
        total_rows = sum(
            sum(v for v in stats.values() if isinstance(v, int))
            for stats in results 
            if isinstance(stats, dict) and "error" not in stats
        )
        errors = sum(1 for stats in results if isinstance(stats, dict) and "error" in stats)
        
        print("\n" + "="*80)
        print("✅ Patient Processing Complete")
        print("="*80)
        print(f"  Patients processed: {len(patient_ids)}")
        print(f"  Total rows written: {total_rows}")
        print(f"  Errors:             {errors}")
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
            max_concurrent: Maximum number of concurrent patient processes
            patient_subset: Optional list of patient IDs to process
        """
        # Phase 1: Build TAK repository
        self.build_repository()
        
        # Phase 2: Process patients (async)
        patient_stats = asyncio.run(
            self.process_all_patients_async(
                max_concurrent=max_concurrent,
                patient_subset=patient_subset
            )
        )
        
        return patient_stats
    
    async def run_async(
        self,
        max_concurrent: int = 4,
        patient_subset: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Jupyter-friendly version: Build repository → Process patients (already in event loop).
        
        Usage in Jupyter:
            patient_stats = await mediator.run_async(patient_subset=[1000, 1001])
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes
            patient_subset: Optional list of patient IDs to process
        """
        # Phase 1: Build TAK repository (sync)
        self.build_repository()
        
        # Phase 2: Process patients (async, reuses existing event loop)
        patient_stats = await self.process_all_patients_async(
            max_concurrent=max_concurrent,
            patient_subset=patient_subset
        )
        
        return patient_stats
    

def _cli_main(argv):
    """CLI entry point for Mediator pipeline."""
    parser = argparse.ArgumentParser(
        prog="mediator",
        description="Run Mediator KBTA pipeline on patient data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients (default KB and DB paths)
  python -m core.mediator

  # Process specific patients
  python -m core.mediator --patients 1,2,3,4,5

  # Custom concurrency
  python -m core.mediator --max-concurrent 8

  # Custom KB and DB paths
  python -m core.mediator --kb core/knowledge-base --db data/mediator.db

  # Debug logging + patient subset
  python -m core.mediator --patients 101,102,103 --log-level DEBUG
        """
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
        default=DB_PATH,numeric_level = getattr(logging, args.log_level.upper(), logging.INFO),
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
        help="Maximum concurrent patient processes (default: 4)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args(argv)
    
    # Setup dual logging (console shows ERRORS only, file shows all)
    log_file = Path(args.db).parent / "mediator_run.log"  # backend/data/mediator_run.log
    
    # Remove existing log file (restart fresh each run)
    if log_file.exists():
        log_file.unlink()
    
    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')  # Minimal for console
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler: ERROR and above only (no warnings/info in terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)  # CORRECTED: Only show errors in console
    console_handler.setFormatter(console_formatter)
    
    # File handler: ALL levels (verbose logs)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(numeric_level)  # Respect --log-level flag
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture everything
        handlers=[console_handler, file_handler]
    )
    
    logger.info("="*80)
    logger.info(f"Mediator Run Started | Log Level: {args.log_level}")
    logger.info(f"Knowledge Base: {args.kb}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*80)
    
    # Validate inputs
    if not args.kb.exists():
        print(f"[Error] Knowledge base folder not found: {args.kb}")
        sys.exit(1)
    
    if not args.db.exists():
        print(f"[Error] Database file not found: {args.db}")
        print("Run: python -m backend.dataaccess --create_db --load_csv <file>")
        sys.exit(1)
    
    # Parse patient subset
    patient_subset = None
    if args.patients:
        try:
            patient_subset = [int(p.strip()) for p in args.patients.split(',')]
            logger.info(f"Processing patient subset: {patient_subset}")
        except ValueError as e:
            print(f"[Error] Invalid patient ID format: {e}")
            sys.exit(1)
    
    # Initialize Mediator
    try:
        da = DataAccess(db_path=str(args.db))
        mediator = Mediator(knowledge_base_path=args.kb, data_access=da)
    except Exception as e:
        print(f"[Error] Failed to initialize Mediator: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        patient_stats = mediator.run(
            max_concurrent=args.max_concurrent,
            patient_subset=patient_subset
        )
        
        # Final summary
        errors = sum(1 for stats in patient_stats.values() if "error" in stats)
        total_rows = sum(
            sum(v for k, v in stats.items() if k != "error" and isinstance(v, int))
            for stats in patient_stats.values()
            if "error" not in stats
        )
        
        print("\n" + "="*80)
        print("Pipeline Complete")
        print("="*80)
        print(f"Patients processed: {len(patient_stats)}")
        print(f"Total rows written: {total_rows}")
        if errors > 0:
            print(f"Errors: {errors} patients failed")
        print("="*80)
        
        logger.info("="*80)
        logger.info("Pipeline Complete Successfully")
        logger.info("="*80)
        
        sys.exit(0 if errors == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n[Warning] Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        logging.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main(sys.argv[1:])
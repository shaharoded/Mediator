"""
TO-DO:
- Differentiate between repo building logs which are important vs. patient processing logs which can be verbose, and should probably be stored in a log file, and produce end-of-run stats.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import asyncio
import logging
import json
import sys
import argparse
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .tak.tak import TAKRepository, set_tak_repository, TAK
from .tak.raw_concept import RawConcept
from .tak.event import Event
from .tak.state import State
from .tak.trend import Trend
from .tak.context import Context
from .tak.pattern import Pattern  # TODO: Implement Pattern TAK
from .config import TAK_FOLDER

# Add parent directory to path for backend imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.dataaccess import DataAccess 
from backend.config import (
    DB_PATH,
    INSERT_ABSTRACTED_MEASUREMENT_QUERY,
    GET_DATA_BY_PATIENT_CONCEPTS_QUERY)


logger = logging.getLogger(__name__)


class Mediator:
    """
    Orchestrates the KBTA pipeline:
    1. Load & validate TAK repository (raw-concepts → events → states → trends → contexts → patterns)
    2. For each patient, for each TAK: extract filtered input → apply TAK → write output to DB
    3. Supports async parallelism for patient-level processing
    
    Design: Full caching — all TAK outputs cached in memory during patient processing (~500KB per patient).
    """
    
    def __init__(self, 
                 knowledge_base_path: Path, 
                 data_access: DataAccess):
        """
        Initialize Mediator with knowledge base and DB access.
        
        Args:
            knowledge_base_path: Path to knowledge-base folder (contains raw-concepts/, events/, states/, trends/, contexts/)
            data_access: DataAccess instance for DB operations
        """
        self.kb_path = Path(knowledge_base_path)
        self.da = data_access
        self.repo: Optional[TAKRepository] = None
        
        # TAK execution order (dependencies resolved), populated in build_repository()
        # Defines the execution order for applying TAKs to patients, which TAKRepository does not track (key - value store only)
        self.raw_concepts: List[RawConcept] = []
        self.events: List[Event] = []
        self.states: List[State] = []
        self.trends: List[Trend] = []
        self.contexts: List[Context] = []
        self.patterns: List[Pattern] = []  # TODO: Implement Pattern TAK
        
        # Load global clippers (START/END events that clip all abstractions)
        self.global_clippers = self._load_global_clippers()
    
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
        
        # Load TAKs in dependency order with progress tracking
        phases = [
            ("Raw Concepts", self.kb_path / "raw-concepts", RawConcept, self.raw_concepts),
            ("Events", self.kb_path / "events", Event, self.events),
            ("States", self.kb_path / "states", State, self.states),
            ("Trends", self.kb_path / "trends", Trend, self.trends),
            ("Contexts", self.kb_path / "contexts", Context, self.contexts),
            # ("Patterns", self.kb_path / "patterns", Pattern, self.patterns),  # TODO
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
        
        # Set global repository (required for cross-TAK validation)
        set_tak_repository(repo)
        
        # Run business-logic validation on all TAKs
        print("\n[Validation] Running business-logic checks on TAK repository...")
        try:
            repo.validate_all()
        except Exception as e:
            raise RuntimeError(f"TAK validation failed: {e}") from e
        
        # Summary
        print("\n" + "="*80)
        print("✅ TAK Repository Built Successfully")
        print("="*80)
        print(f"  Raw Concepts: {len(self.raw_concepts)}")
        print(f"  Events:       {len(self.events)}")
        print(f"  States:       {len(self.states)}")
        print(f"  Trends:       {len(self.trends)}")
        print(f"  Contexts:     {len(self.contexts)}")
        print(f"  Patterns:     {len(self.patterns)}")
        print(f"  TOTAL TAKs:   {len(repo.taks)}")
        print("="*80 + "\n")
        
        self.repo = repo
        return repo
    
    def get_patient_ids(self, patient_subset: Optional[List[int]] = None) -> List[int]:
        """
        Retrieve patient IDs from InputPatientData.
        
        Args:
            patient_subset: Optional list of patient IDs to filter. If None, returns all patients.
        
        Returns:
            List of patient IDs (sorted)
        """
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
    
    def get_input_for_raw_concept(self, patient_id: int, raw_concept: RawConcept) -> pd.DataFrame:
        """
        Query InputPatientData for a RawConcept's df by its attributes.
        Returns DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
        """
        concept_names = [a["name"] for a in raw_concept.attributes]
        table = "InputPatientData"
        
        # Load query template and replace placeholders
        with open(GET_DATA_BY_PATIENT_CONCEPTS_QUERY, 'r') as f:
            query_template = f.read()
        
        query = query_template.replace("{table}", table)
        placeholders = ','.join('?' * len(concept_names))
        query = query.replace("{CONCEPT_PLACEHOLDERS}", placeholders)
        
        params = (patient_id, *concept_names)
        rows = self.da.fetch_records(query, params)
        
        if not rows:
            return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
        
        df = pd.DataFrame(rows, columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
        df["AbstractionType"] = "raw-concept"  # dummy value for input data
        return df
    
    def get_input_for_tak(self, patient_id: int, tak: TAK, tak_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Recursively resolve dependencies and build input DataFrame for a TAK.
        Uses cached tak_outputs to provide TAK DataFrames to derived TAKs (avoids redundant computation).
        
        Args:
            patient_id: Patient ID
            tak: TAK instance to prepare input for
            tak_outputs: Cache of {tak_name: DataFrame} for ALL TAK families
        
        Returns:
            DataFrame ready for tak.apply()
        """
        # CASE 1: RawConcept → query InputPatientData (or use cache)
        if isinstance(tak, RawConcept):
            if tak.name in tak_outputs:
                return tak_outputs[tak.name]
            return self.get_input_for_raw_concept(patient_id, tak)
        
        # CASE 2: Event → resolve derived_from (list of raw-concepts)
        if isinstance(tak, Event):
            dfs = []
            for df_spec in tak.derived_from:
                parent_name = df_spec["name"]
                parent_tak = self.repo.get(parent_name)
                
                # Get from cache or compute
                if parent_name in tak_outputs:
                    df_parent = tak_outputs[parent_name]
                else:
                    df_parent = self.get_input_for_raw_concept(patient_id, parent_tak)
                    tak_outputs[parent_name] = df_parent
                
                dfs.append(df_parent)
            
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
            return pd.concat(dfs, ignore_index=True)
        
        # CASE 3: State → resolve single derived_from (RawConcept OR Event)
        if isinstance(tak, State):
            parent_name = tak.derived_from
            parent_tak = self.repo.get(parent_name)
            
            if isinstance(parent_tak, RawConcept):
                if parent_name in tak_outputs:
                    return tak_outputs[parent_name]
                df_parent = self.get_input_for_raw_concept(patient_id, parent_tak)
                tak_outputs[parent_name] = df_parent
                return df_parent
            elif isinstance(parent_tak, Event):
                # Check cache first (Event should already be computed)
                if parent_name in tak_outputs:
                    return tak_outputs[parent_name]
                # Compute Event if not cached (shouldn't happen with correct execution order)
                df_event_input = self.get_input_for_tak(patient_id, parent_tak, tak_outputs)
                df_event_output = parent_tak.apply(df_event_input)
                tak_outputs[parent_name] = df_event_output
                return df_event_output
            else:
                # Never happens due to prior validation
                raise RuntimeError(f"[{tak.name}] States can only derive from RawConcepts or Events")
        
        # CASE 4: Trend → resolve single derived_from (RawConcept only)
        if isinstance(tak, Trend):
            parent_name = tak.derived_from
            parent_tak = self.repo.get(parent_name)
            
            if parent_name in tak_outputs:
                return tak_outputs[parent_name]
            df_parent = self.get_input_for_raw_concept(patient_id, parent_tak)
            tak_outputs[parent_name] = df_parent
            return df_parent
        
        # CASE 5: Context → resolve derived_from (list of RawConcepts only) + clippers
        if isinstance(tak, Context):
            dfs = []
            
            # Resolve parent (derived_from) — ALL must be RawConcepts
            for df_spec in tak.derived_from:
                parent_name = df_spec["name"]
                parent_tak = self.repo.get(parent_name)
                if parent_tak is None:
                    logger.warning(f"[{tak.name}] Derived-from '{parent_name}' not found")
                    continue
                
                # Get from cache or compute
                if parent_name in tak_outputs:
                    df_parent = tak_outputs[parent_name]
                else:
                    df_parent = self.get_input_for_raw_concept(patient_id, parent_tak)
                    tak_outputs[parent_name] = df_parent
                
                dfs.append(df_parent)
            
            # Resolve clippers (also RawConcepts)
            if tak.clippers:
                for clipper in tak.clippers:
                    clipper_name = clipper["name"]
                    clipper_tak = self.repo.get(clipper_name)
                    if clipper_tak is None:
                        logger.warning(f"[{tak.name}] Clipper '{clipper_name}' not found")
                        continue
                    
                    if clipper_name in tak_outputs:
                        df_clipper = tak_outputs[clipper_name]
                    else:
                        df_clipper = self.get_input_for_raw_concept(patient_id, clipper_tak)
                        tak_outputs[clipper_name] = df_clipper
                    
                    dfs.append(df_clipper)
            
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
            return pd.concat(dfs, ignore_index=True)
        
        # CASE 6: Pattern → resolve derived_from (Events, States, Trends, Contexts)
        if isinstance(tak, Pattern):
            dfs = []
            
            for df_spec in tak.derived_from:
                parent_name = df_spec["name"]
                
                # Check cache (all dependencies should be computed by now)
                if parent_name not in tak_outputs:
                    logger.warning(f"[{tak.name}] Parent '{parent_name}' not found in cache (should be computed earlier)")
                    continue
                
                dfs.append(tak_outputs[parent_name])
            
            if not dfs:
                return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
            return pd.concat(dfs, ignore_index=True)
    
    def write_output(self, df: pd.DataFrame) -> int:
        """
        Write TAK output to OutputPatientData.
        
        Args:
            df: DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
        
        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0
        
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
        return self.da.insert_many(INSERT_ABSTRACTED_MEASUREMENT_QUERY, rows, batch_size=1000)
    
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
    
    def _get_global_clipper_times(self, patient_id: int) -> Optional[pd.DataFrame]:
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
        rows = self.da.fetch_records(query, params)
        
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
        
        Uses vectorized operations for efficient processing.
        
        Args:
            df: Abstraction output DataFrame
            clipper_df: DataFrame with columns [ConceptName, StartDateTime] from InputPatientData
        
        Returns:
            Clipped DataFrame (with adjusted boundaries)
        """
        if df.empty or clipper_df is None or clipper_df.empty:
            return df
        
        # OPTIMIZATION 1: Convert timestamps ONCE (avoid repeated conversions)
        df = df.copy()
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
        df["EndDateTime"] = pd.to_datetime(df["EndDateTime"])
        
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
            logger.info(f"[Global Clippers] Dropped {dropped_count} invalid intervals (StartDateTime >= EndDateTime after clipping)")
        
        return df[valid_mask]
    
    def _process_patient_sync(self, patient_id: int) -> Dict[str, int]:
        """Synchronous patient processing with full caching and global clippers."""
        stats = {}
        tak_outputs = {}  # Cache ALL TAK outputs
        
        try:
            # Phase 0: Query global clipper times ONCE per patient (from InputPatientData, BEFORE any TAK processing)
            clipper_df = self._get_global_clipper_times(patient_id)
            
            # Phase 1: Apply raw-concepts (cached, NOT written to DB)
            # NOTE: Raw-concepts use different input method (query InputPatientData directly)
            for rc in self.raw_concepts:
                df_input = self.get_input_for_raw_concept(patient_id, rc)
                if df_input.empty:
                    continue
                df_output = rc.apply(df_input)
                tak_outputs[rc.name] = df_output  # Cache
                stats[rc.name] = len(df_output)
            
            # Phases 2-6: Apply all other TAK families (unified loop)
            tak_families = [
                ("events", self.events),
                ("states", self.states),
                ("trends", self.trends),
                ("contexts", self.contexts),
                ("patterns", self.patterns),
            ]
            
            for family_name, tak_list in tak_families:
                for tak in tak_list:
                    df_input = self.get_input_for_tak(patient_id, tak, tak_outputs)
                    if df_input.empty:
                        continue
                    df_output = tak.apply(df_input)
                    
                    # Apply global clippers BEFORE caching/writing (using InputPatientData clippers)
                    df_output = self._apply_global_clippers(df_output, clipper_df)
                    
                    tak_outputs[tak.name] = df_output  # Cache
                    rows_written = self.write_output(df_output)
                    stats[tak.name] = rows_written
            
            return stats
            
        except Exception as e:
            logger.error(f"[Patient {patient_id}] Processing failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def process_all_patients_async(self, max_concurrent: int = 4, patient_subset: Optional[List[int]] = None) -> Dict[int, Dict[str, int]]:
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
        
        # Summary
        total_rows = sum(sum(stats.values()) for stats in results if "error" not in stats)
        errors = sum(1 for stats in results if "error" in stats)
        
        print("\n" + "="*80)
        print("✅ Patient Processing Complete")
        print("="*80)
        print(f"  Patients processed: {len(patient_ids)}")
        print(f"  Total rows written: {total_rows}")
        print(f"  Errors:             {errors}")
        print("="*80 + "\n")
        
        return patient_stats
    
    def run(self, max_concurrent: int = 4, patient_subset: Optional[List[int]] = None):
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
    
    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='[%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mediator_run.log', mode='a')
        ]
    )
    
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
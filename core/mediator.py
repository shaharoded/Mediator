"""
TO-DO:
- Add global clippers (ADMISSION, RELEASE, DEATH) that will clip all states/trends/patterns/contexts
"""

from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import asyncio
import logging
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .tak.tak import TAKRepository, set_tak_repository, TAK
from .tak.raw_concept import RawConcept
from .tak.event import Event
from .tak.state import State
from .tak.trend import Trend
from .tak.context import Context
# from .tak.pattern import Pattern  # TODO: Implement Pattern TAK

# Backend imports (moved to top)
from backend.config import INSERT_ABSTRACTED_MEASUREMENT_QUERY

logger = logging.getLogger(__name__)


class Mediator:
    """
    Orchestrates the KBTA pipeline:
    1. Load & validate TAK repository (raw-concepts → events → states → trends → contexts → patterns)
    2. For each patient: extract input → apply TAKs → write output to DB
    3. Supports async parallelism for patient-level processing
    """
    
    def __init__(self, knowledge_base_path: Path, data_access):
        """
        Initialize Mediator with knowledge base and DB access.
        
        Args:
            knowledge_base_path: Path to knowledge-base folder (contains raw-concepts/, events/, states/, trends/, contexts/)
            data_access: DataAccess instance for DB operations
        """
        self.kb_path = Path(knowledge_base_path)
        self.da = data_access
        self.repo: Optional[TAKRepository] = None
        
        # TAK execution order (dependencies resolved)
        self.raw_concepts: List[RawConcept] = []
        self.events: List[Event] = []
        self.states: List[State] = []
        self.trends: List[Trend] = []
        self.contexts: List[Context] = []
        self.patterns: List = []  # TODO: Implement Pattern TAK
        
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
        print("\n[Validation] Running business-logic checks...")
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
        print(f"  Patterns:     {len(self.patterns)} (TODO)")
        print(f"  TOTAL TAKs:   {len(repo.taks)}")
        print("="*80 + "\n")
        
        self.repo = repo
        return repo
    
    def get_patient_ids(self) -> List[int]:
        """Retrieve all unique patient IDs from InputPatientData."""
        query = "SELECT DISTINCT PatientId FROM InputPatientData ORDER BY PatientId;"
        rows = self.da.fetch_records(query, ())
        return [int(r[0]) for r in rows]
    
    def get_input_for_tak(self, patient_id: int, tak: TAK) -> pd.DataFrame:
        """
        Extract relevant input data for a TAK from InputPatientData or OutputPatientData.
        
        Args:
            patient_id: Patient ID
            tak: TAK instance (determines which concepts to query)
        
        Returns:
            DataFrame with columns: PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
        """
        # Determine source concepts based on TAK family
        if isinstance(tak, RawConcept):
            # Raw concepts query InputPatientData directly
            concept_names = [a["name"] for a in tak.attributes]
            table = "InputPatientData"
        elif isinstance(tak, Event):
            # Events query raw-concepts from InputPatientData
            concept_names = [df["name"] for df in tak.derived_from]
            table = "InputPatientData"
        elif isinstance(tak, (State, Trend)):
            # States/Trends query their parent from OutputPatientData
            concept_names = [tak.derived_from]
            table = "OutputPatientData"
        elif isinstance(tak, Context):
            # Contexts query their parent + clippers from OutputPatientData
            concept_names = [df["name"] for df in tak.derived_from]
            # Add clipper concept names
            if tak.clippers:
                concept_names.extend([c["name"] for c in tak.clippers])
            table = "OutputPatientData"
        else:
            raise ValueError(f"Unsupported TAK family: {tak.family}")
        
        # Build query with IN clause for concept names
        placeholders = ','.join('?' * len(concept_names))
        query = f"""
            SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value
            FROM {table}
            WHERE PatientId = ? AND ConceptName IN ({placeholders})
            ORDER BY StartDateTime ASC;
        """
        params = (patient_id, *concept_names)
        
        rows = self.da.fetch_records(query, params)
        if not rows:
            return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value", "AbstractionType"])
        
        df = pd.DataFrame(rows, columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
        
        # Add AbstractionType column (required by TAK.apply() signature)
        if table == "InputPatientData":
            df["AbstractionType"] = "raw-concept"  # dummy value for input data
        else:
            # Query AbstractionType from OutputPatientData
            query_abs = f"""
                SELECT ConceptName, AbstractionType
                FROM {table}
                WHERE PatientId = ? AND ConceptName IN ({placeholders})
                GROUP BY ConceptName;
            """
            abs_rows = self.da.fetch_records(query_abs, params)
            abs_map = {r[0]: r[1] for r in abs_rows}
            df["AbstractionType"] = df["ConceptName"].map(abs_map).fillna("unknown")
        
        return df
    
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
    
    def _process_patient_sync(self, patient_id: int) -> Dict[str, int]:
        """Synchronous patient processing (called from async executor)."""
        stats = {}
        
        try:
            # Phase 1: Apply raw-concepts (output cached in memory, not written to DB)
            raw_outputs = {}
            for rc in self.raw_concepts:
                df_input = self.get_input_for_tak(patient_id, rc)
                if df_input.empty:
                    continue
                df_output = rc.apply(df_input)
                raw_outputs[rc.name] = df_output
                stats[rc.name] = len(df_output)
            
            # Phase 2: Apply events (write to DB)
            for event in self.events:
                # Events query InputPatientData directly
                df_input = self.get_input_for_tak(patient_id, event)
                if df_input.empty:
                    continue
                df_output = event.apply(df_input)
                rows_written = self.write_output(df_output)
                stats[event.name] = rows_written
            
            # Phase 3: Apply states (write to DB)
            for state in self.states:
                # States query parent TAK from OutputPatientData (or raw_outputs cache)
                if state.derived_from in raw_outputs:
                    df_input = raw_outputs[state.derived_from]
                else:
                    df_input = self.get_input_for_tak(patient_id, state)
                
                if df_input.empty:
                    continue
                df_output = state.apply(df_input)
                rows_written = self.write_output(df_output)
                stats[state.name] = rows_written
            
            # Phase 4: Apply trends (write to DB)
            for trend in self.trends:
                if trend.derived_from in raw_outputs:
                    df_input = raw_outputs[trend.derived_from]
                else:
                    df_input = self.get_input_for_tak(patient_id, trend)
                
                if df_input.empty:
                    continue
                df_output = trend.apply(df_input)
                rows_written = self.write_output(df_output)
                stats[trend.name] = rows_written
            
            # Phase 5: Apply contexts (write to DB)
            for context in self.contexts:
                df_input = self.get_input_for_tak(patient_id, context)
                if df_input.empty:
                    continue
                df_output = context.apply(df_input)
                rows_written = self.write_output(df_output)
                stats[context.name] = rows_written
            
            # Phase 6: Apply patterns (TODO)
            # for pattern in self.patterns:
            #     ...
            
            return stats
            
        except Exception as e:
            logger.error(f"[Patient {patient_id}] Processing failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def process_all_patients_async(self, max_concurrent: int = 4) -> Dict[int, Dict[str, int]]:
        """
        Process all patients through TAK pipeline with async parallelism.
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes
        
        Returns:
            Dict mapping patient_id → {tak_name: rows_written}
        """
        patient_ids = self.get_patient_ids()
        
        if not patient_ids:
            print("[Warning] No patients found in InputPatientData.")
            return {}
        
        print("\n" + "="*80)
        print(f"PHASE 2: Processing {len(patient_ids)} Patients (max_concurrent={max_concurrent})")
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
    
    def run(self, max_concurrent: int = 4):
        """
        Main entry point: Build repository → Process patients → Write outputs.
        
        Args:
            max_concurrent: Maximum number of concurrent patient processes
        """
        # Phase 1: Build TAK repository
        self.build_repository()
        
        # Phase 2: Process all patients (async)
        patient_stats = asyncio.run(self.process_all_patients_async(max_concurrent=max_concurrent))
        
        return patient_stats
    


if __name__ == "__main__":

    # --- CONFIGURABLE INPUT ---
    patient_id = "123456782"
    snapshot_date = "2025-08-02 12:00:00"

    # --- RUN MEDIATOR ---
    engine = Mediator()
    result_df = engine.run(patient_id=patient_id, snapshot_date=snapshot_date)

    # --- DISPLAY RESULT ---
    if result_df.empty:
        print(f"[Info] No data available for Patient {patient_id} on snapshot {snapshot_date}")
    else:
        print(f"[Info] Abstracted records for Patient {patient_id} on {snapshot_date}:")
        print(result_df.to_string(index=False))
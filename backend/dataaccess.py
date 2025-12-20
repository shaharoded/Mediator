import sqlite3
import os
import time
import pandas as pd
import logging
from contextlib import contextmanager

# optional dask import: prefer it but fall back gracefully
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except Exception:
    dd = None
    DASK_AVAILABLE = False

import argparse
import sys
from tqdm import tqdm

# Heuristic file-size threshold to enable Dask automatically (in bytes)
DASK_FILESIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

# Local Code
from .config import *

logger = logging.getLogger(__name__)


class DataAccess:
    def __init__(self, db_path=DB_PATH, auto_create=False):
        """
        Initialize the DataAccess class, connect to SQLite DB, and optionally create tables.

        Args:
            db_path (str): Path to SQLite database file.
            auto_create (bool): If True, create or reinitialize the database automatically.
                                If the file exists and contains tables, user will be prompted for confirmation.
        """
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            print(f"[Info] Database folder '{db_dir}' does not exist. Creating it...")
            os.makedirs(db_dir, exist_ok=True)

        db_missing = not os.path.exists(db_path)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self.conn.commit()

        if auto_create:
            if not db_missing and self.__check_tables_exist():
                # Safe prompt if existing DB has tables
                resp = input(f"[Warning] Database '{db_path}' already exists and has tables. Recreate it? (y/N): ").strip().lower()
                if resp != "y":
                    print("[Info] Keeping existing database structure.")
                else:
                    print(f"[Info]: Reinitializing database at {db_path}")
                    self.create_db(drop=True)
            else:
                print(f"[Info]: Initializing new database at {db_path}")
                self.create_db(drop=True)

        elif db_missing:
            raise FileNotFoundError(
                f"Database '{db_path}' does not exist. "
                "Run with auto_create=True or use --create_db."
            )
        else:
            if not self.__check_tables_exist():
                raise RuntimeError(
                    f"Database '{db_path}' exists but has no tables. "
                    "Run DataAccess(auto_create=True) or use --create_db."
                )

        logger.debug(f"Connected to SQLite: {self.db_path}")

    @contextmanager
    def fast_load(self):
        self.drop_input_indexes()
        self.conn.execute("PRAGMA synchronous=OFF;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-100000;")
        self.conn.execute("PRAGMA foreign_keys=OFF;")
        try:
            yield
        finally:
            self.create_input_indexes()
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
    
    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def check_record(self, query_or_path, params):
        """
        A general function supposed to return a bool value if a searched record (based on params) exists in the snapshot of the DB.
        The operation is determined by the query, that should return 0 or 1.

        Args:
            query_or_path (str): str (describing the query) or .sql file path
            params (tuple): A tuple of size <0 with the input parameters needed to run the query, based on it's placeholders (?) 
        NOTE: Not pivotal for the Mediator's task
        """
        if os.path.isfile(query_or_path):
            with open(query_or_path, 'r') as file:
                query = file.read()
        else:
            query = query_or_path  # assume raw SQL
        result = self.fetch_records(query, params)
        return bool(result)
        
    def get_attr(self, query_or_path, params):
        """
        A general function supposed to return a specific value like unit, date etc.
        The operation is determined by the query, that should have 1 item in the SELECT section.

        Args:
            query_or_path (str): str (describing the query) or .sql file path
            params (tuple): A tuple of size <0 with the input parameters needed to run the query, based on it's placeholders (?)
        NOTE: Not pivotal for the Mediator's task
        """
        if os.path.isfile(query_or_path):
            with open(query_or_path, 'r') as file:
                query = file.read()
        else:
            query = query_or_path  # assume raw SQL
        result = self.fetch_records(query, params)
        return result[0][0] if result else None

    
    def execute_query(self, query_or_path, params=()):
        """
        Executes an INSERT/UPDATE/DELETE query.
        Accepts either a path to a .sql file or a raw SQL string.

        Args:
            query_or_path (str): str (describing the query) or .sql file path
            params (tuple): A tuple of size <0 with the input parameters needed to run the query, based on it's placeholders (?)
        """
        if os.path.isfile(query_or_path):
            with open(query_or_path, 'r') as file:
                query = file.read()
        else:
            query = query_or_path  # assume raw SQL

        self.cursor.execute(query, params)
        self.conn.commit()

    # Helper for batch inserts using a query file path and list of tuples
    def insert_many(self, query_path, rows, batch_size=1000, use_retry=True):
        """
        Bulk-insert rows using `executemany` in large batches.

        Notes:
        • This function DOES NOT commit; the caller is responsible for wrapping the operation in a transaction.
        • Returns a precise count of newly inserted rows using `total_changes` (works with INSERT OR IGNORE).

        Args:
            query_path (str): Path to a .sql INSERT template with ? placeholders.
            rows (Iterable[Tuple]): Row tuples matching the INSERT template.
            batch_size (int): Rows per executemany call.
            use_retry (bool): Allows for retries, but single writer mechanism should avoid the need for those in practice aside from edge cases.

        Returns:
        int: Number of rows inserted (excluding ignored duplicates).
        """
        if not os.path.isfile(query_path):
            raise FileNotFoundError(f"Query file not found: {query_path}")
        with open(query_path, 'r') as f:
            query = f.read()
        
        def _run_batch(cursor, query_sql, batch, retries=5, base_delay=0.1):
            if not use_retry:
                before = cursor.connection.total_changes
                cursor.executemany(query_sql, batch)
                return cursor.connection.total_changes - before

            for attempt in range(retries):
                try:
                    before = cursor.connection.total_changes
                    cursor.executemany(query_sql, batch)
                    return cursor.connection.total_changes - before
                except sqlite3.OperationalError as e:
                    msg = str(e).lower()
                    if ("locked" in msg) or ("busy" in msg):
                        if attempt == retries - 1:
                            raise
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise

        batch = []
        total_inserted = 0
        for r in rows:
            batch.append(r)
            if len(batch) >= batch_size:
                total_inserted += _run_batch(self.cursor, query, batch)
                batch = []
        if batch:
            total_inserted += _run_batch(self.cursor, query, batch)

        return total_inserted

    def fetch_records(self, query_or_path, params=()):
        """
        Executes a SELECT query (GET operations).
        Accepts either a path to a .sql file or a raw SQL string.
        Returns all rows.

        Args:
            query_or_path (str): str (describing the query) or .sql file path
            params (tuple): A tuple of size <0 with the input parameters needed to run the query, based on it's placeholders (?)
        """
        if os.path.isfile(query_or_path):
            with open(query_or_path, 'r') as file:
                query = file.read()
        else:
            query = query_or_path  # assume raw SQL

        return self.cursor.execute(query, params).fetchall()

    def __execute_script(self, script_path):
        """
        Execute a DDL script from a file.
        Used to initialize tables in the DB.

        Args:
            query_or_path (str): str (describing the query) or .sql file path
            params (tuple): A tuple of size <0 with the input parameters needed to run the query, based on it's placeholders (?)
        """
        with open(script_path, 'r') as file:
            script = file.read()
        self.cursor.executescript(script)
        self.conn.commit()

    def __check_tables_exist(self):
        """
        Ensuring DB was initialized.
        """
        result = self.fetch_records(CHECK_TABLE_EXISTS_QUERY, ())
        return bool(result)

    # Drop tables (used by create_db --drop)
    def drop_tables(self):
        """
        Drop the main tables if they exist.
        """
        for t in ('InputPatientData', 'OutputPatientData', 'PatientQAScores'):
            self.cursor.execute(f"DROP TABLE IF EXISTS {t};")
        self.conn.commit()

    # new: create or recreate DB tables
    def create_db(self, drop=False):
        """
        Create DB tables. If drop=True, drop existing tables first.
        """
        if drop:
            print("[Info] Dropping existing tables...")
            self.drop_tables()
        print("[Info] Creating tables from DDL...")
        self.__execute_script(INITIALIZE_TABLES_DDL)
        self.__print_db_info()
    
    # Index creation and deletion helpers to avoid insert overhead
    def drop_input_indexes(self):
        self.cursor.executescript("""
        DROP INDEX IF EXISTS idx_input_patientid;
        DROP INDEX IF EXISTS idx_input_concept;
        DROP INDEX IF EXISTS idx_input_starttime;
        """)
        self.conn.commit()

    def create_input_indexes(self):
        self.cursor.executescript("""
        CREATE INDEX IF NOT EXISTS idx_input_patientid ON InputPatientData (PatientId);
        CREATE INDEX IF NOT EXISTS idx_input_concept ON InputPatientData (ConceptName);
        CREATE INDEX IF NOT EXISTS idx_input_starttime ON InputPatientData (StartDateTime);
        """)
        self.conn.commit()

    def drop_output_indexes(self):
        self.cursor.executescript("""
        DROP INDEX IF EXISTS idx_output_patientid;
        DROP INDEX IF EXISTS idx_output_concept;
        DROP INDEX IF EXISTS idx_output_starttime;
        """)
        self.conn.commit()

    def create_output_indexes(self):
        self.cursor.executescript("""
        CREATE INDEX IF NOT EXISTS idx_output_patientid ON OutputPatientData (PatientId);
        CREATE INDEX IF NOT EXISTS idx_output_concept ON OutputPatientData (ConceptName);
        CREATE INDEX IF NOT EXISTS idx_output_starttime ON OutputPatientData (StartDateTime);
        """)
        self.conn.commit()

    # Table stats helper
    def get_table_stats(self):
        """
        Return row counts and distinct patient counts for main tables.
        """
        stats = {}
        for t in ('InputPatientData', 'OutputPatientData', 'PatientQAScores'):
            try:
                rows = self.fetch_records(f"SELECT COUNT(*) FROM {t};", ())[0][0]
                patients = self.fetch_records(f"SELECT COUNT(DISTINCT PatientId) FROM {t};", ())[0][0]
            except Exception:
                rows, patients = 0, 0
            stats[t] = {'rows': rows, 'n_patients': patients}
        return stats

    # Load CSV to input table with prompt, tqdm, auto dask heuristic
    def load_csv_to_input(self, csv_path, if_exists='append', clear_output_and_qa=False, yes=False, batch_size=2000):
        """
        Load a CSV into InputPatientData with strict validation and fast bulk insert.

        Behavior:
        • Validates each chunk first; if any chunk fails, the entire load aborts and no rows are committed.
        • Disables indexes and (optionally) foreign key checks during the load, then recreates indexes afterwards.
        • Runs the whole operation in a single transaction for performance.
        • Uses pandas (or Dask for large files) to stream chunks.

        Required CSV columns:
        PatientId (int), ConceptName (str), StartDateTime (datetime), EndDateTime (datetime), Value (str)
        Optional: Unit

        Args:
        csv_path (str): Path to input CSV file.
        if_exists (str): 'append' or 'replace'. With 'replace', InputPatientData is cleared before load.
        clear_output_and_qa (bool): If True, clears OutputPatientData and PatientQAScores before load.
        yes (bool): Auto-confirm prompts if tables are non-empty.
        batch_size (int): Number of rows per executemany batch (affects DB roundtrips).

        Returns:
        int: Number of rows actually inserted (excludes INSERT OR IGNORE duplicates).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # prompt if table not empty
        stats = self.get_table_stats()
        in_stats = stats.get('InputPatientData', {'rows':0,'n_patients':0})
        if in_stats['rows'] > 0 and not yes:
            print(f"[Info] InputPatientData is not empty: rows={in_stats['rows']}, patients={in_stats['n_patients']}")
            resp = input("Continue loading? (y/N) ").strip().lower()
            if resp != 'y':
                print("Aborted by user.")
                return 0

        # optional clearing
        if if_exists == 'replace':
            print("[Info] Clearing InputPatientData...")
            self.execute_query("DELETE FROM InputPatientData;", ())
        if clear_output_and_qa:
            print("[Info] Clearing OutputPatientData and PatientQAScores...")
            self.execute_query("DELETE FROM OutputPatientData;", ())
            self.execute_query("DELETE FROM PatientQAScores;", ())

        # decide dask usage
        filesize = os.path.getsize(csv_path)
        use_dask_effective = DASK_AVAILABLE and (filesize >= DASK_FILESIZE_THRESHOLD)

        # map header -> indices using header only
        header_df = pd.read_csv(csv_path, nrows=0)
        cols = list(header_df.columns)
        col_index = {c.lower(): i for i, c in enumerate(cols)}
        def find_index(candidates):
            for name in candidates:
                idx = col_index.get(name.lower())
                if idx is not None:
                    return idx
            return None

        indices = {
            'patient_idx': find_index(['PatientId','patientid','patient_id','Patient Id','patient id']),
            'concept_idx': find_index(['ConceptName','concept','LOINC-NUM','Code','Concept','concept name']),
            'start_idx': find_index(['StartDateTime','Start','StartTime','Timestamp','Time','startdatetime']),
            'end_idx': find_index(['EndDateTime','End','EndTime','enddatetime']),
            'value_idx': find_index(['Value','value','Result','result']),
            'unit_idx': find_index(['Unit','unit','Units']),
            'cols': cols
        }

        # required column check (fail fast)
        required_missing = []
        if indices['patient_idx'] is None:
            required_missing.append('PatientId')
        if indices['concept_idx'] is None:
            required_missing.append('ConceptName')
        if indices['start_idx'] is None:
            required_missing.append('StartDateTime')
        if indices['end_idx'] is None:
            required_missing.append('EndDateTime')
        if indices['value_idx'] is None:
            required_missing.append('Value')
        if required_missing:
            raise ValueError(f"Missing required columns: {', '.join(required_missing)}. Rename CSV headers or preprocess file.")

        # validation helper for a pandas DataFrame chunk/partition (strict: any failure aborts)
        def validate_chunk_strict(df, idxs):
            ln = len(df)
            if ln == 0:
                return
            # PatientId -> all integer-convertible
            p_series = df.iloc[:, idxs['patient_idx']].astype(str).str.strip().replace({'': None})
            p_numeric = pd.to_numeric(p_series, errors='coerce')
            if p_numeric.isna().any():
                n_bad = int(p_numeric.isna().sum())
                raise ValueError(f"PatientId validation failed: {n_bad}/{ln} rows are not integer-convertible.")
            # ConceptName non-empty
            c_series = df.iloc[:, idxs['concept_idx']].astype(str).str.strip()
            if (c_series == "").any() or c_series.isna().any():
                raise ValueError("ConceptName validation failed: empty values found.")
            # Value required non-empty
            v_series = df.iloc[:, idxs['value_idx']].astype(str).str.strip().replace({'': None})
            if v_series.isna().any():
                raise ValueError("Value validation failed: empty values found.")
            # Start/End datetimes parseable
            s_series = df.iloc[:, idxs['start_idx']]
            e_series = df.iloc[:, idxs['end_idx']]
            parsed_s = pd.to_datetime(s_series, errors='coerce')
            parsed_e = pd.to_datetime(e_series, errors='coerce')
            n_bad_s = int(parsed_s.isna().sum())
            n_bad_e = int(parsed_e.isna().sum())
            if n_bad_s > 0 or n_bad_e > 0:
                raise ValueError(f"Datetime validation failed: {n_bad_s}/{ln} unparsable StartDateTime, {n_bad_e}/{ln} unparsable EndDateTime.")

        # normalize chunk in-place for insertion
        def normalize_chunk_for_insert(df, idxs):
            # coerce PatientId -> int (and ensure no NaN)
            df.iloc[:, idxs['patient_idx']] = pd.to_numeric(df.iloc[:, idxs['patient_idx']], errors='coerce').astype('Int64').astype(int)
            # parse and format datetimes
            df.iloc[:, idxs['start_idx']] = pd.to_datetime(df.iloc[:, idxs['start_idx']], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            df.iloc[:, idxs['end_idx']] = pd.to_datetime(df.iloc[:, idxs['end_idx']], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            # ensure Value is string (non-null)
            df.iloc[:, idxs['value_idx']] = df.iloc[:, idxs['value_idx']].astype(str)
            return df

        # generator that yields tuples matching INSERT_PATIENT_QUERY
        def rows_from_df_for_insert(df, idxs):
            p_idx = idxs['patient_idx']; c_idx = idxs['concept_idx']; s_idx = idxs['start_idx']; e_idx = idxs['end_idx']; v_idx = idxs['value_idx']; u_idx = idxs['unit_idx']
            for tup in df.itertuples(index=False, name=None):
                def at(i):
                    if i is None:
                        return None
                    try:
                        v = tup[i]
                        if pd.isna(v):
                            return None
                        return v
                    except Exception:
                        return None
                yield (
                    int(at(p_idx)) if at(p_idx) is not None else None,
                    str(at(c_idx)) if at(c_idx) is not None else None,
                    str(at(s_idx)) if at(s_idx) is not None else None,
                    str(at(e_idx)) if at(e_idx) is not None else None,
                    str(at(v_idx)) if at(v_idx) is not None else None,
                    None if u_idx is None else (str(at(u_idx)) if at(u_idx) is not None else None)
                )

        # single-pass transactional processing
        total_inserted = 0
        
        try:
            with self.fast_load():
                self.conn.execute("BEGIN")
                if use_dask_effective:
                    print("[Info] Large file detected and Dask available — validating+inserting partitions...")
                    ddf = dd.read_csv(csv_path, assume_missing=True, blocksize="64MB")
                    delayed_parts = ddf.to_delayed()
                    with tqdm(total=len(delayed_parts), desc="dask partitions") as pbar:
                        for d in delayed_parts:
                            part_df = d.compute()
                            validate_chunk_strict(part_df, indices)
                            part_df = normalize_chunk_for_insert(part_df, indices)
                            rows_iter = rows_from_df_for_insert(part_df, indices)
                            inserted = self.insert_many(INSERT_PATIENT_QUERY, rows_iter, batch_size=batch_size)
                            total_inserted += inserted
                            pbar.update(1)
                else:
                    print("[Info] Validating+inserting CSV in chunks (pandas)...")
                    chunksize = max(batch_size * 10, 100_000)
                    reader = pd.read_csv(csv_path, dtype=str, chunksize=chunksize)
                    with tqdm(unit="chunk", desc="CSV chunks") as pbar:
                        for chunk in reader:
                            validate_chunk_strict(chunk, indices)
                            chunk = normalize_chunk_for_insert(chunk, indices)
                            rows_iter = rows_from_df_for_insert(chunk, indices)
                            inserted = self.insert_many(INSERT_PATIENT_QUERY, rows_iter, batch_size=batch_size)
                            total_inserted += inserted
                            pbar.update(1)
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            print("[Error] Load aborted; no data was committed.")
            raise


        print(f"[Info] Finished loading. Inserted {total_inserted} rows.")
        self.__print_db_info()
        return total_inserted

    def __print_db_info(self):
        """
        Printing DB information, including the total number of tables created
        and the number of rows in each table.
        """
        print("[Info]: DB initiated successfully!")

        tables = self.fetch_records("SELECT name FROM sqlite_master WHERE type='table';", ())

        print(f"[Info]: Total tables created: {len(tables)}")

        for (table_name,) in tables:
            count = self.fetch_records(f"SELECT COUNT(*) FROM {table_name};", ())[0][0]
            print(f"[Info]: Table '{table_name}' - Rows: {count}")

# CLI entrypoint
def _cli_main(argv):
    parser = argparse.ArgumentParser(prog="dataaccess", description="DB operations for Mediator")
    parser.add_argument("--create_db", action="store_true", help="Create or recreate the database")
    parser.add_argument("--load_csv", metavar="CSV_PATH", help="Load CSV into InputPatientData")
    parser.add_argument("--replace-input", action="store_true", help="With --load_csv: clear InputPatientData before load")
    parser.add_argument("--clear-output-qa", action="store_true", help="With --load_csv: clear OutputPatientData and PatientQAScores before load")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm prompts")
    args = parser.parse_args(argv)

    da = DataAccess(auto_create=args.create_db)

    if args.load_csv:
        da.load_csv_to_input(
            args.load_csv,
            if_exists='replace' if args.replace_input else 'append',
            clear_output_and_qa=args.clear_output_qa,
            yes=args.yes
        )

if __name__ == "__main__":
    _cli_main(sys.argv[1:])


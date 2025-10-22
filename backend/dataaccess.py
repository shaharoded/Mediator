import sqlite3
import os
import pandas as pd

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
from backend.config import *

class DataAccess:
    def __init__(self, db_path=DB_PATH):
        '''
        Initialize the DataAccess class, connect to SQLite DB and ensure required tables exist.
        No automatic data import on init.
        '''
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        print(f"[DEBUG] Connected to SQLite: {self.db_path}")

        # Ensure tables exist (create if missing)
        if not self.__check_tables_exist():
            print('[Info]: Creating DB tables...')
            self.__execute_script(INITIATE_TABLES_DDL)
        self.__print_db_info()

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
    def insert_many(self, query_path, rows, batch_size=1000):
        """
        Insert many rows using executemany in batches.
        query_path: path to .sql insert template (with ? placeholders)
        rows: iterable of tuples
        """
        if not os.path.isfile(query_path):
            raise FileNotFoundError(f"Query file not found: {query_path}")
        with open(query_path, 'r') as f:
            query = f.read()

        batch = []
        count = 0
        for r in rows:
            batch.append(r)
            if len(batch) >= batch_size:
                self.cursor.executemany(query, batch)
                self.conn.commit()
                count += len(batch)
                batch = []
        if batch:
            self.cursor.executemany(query, batch)
            self.conn.commit()
            count += len(batch)
        return count

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

    # new: drop tables (used by create_db --drop)
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
        self.__execute_script(INITIATE_TABLES_DDL)
        self.__print_db_info()

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
        Load CSV into InputPatientData. Strict validation-first semantics preserved — if any row fails validation,
        no rows are committed. Implementation does single-pass over data inside a DB transaction to avoid double-read.
        Required fields: PatientId (integer), ConceptName, StartDateTime (parseable), EndDateTime (parseable), Value.
        Unit is optional.

        Args:
            csv_path (str): Path to input CSV file
            if_exists (str): Behavior when the table already contains data. One of 'append', 'replace'.
            clear_output_and_qa (bool): Whether to clear OutputPatientData and PatientQAScores before loading.
            yes (bool): Whether to auto-confirm prompts.
            batch_size (int): Number of rows to process in each batch.
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
            parsed_s = pd.to_datetime(s_series, errors='coerce', infer_datetime_format=True)
            parsed_e = pd.to_datetime(e_series, errors='coerce', infer_datetime_format=True)
            n_bad_s = int(parsed_s.isna().sum())
            n_bad_e = int(parsed_e.isna().sum())
            if n_bad_s > 0 or n_bad_e > 0:
                raise ValueError(f"Datetime validation failed: {n_bad_s}/{ln} unparsable StartDateTime, {n_bad_e}/{ln} unparsable EndDateTime.")

        # normalize chunk in-place for insertion
        def normalize_chunk_for_insert(df, idxs):
            # coerce PatientId -> int (and ensure no NaN)
            df.iloc[:, idxs['patient_idx']] = pd.to_numeric(df.iloc[:, idxs['patient_idx']], errors='coerce').astype('Int64').astype(int)
            # parse and format datetimes
            df.iloc[:, idxs['start_idx']] = pd.to_datetime(df.iloc[:, idxs['start_idx']], errors='coerce', infer_datetime_format=True).dt.strftime('%Y-%m-%d %H:%M:%S')
            df.iloc[:, idxs['end_idx']] = pd.to_datetime(df.iloc[:, idxs['end_idx']], errors='coerce', infer_datetime_format=True).dt.strftime('%Y-%m-%d %H:%M:%S')
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
                chunksize = max(batch_size * 10, 10000)
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
    parser.add_argument("--create_db", action="store_true", help="Create DB tables")
    parser.add_argument("--drop", action="store_true", help="With --create_db: drop existing tables first")
    parser.add_argument("--load_csv", metavar="CSV_PATH", help="Load CSV into InputPatientData")
    parser.add_argument("--replace-input", action="store_true", help="With --load_csv: clear InputPatientData before load")
    parser.add_argument("--clear-output-qa", action="store_true", help="With --load_csv: clear OutputPatientData and PatientQAScores before load")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm prompts")
    args = parser.parse_args(argv)

    da = DataAccess()

    if args.create_db:
        da.create_db(drop=args.drop)

    if args.load_csv:
        da.load_csv_to_input(args.load_csv,
                             if_exists='replace' if args.replace_input else 'append',
                             clear_output_and_qa=args.clear_output_qa,
                             yes=args.yes)

if __name__ == "__main__":
    _cli_main(sys.argv[1:])


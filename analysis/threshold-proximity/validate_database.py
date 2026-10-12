"""Compare the entire CSV export with the configured Mediator input database.

Database connection is read-only, including its current WAL contents. No writes,
schema changes, or database initialization use the Mediator DataAccess class.
"""
import csv
import itertools
import json
from pathlib import Path
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import sqlite3
import time

ROOT = RESULTS_ROOT / 'threshold-proximity'
ROOT.mkdir(parents=True, exist_ok=True)
database=MEDIATOR_ROOT/'backend/data/mediator.db'
source=PERSONAL/'mimic-dataset/output/concept_events.csv'
columns=['PatientId','ConceptName','StartDateTime','EndDateTime','Value']
conn=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
conn.execute('pragma query_only=on')
conn.execute('begin')
cursor=conn.execute('select '+','.join(columns)+' from InputPatientData order by RowId')
matched=mismatched=csv_rows=db_rows=0
differences={name:0 for name in columns}
start=time.monotonic()
with source.open(newline='',encoding='utf-8-sig') as f:
    reader=csv.reader(f)
    header=next(reader)
    indices=[header.index(name) for name in columns]
    for i,(csv_row,db_row) in enumerate(itertools.zip_longest(reader,cursor),1):
        if csv_row is not None: csv_rows+=1
        if db_row is not None: db_rows+=1
        left=tuple(csv_row[j] for j in indices) if csv_row is not None else None
        right=tuple(str(value) if value is not None else '' for value in db_row) if db_row is not None else None
        if left==right:
            matched+=1
        else:
            mismatched+=1
            if left is not None and right is not None:
                for j,name in enumerate(columns):
                    differences[name]+=int(left[j]!=right[j])
        if i%2_000_000==0:
            print(f'Compared {i:,} rows; mismatches {mismatched:,}',flush=True)
conn.rollback()
conn.close()
result=dict(database=database.relative_to(MEDIATOR_ROOT).as_posix(),database_engine='SQLite',table='InputPatientData',
            csv=source.relative_to(PERSONAL).as_posix(),columns=columns,comparison='All rows in CSV order versus database RowId order; exact string field equality',
            csv_rows=csv_rows,database_rows=db_rows,matching_rows=matched,mismatching_rows=mismatched,
            field_differences=differences,exact_match=(mismatched==0),elapsed_seconds=time.monotonic()-start,
            database_connection='mode=ro and query_only=on; read transaction',
            csv_bytes=source.stat().st_size,database_bytes=database.stat().st_size)
(ROOT/'database_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['csv_rows','database_rows','matching_rows','mismatching_rows','exact_match']},indent=2),flush=True)

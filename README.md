# Mediator — Temporal Abstraction

## Purpose
The Mediator converts time-stamped clinical input into interval-based abstractions for research, development and predictive models. It implements a compact Knowledge-Based Temporal Abstraction (KBTA) style pipeline locally.

## Implemented abstractions
- **Events** — point-in-time occurrences (e.g., medication administration, dysglycemia).
  - Can be derived from **multiple raw-concepts** (bridging data gaps).
  - Support flexible constraints: `equal`, `min`, `max`, or `min + max` (range).
- **Contexts** — background facts that affect abstraction logic. Similar to `Events` but might be time bounded.
- **States** — symbolic intervals derived from numeric concepts by discretization or from nominal concepts by equality (e.g., "high glucose").
- **Trends** — gradients over time (e.g., increasing, decreasing).
- **Patterns** — ordered compositions of events/intervals computed from abstractions.
- **QA-patterns** — post-analysis scoring of detected or missing patterns.

## Repository layout
```
Mediator/
├── backend/
│   ├── data/
│   │   ├── generate_synthetic_data.ipynb   # Notebook to generate templated data for tests
│   │   ├── mediator.db                     # SQLite DB file (created by CLI)
│   │   └── synthetic_input_data.csv        # Example input CSV, output from notebook
│   ├── queries/                            # SQL templates (DDL, DML, SELECT)
│   │   ├── create_tables.sql
│   │   ├── insert_raw_concept.sql
│   │   ├── insert_abstracted_concept.sql
│   │   ├── insert_qa_score.sql
│   │   ├── get_input_by_patient.sql
│   │   ├── get_input_by_patient_and_concepts.sql
│   │   ├── get_output_by_patient.sql
│   │   ├── get_output_by_patient_and_concepts.sql
│   │   ├── get_qa_scores.sql
│   │   └── check_table_exists.sql
│   ├── config.py                           # Paths to DB and SQL templates
│   ├── dataaccess.py                       # Main DB access and CLI
├── core/
│   ├── config.py                           # Paths to TAK files for engine
│   ├── utils.py                            # Utility Functions
│   ├── tak.py                              # Defines the TAK object and parser
│   ├── mediator.py                         # Abstraction engine
│   └── TAKs/                               # TAK/rule files (for abstraction logic)
├── images/
├── unittests/
├── README.md                               # This file
└── requirements.txt
```

## Database summary
- **Tables:**
  - `InputPatientData(RowId, PatientId INTEGER, ConceptName TEXT, StartDateTime TEXT, EndDateTime TEXT, Value TEXT, Unit TEXT)`
  - `OutputPatientData(RowId, PatientId INTEGER, ConceptName TEXT, StartDateTime TEXT, EndDateTime TEXT, Value TEXT, AbstractionType TEXT)`
  - `PatientQAScores(RowId, PatientId INTEGER, PatternName TEXT, StartDateTime TEXT, EndDateTime TEXT, Score REAL)`
- **Constraints / indexes:**
  - `UNIQUE(PatientId, ConceptName, StartDateTime)` prevents duplicates.
  - Indexes on PatientId, ConceptName, StartDateTime for efficient queries.
  - INSERTs use `INSERT OR IGNORE` to respect uniqueness.

## CSV input requirements (strict)
- **Required columns** (headers are matched heuristically; canonical names recommended):
  - `PatientId` — integers only (all rows validated).
  - `ConceptName` — non-empty.
  - `StartDateTime` — parseable by pandas.to_datetime (all rows validated).
  - `EndDateTime` — parseable by pandas.to_datetime (all rows validated).
  - `Value` — non-empty.
- **Optional:**
  - `Unit` — optional text.
- Dates are normalized to `YYYY-MM-DD HH:MM:SS` on insert.
- **Validation policy:** loader validates every row; any failure aborts the entire load (transaction rollback).

## CLI
- **Create DB tables:**
  ```
  python backend/dataaccess.py --create_db
  ```
  - Drop existing and recreate:
  ```
  python backend/dataaccess.py --create_db --drop
  ```
- **Load CSV into InputPatientData:**
  - Interactive (prompts if table not empty):
    ```
    python backend/dataaccess.py --load_csv data/my_input.csv
    ```
  - Replace input, clear outputs, auto-confirm:
    ```
    python backend/dataaccess.py --load_csv data/my_input.csv --replace-input --clear-output-qa --yes
    ```
- Loader auto-selects Dask for large files (>=100 MB) if available; otherwise uses pandas chunking.
- Loading is transactional: validations + inserts per chunk; any validation error rolls back all changes.

## Programmatic usage
```python
from backend.dataaccess import DataAccess
da = DataAccess()
da.create_db(drop=False)
da.load_csv_to_input("data/my_input.csv", if_exists='append', clear_output_and_qa=False, yes=True)
rows = da.fetch_records(GET_INPUT_BY_PATIENT, (patient_id,))
```

## SQL templates
- Located at `backend/queries/*.sql` and referenced from `backend/backend_config.py`.
- Use provided templates for common operations (get by patient, get by patient+concepts, insert raw/abstracted/QA).

## Tips
- Prefer canonical headers in CSV (`PatientId`, `ConceptName`, `StartDateTime`, `EndDateTime`, `Value`, `Unit`).
- Back up `data/mediator.db` before destructive operations (e.g., `--create_db --drop`).
- If you want lenient ingestion, add a preprocessing step to clean the CSV before loading.

## Testing

Self contained testing modules were created under `unittests\`. You can run:

```bash
python -m pytest -q
```

OR one-by-one

```bash
python -m pytest unittests/test_raw_concept.py -v
python -m pytest unittests/test_event.py -v
python -m pytest unittests/test_state.py -v
python -m pytest unittests/test_trend.py -v
python -m pytest unittests/test_context.py -v
```

## Git commit tips

To initialize a git repository and publish this project on GitHub:

1. **Initialize the repository (if not already):**
   ```sh
   git init
   ```

2. **Add all files:**
   ```sh
   git add .
   ```

3. **Commit your changes:**
   ```sh
   git commit -m "message"
   ```

4. **Create a new repository on GitHub** (via the GitHub web UI).

5. **Add the remote and push:**
   ```sh
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

6. **After publishing:**
   - Use `git add`, `git commit`, and `git push` to update your repository as you develop.
   - Write clear commit messages describing your changes.

## Citation
This code is a research tool. Track provenance and cite relevant KBTA literature when used in publications.
# Mediator — Knowledge-Based Temporal Abstraction

<p align="center">
  <img src="images/mediator_architecture.png" alt="Mediator Architecture" width="800"/>
</p>

## Overview

The **Mediator** is a Python-based implementation of Knowledge-Based Temporal Abstraction (KBTA) that converts time-stamped clinical data into interval-based symbolic abstractions for research and predictive modeling.

**Key Features:**
- ✅ **Hierarchical abstractions** — Raw Concepts → Events → States → Trends → Contexts → Patterns
- ✅ **XSD schema validation** — Structural validation for all TAK definitions
- ✅ **Production-ready** — SQLite backend, async processing, comprehensive tests
- ✅ **Extensible** — XML-based TAK definitions (no code changes needed)

**Theoretical Foundation:**

This implementation is based on the KBTA framework:

1. **Shahar, Y., & Musen, M. A. (1996).** "Knowledge-based temporal abstraction in clinical domains." *Artificial Intelligence in Medicine*, 8(3), 267-298.
   - DOI: [10.1016/0933-3657(95)00036-4](https://doi.org/10.1016/0933-3657(95)00036-4)

2. **Shalom, E., Goldstein, A., Weiss, R., Selivanova, M., Cohen, N. M., & Shahar, Y. (2024).** "Implementation and evaluation of a system for assessment of the quality of long-term management of patients at a geriatric hospital." *Journal of Biomedical Informatics*, 156, 104686.
   - DOI: [10.1016/j.jbi.2024.104686](https://doi.org/10.1016/j.jbi.2024.104686)

---

## Repository Structure

```
Mediator/
├── backend/                                # Database layer
│   ├── data/
│   │   ├── generate_synthetic_data.ipynb   # Synthetic data generator
│   │   ├── mediator.db                     # SQLite database (auto-created)
│   │   └── synthetic_input_data.csv        # Example input CSV
│   ├── queries/                            # SQL templates (DDL, DML, SELECT)
│   ├── config.py                           # Database paths
│   └── dataaccess.py                       # Database access + CLI
├── core/                                   # TAK engine
│   ├── knowledge-base/                     # TAK definitions (XML)
│   │   ├── raw-concepts/                   # Single/multi-attribute concepts
│   │   ├── events/                         # Point-in-time events
│   │   ├── states/                         # Interval-based states
│   │   ├── trends/                         # Slope-based trends
│   │   ├── contexts/                       # Background contexts
│   │   ├── patterns/                       # Temporal patterns (TODO)
│   │   ├── global_clippers.json            # Global START/END clippers
│   │   ├── tak_schema.xsd                  # XSD validation schema
│   │   └── TAK_README.json                 # TAK documentation + instructions
│   ├── tak/                                # TAK implementation
│   │   ├── tak.py                          # Base classes + repository
│   │   ├── raw_concept.py                  # RawConcept TAK
│   │   ├── event.py                        # Event TAK
│   │   ├── state.py                        # State TAK
│   │   ├── trend.py                        # Trend TAK
│   │   ├── context.py                      # Context TAK
│   │   ├── pattern.py                      # Pattern TAK (TODO)
│   │   └── utils.py                        # Shared utilities
│   ├── config.py                           # TAK paths
│   └── mediator.py                         # Orchestration engine + CLI
├── images/                                 # Documentation assets
├── unittests/                              # Comprehensive test suite
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Mediator.git
cd Mediator

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Database

```bash
# Create database tables
python -m backend.dataaccess --create_db

# Load example data
python -m backend.dataaccess --load_csv backend/data/synthetic_input_data.csv
```

### 3. Run Pipeline

```bash
# Process all patients
python -m core.mediator

# Process specific patients
python -m core.mediator --patients 1000,1001,1002

# Debug mode
python -m core.mediator --log-level DEBUG --patients 1000
```

---

## Database Schema

### CSV Input Requirements

**Required columns:**
- `PatientId` — Integer patient identifier
- `ConceptName` — Concept/measurement name (matches TAK names)
- `StartDateTime` — Timestamp (YYYY-MM-DD HH:MM:SS)
- `EndDateTime` — Timestamp
- `Value` — Measurement value (numeric or string)

**Optional columns:**
- `Unit` — Measurement unit

**Example CSV:**
```csv
PatientId,ConceptName,StartDateTime,EndDateTime,Value,Unit
1000,GLUCOSE_LAB_MEASURE,2024-01-01 08:00:00,2024-01-01 08:00:00,120,mg/dL
1000,BASAL_DOSAGE,2024-01-01 21:00:00,2024-01-01 21:00:00,15,U
1000,BASAL_ROUTE,2024-01-01 21:00:00,2024-01-01 21:00:00,SubCutaneous,
```

---

## CLI Reference

### Database Operations

```bash
# Create/recreate database
python -m backend.dataaccess --create_db [--drop]

# Load CSV
python -m backend.dataaccess --load_csv data/input.csv

# Replace existing data
python -m backend.dataaccess --load_csv data/input.csv \
    --replace-input --clear-output-qa --yes
```

**Features:**
- ✅ Auto-detects large files and uses Dask if available (≥100 MB)
- ✅ Transactional loading (all-or-nothing validation)
- ✅ Interactive prompts (overridable with `--yes`)

---

### Core (Mediator Pipeline)

```bash
# Process all patients
python -m core.mediator

# Process subset
python -m core.mediator --patients 1000,1001

# Custom concurrency
python -m core.mediator --max-concurrent 8

# Custom KB and DB paths
python -m core.mediator --kb core/knowledge-base --db data/mediator.db

# Debug logging
python -m core.mediator --log-level DEBUG --patients 101,102,103
```

**Workflow:**
1. **Build TAK repository** — Load and validate all TAK definitions from `knowledge-base/`
2. **Process patients** — For each patient:
   - Extract input data from `InputPatientData`
   - Apply TAKs in dependency order (Raw Concepts → Events → States → Trends → Contexts)
   - Write abstractions to `OutputPatientData`
3. **Report stats** — Print timing and output row counts per patient

---

## Programmatic Usage

### Database Access

```python
from backend.dataaccess import DataAccess

da = DataAccess()

# Create tables
da.create_db(drop=False)
da.load_csv_to_input("data/input.csv", if_exists='append')

# Query
rows = da.fetch_records(
    "SELECT * FROM InputPatientData WHERE PatientId = ?",
    (1000,)
)
```

### TAK Processing

```python
from pathlib import Path
from core.mediator import Mediator

mediator = Mediator(Path("core/knowledge-base"))
repo = mediator.build_repository()

# Process patients
stats = mediator.run(max_concurrent=4, patient_subset=[1000, 1001])
print(stats)
```

---

## TAK Documentation

For detailed information about TAK families, XML schema, validation rules, and examples:

📖 **See:** [`core/knowledge-base/TAK_README.md`](core/knowledge-base/TAK_README.md)

**Quick TAK Reference:**
- **Raw Concepts** — Bridge InputPatientData → pipeline (multi-attr tuples, numeric ranges, nominal values)
- **Events** — Point-in-time occurrences (multi-source, flexible constraints)
- **States** — Interval-based symbolic states (discretization + merging)
- **Trends** — Slope-based trends (Increasing/Decreasing/Steady)
- **Contexts** — Background facts (windowing + clipping)

---

## Testing

```bash
# Run all tests
python -m pytest unittests/ -v

# Run specific test modules
python -m pytest unittests/test_raw_concept.py -v
python -m pytest unittests/test_event.py -v
python -m pytest unittests/test_state.py -v
python -m pytest unittests/test_trend.py -v
python -m pytest unittests/test_context.py -v
python -m pytest unittests/test_mediator.py -v
```

**Coverage:** 53 tests covering all TAK families + end-to-end pipeline.

---

## Common Issues

### Database Not Found
**Error:** `Database file 'mediator.db' does not exist`

**Fix:**
```bash
python -m backend.dataaccess --create_db
```

### CSV Validation Failure
**Error:** `Missing required columns: PatientId`

**Fix:** Rename CSV headers to match canonical names (`PatientId`, `ConceptName`, etc.)

### TAK Validation Error
**Error:** `Element 'tuple-order': This element is not expected`

**Fix:** Check XML element order matches schema (see [`TAK_README.md`](core/knowledge-base/TAK_README.md#critical-element-order-rules))

---

## Citation

```bibtex
@article{shahar1996knowledge,
  title={Knowledge-based temporal abstraction in clinical domains},
  author={Shahar, Yuval and Musen, Mark A},
  journal={Artificial Intelligence in Medicine},
  volume={8},
  number={3},
  pages={267--298},
  year={1996},
  publisher={Elsevier}
}

@article{shalom2024implementation,
  title={Implementation and evaluation of a system for assessment of the quality of long-term management of patients at a geriatric hospital},
  author={Shalom, Erez and Goldstein, Avraham and Weiss, Robert and Selivanova, Marina and Cohen, Nir Menachemi and Shahar, Yuval},
  journal={Journal of Biomedical Informatics},
  volume={156},
  pages={104686},
  year={2024},
  publisher={Elsevier}
}
```
---

**Maintained by:** Shahar Oded
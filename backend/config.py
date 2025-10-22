import os

# Get project root (one level up from backend/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# DB
DB_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'mediator.db')

# DDL Queries
INITIATE_TABLES_DDL = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'create_tables.sql')

# DML Queries
INSERT_PATIENT_QUERY = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'insert_raw_concept.sql')
INSERT_ABSTRACTED_MEASUREMENT_QUERY = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'insert_abstracted_concept.sql')
INSERT_QA_SCORE_QUERY = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'insert_qa_score.sql')

# SQL Queries
CHECK_TABLE_EXISTS_QUERY = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'check_table_exists.sql')
GET_DATA_BY_PATIENT = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'get_data_by_patient.sql')
GET_DATA_BY_PATIENT_CONCEPTS = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'get_data_by_patient_and_concepts.sql')
GET_QA_SCORES = os.path.join(PROJECT_ROOT, 'backend', 'queries', 'get_qa_scores.sql')
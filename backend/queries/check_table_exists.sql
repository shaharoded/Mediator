-- Purpose: Check whether the main InputPatientData table exists (used to detect fresh DB).
SELECT name FROM sqlite_master
WHERE type='table' AND name='InputPatientData';
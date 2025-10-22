-- Purpose: Insert a single abstraction into OutputPatientData.
-- Notes: Uses INSERT OR IGNORE so duplicates (PatientId, ConceptName, StartDateTime) are not inserted.
INSERT OR IGNORE INTO OutputPatientData (
    PatientId,
    ConceptName,
    StartDateTime,
    EndDateTime,
    Value,
    AbstractionType
) VALUES (?, ?, ?, ?, ?, ?);

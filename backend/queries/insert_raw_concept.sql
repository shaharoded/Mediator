-- Purpose: Insert a single raw input concept record into InputPatientData.
-- Notes: Uses INSERT OR IGNORE so duplicates (PatientId, ConceptName, StartDateTime) are not inserted.
INSERT OR IGNORE INTO InputPatientData (
    PatientId,
    ConceptName,
    StartDateTime,
    EndDateTime,
    Value,
    Unit
) VALUES (?, ?, ?, ?, ?, ?);

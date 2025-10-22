-- Purpose: Return all input/raw records for a given PatientId (ordered by StartDateTime).
-- Params: PatientId
SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value, Unit
FROM InputPatientData
WHERE PatientId = ?
ORDER BY StartDateTime ASC;

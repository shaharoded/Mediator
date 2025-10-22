-- Purpose: Return all output/abstraction records for a given PatientId (ordered by StartDateTime).
-- Params: PatientId
SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
FROM OutputPatientData
WHERE PatientId = ?
ORDER BY StartDateTime ASC;
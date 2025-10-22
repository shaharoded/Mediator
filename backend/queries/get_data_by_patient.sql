-- Purpose: Return all records for a given PatientId (ordered by StartDateTime).
-- Params: TableName, PatientId
SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value
FROM ?
WHERE PatientId = ?
ORDER BY StartDateTime ASC;

-- Purpose: Return input records for a PatientId filtered to a dynamic list of ConceptName values.
-- Template: Caller must replace {CONCEPT_PLACEHOLDERS} with "?, ?, ..." (N placeholders)
-- Params order: TableName, PatientId, <concept1>, <concept2>, ... <conceptN>
SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value
FROM ?
WHERE PatientId = ?
  AND ConceptName IN ({CONCEPT_PLACEHOLDERS})
ORDER BY ConceptName ASC, StartDateTime ASC;

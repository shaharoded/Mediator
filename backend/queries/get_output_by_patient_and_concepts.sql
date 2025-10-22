-- Purpose: Return output/abstraction records for a PatientId filtered by a dynamic set of ConceptName values.
-- Template: Caller must replace {CONCEPT_PLACEHOLDERS} with "?, ?, ..." (N placeholders)
-- Params order: PatientId, <concept1>, <concept2>, ... <conceptN>
SELECT PatientId, ConceptName, StartDateTime, EndDateTime, Value, AbstractionType
FROM OutputPatientData
WHERE PatientId = ?
  AND ConceptName IN ({CONCEPT_PLACEHOLDERS})
ORDER BY ConceptName ASC, StartDateTime ASC;

-- Purpose: Retrieve QA scores for all patients (ordered for easier review/processing).
SELECT PatientId, PatternName, StartDateTime, EndDateTime, Score, Details
FROM PatientQAScores
ORDER BY PatientId ASC, StartDateTime ASC, PatternName ASC;

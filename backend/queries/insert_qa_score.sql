-- Purpose: Insert one QA score record into PatientQAScores.
-- Notes: Uses INSERT OR IGNORE to avoid duplicate QA rows for same (PatientId, PatternName, StartDateTime).
INSERT OR IGNORE INTO PatientQAScores (
    PatientId,
    PatternName,
    StartDateTime,
    EndDateTime,
    Score
) VALUES (?, ?, ?, ?, ?);

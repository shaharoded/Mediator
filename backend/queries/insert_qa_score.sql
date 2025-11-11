-- Insert Pattern compliance scores into PatientQAScores table
-- Uses INSERT OR IGNORE to skip duplicates (unique constraint on PatientId, PatternName, StartDateTime)

INSERT OR IGNORE INTO PatientQAScores (
    PatientId,
    PatternName,
    StartDateTime,
    EndDateTime,
    ComplianceType,
    ComplianceScore
) VALUES (?, ?, ?, ?, ?, ?);

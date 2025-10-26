"""
TO-DO:
- Add global clippers (ADMISSION, RELEASE, DEATH) that will clip all states/trends/patterns/contexts
"""

from datetime import timedelta
import pandas as pd

# Local Code
from backend.dataaccess import DataAccess
from core.tak.tak import *
from core.tak.utils import *
from backend.config import * 
from core.config import *


class Mediator:
    """
    The Mediator engine orchestrates the temporal abstraction process.
    It:
    - Loads abstraction rules (TAK files)
    - Retrieves raw measurement data for a patient
    - Applies all applicable abstraction rules
    - Merges overlapping intervals with the same label
    - Returns both abstracted intervals and untouched raw measurements

    Attributes:
        parser (TAKParser): Loads TAK rules from XML files.
        tak_rules (list of TAKRule): All loaded rules.
        db (DataAccess): Database interface for patient and measurement retrieval.
    """
    def __init__(self, tak_folder=TAK_FOLDER):
        self.parser = TAKParser(tak_folder)
        self.tak_rules = self.parser.load_all_taks()
        self.db = DataAccess()
    

    def _get_patient_records(self, patient_id, table="InputPatientData", concepts=[]):
        """
        Retrieves patient measurement records and patient-level attributes for abstraction.
        
        Args:
            patient_id (int): The ID of the patient.
            table (str): The database table to query from. Choose from 'InputPatientData' or 'OutputPatientData'.
            concepts (list): The list of concepts to retrieve.

        Returns:
            pd.DataFrame: DataFrame of patient measurement records.
        """
        # Construct query
        if concepts:
            concept_placeholders = ",".join(["?"]*len(concepts))
            params = [patient_id, table] + concepts

            with open(GET_DATA_BY_PATIENT_CONCEPTS, 'r') as f:
                base_query = f.read()
            base_query = base_query.replace("{CONCEPT_PLACEHOLDERS}", concept_placeholders)
        else:
            params = [table, patient_id]

            with open(GET_DATA_BY_PATIENT, 'r') as f:
                base_query = f.read()

        # Get records as DataFrame
        patient_records = pd.DataFrame(self.db.fetch_records(base_query, params))

        return patient_records
    

    def _merge_intervals(self, df):
        """
        Merge/tidy intervals for a single patient's records that already include both abstracted
        ('discretisized') and untouched ('original') rows.

        Rules within each (LOINC-Code, ConceptName) group:
        - If same Value and intervals touch/overlap -> merge by extending EndDateTime.
        - If different Value and the next starts before current ends -> clip current EndDateTime to next StartDateTime.

        Args:
            df (pd.DataFrame): A single patient's records. Columns required:
                ['PatientId', 'LOINC-Code', 'ConceptName', 'Value', 'StartDateTime', 'EndDateTime', 'Source'].

        Returns:
            pd.DataFrame: Cleaned intervals (no overlaps inside a group), same schema plus preserved/unioned Source.
        """
        # Empty in, empty out
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "PatientId", "LOINC-Code", "ConceptName", "Value",
                "StartDateTime", "EndDateTime", "Source"
            ])

        # Validate and normalize inputs
        required = {"PatientId", "LOINC-Code", "ConceptName", "Value", "StartDateTime", "EndDateTime", "Source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for merge: {missing}")

        df = df.copy()
        df["StartDateTime"] = pd.to_datetime(df["StartDateTime"], errors="coerce")
        df["EndDateTime"]   = pd.to_datetime(df["EndDateTime"],   errors="coerce")

        # Deterministic ordering inside each group
        df = df.sort_values(by=["LOINC-Code", "ConceptName", "StartDateTime", "Value"])

        merged_rows = []

        # Merge within (LOINC, ConceptName)
        for _, g in df.groupby(["LOINC-Code","ConceptName"], sort=False):
            current = None

            for _, row in g.iterrows():
                row = row.copy()

                if current is None:
                    current = row
                    continue

                same_value = (row["Value"] == current["Value"])
                overlap_or_touching = row["StartDateTime"] <= current["EndDateTime"]

                if same_value and overlap_or_touching:
                    current["EndDateTime"] = max(current["EndDateTime"], row["EndDateTime"])
                else:
                    # Different value or disjoint: clip to avoid overlap
                    if row["StartDateTime"] < current["EndDateTime"]:
                        current["EndDateTime"] = row["StartDateTime"]

                    # Keep only positive-length intervals
                    if pd.notna(current["StartDateTime"]) and pd.notna(current["EndDateTime"]) \
                    and current["EndDateTime"] >= current["StartDateTime"]:
                        merged_rows.append(current)

                    current = row

            if current is not None:
                if pd.notna(current["StartDateTime"]) and pd.notna(current["EndDateTime"]) \
                and current["EndDateTime"] > current["StartDateTime"]:
                    merged_rows.append(current)

        out = pd.DataFrame(merged_rows, columns=[
            "PatientId","LOINC-Code","ConceptName","Value","StartDateTime","EndDateTime","Source"
        ])
        return out.sort_values(by=["StartDateTime", "LOINC-Code","ConceptName"]).reset_index(drop=True)


    def run(self, patient_id, snapshot_date, relevance=24):
        """
        Run the temporal abstraction engine for a single patient.

        Retrieves raw measurement records, applies applicable abstraction rules,
        merges overlapping abstracted intervals, and returns both abstracted and untouched
        measurement records in a unified format.
        Will set "relevance" duration of 24h for raw records as well as to abstracted records, 
        which may have a longer duration, depends on the intervals.

        Args:
            patient_id (str or int): Patient identifier in the database.
            snapshot_date (str): View of the DB up to this date (default: today).
            relevance (int, optional): Number of hours each measure is relevant for (default: 24 hours).

        Returns:
            pd.DataFrame: All records in unified format:
                ['PatientId', 'LOINC-Code', 'ConceptName', 'Value', 'StartDateTime', 'EndDateTime', 'Source']
        """
        # Step 1: Retrieve raw measurements + patient attributes (e.g., sex)
        raw_rows, params = self._get_patient_records(patient_id, snapshot_date)
        if not raw_rows:
            return pd.DataFrame(columns=[
                "PatientId", "LOINC-Code", "ConceptName", "Value", "StartDateTime", "EndDateTime", "Source"
            ])

        # Step 2: Convert raw rows to DataFrame + add Source column
        raw_df = pd.DataFrame(raw_rows, columns=[
            'LOINC-Code', 'ConceptName', 'Value', 'Unit',
            'ValidStartTime', 'TransactionTime'
        ])
        # Add rows not in output
        raw_df['PatientId'] = patient_id
        raw_df['Source'] = "original_value"
        required_fields = {'LOINC-Code', 'ConceptName', 'Value', 'ValidStartTime'}
        assert required_fields.issubset(raw_df.columns), "Missing required columns in measurement data"

        # Step 3: Apply each applicable abstraction rule
        used_indices = set()
        abstracted_records = []
        for rule in self.tak_rules:
            if not rule.applies_to(params):
                continue

            rule_df = raw_df[raw_df['LOINC-Code'] == rule.loinc_code]
            if rule_df.empty:
                continue

            result = rule.apply(rule_df)
            for row in result['abstracted']:
                abstracted_records.append({
                    "PatientId": patient_id,
                    "LOINC-Code": rule.loinc_code,
                    "ConceptName": rule.abstraction_name,
                    "Value": row["Value"],
                    "StartDateTime": row["StartDateTime"],
                    "EndDateTime": row["EndDateTime"],
                    "Source": "abstracted_value"
                })
            used_indices.update(result['used_indices'])
        
        # Step 4: Cast as df
        if not abstracted_records:
           abstracted_records = pd.DataFrame(columns=[
                "PatientId", "LOINC-Code", "ConceptName", "Value", "StartDateTime", "EndDateTime", "Source"
            ])
        else: 
            abstracted_records = pd.DataFrame(abstracted_records)
            abstracted_records['StartDateTime'] = pd.to_datetime(abstracted_records['StartDateTime'], errors='coerce')
            abstracted_records['EndDateTime'] = pd.to_datetime(abstracted_records['EndDateTime'], errors='coerce')

        # Step 5: Process untouched raw records
        untouched = raw_df[~raw_df.index.isin(used_indices)].copy()
        untouched = untouched.rename(columns={"ValidStartTime": "StartDateTime"})
        untouched['StartDateTime'] = pd.to_datetime(untouched['StartDateTime'], errors='coerce')
        # Strech duration to every unabstracted record
        untouched['EndDateTime'] = untouched['StartDateTime'] + timedelta(hours=relevance)
        untouched['PatientId'] = patient_id

        # Step 6: Combine all
        frames = [
            abstracted_records,
            untouched[["PatientId", "LOINC-Code", "ConceptName", "Value", "StartDateTime", "EndDateTime", "Source"]]
        ]
        merged_records = pd.concat([df for df in frames if not df.empty], ignore_index=True)

        # Step 7: Merge intervals (safely across all LOINC codes)
        final_df = self._merge_intervals(merged_records)

        return final_df.sort_values(by=["PatientId", "StartDateTime"]).reset_index(drop=True)
    


if __name__ == "__main__":

    # --- CONFIGURABLE INPUT ---
    patient_id = "123456782"
    snapshot_date = "2025-08-02 12:00:00"

    # --- RUN MEDIATOR ---
    engine = Mediator()
    result_df = engine.run(patient_id=patient_id, snapshot_date=snapshot_date)

    # --- DISPLAY RESULT ---
    if result_df.empty:
        print(f"[Info] No data available for Patient {patient_id} on snapshot {snapshot_date}")
    else:
        print(f"[Info] Abstracted records for Patient {patient_id} on {snapshot_date}:")
        print(result_df.to_string(index=False))
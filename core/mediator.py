import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
from dateutil import parser as dateparser

# Local Code
from backend.dataaccess import DataAccess
from backend.backend_config import * 


def parse_duration(duration_str):
    """
    Convert a compact duration string (e.g. '72h', '2d', '15m') into a timedelta object.

    Args:
        duration_str (str): A string with a number followed by a unit character:
                            - 'h' for hours
                            - 'd' for days
                            - 'm' for minutes

    Returns:
        timedelta: A timedelta representing the duration.

    Raises:
        ValueError: If the input format is invalid or unit is unsupported.
    """
    unit_map = {'h': 'hours', 'd': 'days', 'm': 'minutes'}
    value = int(duration_str[:-1])
    unit = unit_map[duration_str[-1]]
    return timedelta(**{unit: value})


class TAKRule:
    """
    A single temporal abstraction rule derived from a TAK XML file.

    Attributes:
        abstraction_name (str): The name of the abstract concept (e.g., 'Hemoglobin State').
        loinc_code (str): The LOINC code this rule applies to.
        filters (dict): Key-value constraints to check against patient parameters (e.g., {'sex': 'Male'}).
        good_before (timedelta): Time window before the measurement to consider the interval valid.
        good_after (timedelta): Time window after the measurement to consider the interval valid.
        rules (list of dict): List of abstraction thresholds, each with:
            - 'label': Name of the category (e.g., 'Low', 'Normal')
            - 'min': Minimum numeric value (inclusive) or None
            - 'max': Maximum numeric value (exclusive) or None
    """
    def __init__(self, abstraction_name, loinc_code, filters, persistence, rules):
        self.abstraction_name = abstraction_name
        self.loinc_code = loinc_code
        self.filters = filters  # e.g., {'sex': 'Male'} or {'age_group': 'Adult'}
        self.good_before = parse_duration(persistence['before'])
        self.good_after = parse_duration(persistence['after'])
        self.rules = rules


    def applies_to(self, patient_params):
        """
        Checks whether this rule is applicable to the patient based on filters.

        Args:
            patient_params (dict): Any number of demographic or contextual attributes.

        Returns:
            bool: True if all filters match patient parameters; False otherwise.
        """
        for key, value in self.filters.items():
            if key not in patient_params or str(patient_params[key]).lower() != str(value).lower():
                return False
        return True


    def apply(self, df):
        """
        Applies abstraction to raw measurement data based on defined rules.

        Args:
            df (pd.DataFrame): Must contain measurements with 'Value' and 'ValidStartTime' columns.

        Returns:
            dict:
                - 'abstracted': List of abstracted interval records
                - 'used_indices': Set of indices of original rows that were abstracted
        """
        used_indices = set()
        abstracted_records = []

        for _, row in df.iterrows():
            val = float(row['Value'])
            match = None
            # Find the discrete value for this row, store in match
            for rule in self.rules:
                if ((rule['min'] is None or val >= rule['min']) and
                    (rule['max'] is None or val < rule['max'])):
                    match = rule['label']
                    break
            # Create a perimiter around the matched value based on good_before, good_after
            if match:
                start_time = dateparser.parse(row['ValidStartTime']) - self.good_before
                end_time = dateparser.parse(row['ValidStartTime']) + self.good_after
                abstracted_records.append({
                    "LOINC-Code": self.loinc_code,
                    "ConceptName": self.abstraction_name,
                    "Value": match,
                    "StartDateTime": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "EndDateTime": end_time.strftime('%Y-%m-%d %H:%M:%S')
                })
                used_indices.add(row.name)

        return {"abstracted": abstracted_records, "used_indices": used_indices}


class TAKParser:
    """
    Parses TAK XML files into executable abstraction rules (TAKRule).
    Each TAK file can contain multiple conditions for different demographics.

    Attributes:
        tak_folder (str): Folder path containing TAK XML rule files.
    """
    def __init__(self, tak_folder):
        self.tak_folder = tak_folder
        self._validate_tak_repository() # Validate TAK repo during initialization

    
    def _validate_tak_repository(self):
        tak_files = glob.glob(os.path.join(self.tak_folder, '*.xml'))

        if not tak_files:
            raise FileNotFoundError(f"No TAK files found in folder: {self.tak_folder}")

        for path in tak_files:
            valid, reason = self._validate_tak_file(path)
            if not valid:
                raise ValueError(f"TAK file '{os.path.basename(path)}' is invalid: {reason}")


    def _validate_tak_file(self, path):
        """
        Validates the structure of a TAK file.
        Returns (True, "") if valid; otherwise (False, reason)
        """
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            if root.tag != 'abstraction':
                return False, "Root element must be <abstraction>"

            if 'name' not in root.attrib or 'loinc' not in root.attrib:
                return False, "Missing required attributes 'name' or 'loinc' in <abstraction>"

            conditions = root.findall('condition')
            if not conditions:
                return False, "No <condition> elements found"

            for cond in conditions:
                persistence = cond.find('persistence')
                if persistence is None:
                    return False, "<condition> missing <persistence>"

                if 'good-before' not in persistence.attrib or 'good-after' not in persistence.attrib:
                    return False, "<persistence> must have 'good-before' and 'good-after'"

                rules = cond.findall('rule')
                if not rules:
                    return False, "<condition> must have at least one <rule>"

                for r in rules:
                    if 'value' not in r.attrib:
                        return False, "<rule> missing required 'value' attribute"
                    for bound in ['min', 'max']:
                        if bound in r.attrib:
                            try:
                                float(r.attrib[bound])
                            except ValueError:
                                return False, f"<rule> has non-numeric '{bound}' attribute: {r.attrib[bound]}"

        except ET.ParseError as e:
            return False, f"XML parsing error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

        return True, ""


    def load_all_taks(self):
        """
        Load all the TAK files into a list of TAKRule objects that can be applied to patient data.

        Returns:
            list of TAKRule: Fully constructed abstraction rules with conditions, thresholds, and persistence.
        """
        rules = []
        for path in glob.glob(os.path.join(self.tak_folder, '*.xml')):
            tree = ET.parse(path)
            root = tree.getroot()
            abstraction_name = root.attrib['name']
            loinc_code = root.attrib['loinc']

            for cond in root.findall('condition'):
                # Extract all condition-level attributes dynamically
                filters = {k: v for k, v in cond.attrib.items()}

                # Parse persistence window
                persistence = cond.find('persistence')
                p_before = persistence.attrib['good-before']
                p_after = persistence.attrib['good-after']

                # Parse rule thresholds
                rule_objs = []
                for r in cond.findall('rule'):
                    rule_objs.append({
                        'label': r.attrib['value'],
                        'min': float(r.attrib['min']) if 'min' in r.attrib else None,
                        'max': float(r.attrib['max']) if 'max' in r.attrib else None
                    })

                # Create TAKRule for this condition
                rules.append(TAKRule(
                    abstraction_name,
                    loinc_code,
                    filters,
                    {'before': p_before, 'after': p_after},
                    rule_objs
                ))

        return rules


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
    

    def _get_patient_records(self, patient_id, snapshot_date):
        """
        Retrieves patient measurement records and patient-level attributes for abstraction.
        
        Args:
            patient_id (str): The ID of the patient.
            snapshot_date (str or datetime): The point-in-time view of the DB. Assumed to already be a parsed date.

        Returns:
            tuple: (list of measurement rows, dict of patient demographic params)
        """
        if not self.db.check_record(CHECK_PATIENT_BY_ID_QUERY, (patient_id,)):
            return [], {}

        # Construct query
        filters = [
            "m.PatientId = ?",
            "m.TransactionInsertionTime <= ?",
            "(m.TransactionDeletionTime IS NULL OR m.TransactionDeletionTime > ?)"
        ]
        params = [patient_id, snapshot_date, snapshot_date]

        with open(GET_HISTORY_QUERY, 'r') as f:
            base_query = f.read()

        final_query = base_query.replace("{where_clause}", " AND ".join(filters))
        patient_records = self.db.fetch_records(final_query, params)

        # --- Fetch patient demographic parameters ---
        patient_info = self.db.fetch_records(GET_PATIENT_PARAMS_QUERY, (patient_id,))
        param_dict = {}
        if patient_info:
            row = patient_info[0]  # Single patient expected
            # Gives columns returned by the last query, lowered.
            columns = [desc[0].lower() for desc in self.db.cursor.description]
            param_dict = dict(zip(columns, row))

        return patient_records, param_dict
    

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
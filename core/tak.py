import os
import glob
import xml.etree.ElementTree as ET
from dateutil import parser as dateparser

from core.utils import *


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
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Union
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, get_tak_repository, validate_xml_against_schema, TemporalRelationRule, QATPRule
from .raw_concept import RawConcept
from .event import Event
from .context import Context
from .utils import apply_external_function, parse_duration
from .external_functions import REPO

class Pattern(TAK):
    """
    Base class for all pattern TAKs (local/global).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        abstraction_rules: List[Dict[str, Any]],
        pattern_type: str = "pattern"
    ):
        super().__init__(name=name, categories=categories, description=description, family=pattern_type)
        self.derived_from = derived_from
        self.abstraction_rules = abstraction_rules

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "Pattern":
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

    def validate(self) -> None:
        raise NotImplementedError("Use LocalPattern or GlobalPattern subclasses.")

class LocalPattern(Pattern):
    """
    LocalPattern: interval-based pattern abstraction.
    Output values: True, Partial, False (False only for QATPs).
    """
    def __init__(
        self,
        name: str,
        categories: Tuple[str, ...],
        description: str,
        derived_from: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        abstraction_rules_operator: str,
        abstraction_rules: List[Dict[str, Any]]
    ):
        super().__init__(
            name=name,
            categories=categories,
            description=description,
            derived_from=derived_from,
            abstraction_rules=abstraction_rules,
            pattern_type="local-pattern"
        )
        self.parameters = parameters
        self.abstraction_rules_operator = abstraction_rules_operator

    @classmethod
    def parse(cls, xml_path: Union[str, Path]) -> "LocalPattern":
        xml_path = Path(xml_path)
        validate_xml_against_schema(xml_path)
        root = ET.parse(xml_path).getroot()
        if root.tag != "pattern":
            raise ValueError(f"{xml_path} is not a pattern file")

        name = root.attrib["name"]
        cats = tuple(s.strip() for s in (root.findtext("categories") or "").split(",") if s.strip())
        desc = root.findtext("description") or ""

        # --- derived-from (required, one or more) ---
        df_el = root.find("derived-from")
        if df_el is None:
            raise ValueError(f"{name}: missing <derived-from> block")
        derived_from = []
        for attr_el in df_el.findall("attribute"):
            if "name" not in attr_el.attrib or "tak" not in attr_el.attrib:
                raise ValueError(f"{name}: <derived-from><attribute> must have 'name' and 'tak' attributes")
            spec = {
                "name": attr_el.attrib["name"],
                "tak": attr_el.attrib["tak"]
            }
            derived_from.append(spec)
        if not derived_from:
            raise ValueError(f"{name}: <derived-from> must contain at least one <attribute>")

        # --- parameters (optional) ---
        parameters = []
        params_el = root.find("parameters")
        if params_el is not None:
            for param_el in params_el.findall("parameter"):
                param_spec = {
                    "name": param_el.attrib["name"],
                    "tak": param_el.attrib["tak"],
                    "idx": int(param_el.attrib.get("idx", 0)),
                    "ref": param_el.attrib.get("ref"),
                    "default": param_el.attrib.get("default")
                }
                parameters.append(param_spec)

        # --- abstraction-rules (required) ---
        abs_el = root.find("abstraction-rules")
        if abs_el is None:
            raise ValueError(f"{name}: <pattern> must define <abstraction-rules>")
        abstraction_rules_operator = abs_el.attrib.get("operator", "and")
        abstraction_rules = []
        for rule_el in abs_el.findall("rule"):
            rule_spec = {}

            # --- context (optional) ---
            context_el = rule_el.find("context")
            if context_el is not None:
                context_spec = {}
                for attr_el in context_el.findall("attribute"):
                    attr_name = attr_el.attrib["name"]
                    allowed = set()
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed.add(av.attrib["equal"])
                        else:
                            raise ValueError(f"{name}: only 'equal' constraints supported in context (nominal) attributes")
                    context_spec[attr_name] = allowed
                rule_spec["context"] = context_spec

            # --- temporal-relation (required) ---
            tr_el = rule_el.find("temporal-relation")
            if tr_el is None:
                raise ValueError(f"{name}: rule missing <temporal-relation>")
            how = tr_el.attrib.get("how")
            if how not in ("before", "overlap"):
                raise ValueError(f"{name}: temporal-relation how='{how}' must be 'before' or 'overlap'")
            max_distance = tr_el.attrib.get("max-distance")
            if how == "before" and not max_distance:
                raise ValueError(f"{name}: temporal-relation how='before' requires max-distance")
            temporal_relation = {
                "how": how,
                "max_distance": max_distance
            }

            # anchor
            anchor_el = tr_el.find("anchor")
            if anchor_el is not None:
                anchor_spec = {
                    "select": anchor_el.attrib.get("select", "first")
                }
                anchor_attrs = {}
                for attr_el in anchor_el.findall("attribute"):
                    attr_name = attr_el.attrib["name"]
                    idx = int(attr_el.attrib.get("idx", 0))
                    allowed_values = set()
                    min_val = None
                    max_val = None
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(av.attrib["equal"])
                        if "min" in av.attrib:
                            min_val = float(av.attrib["min"])
                        if "max" in av.attrib:
                            max_val = float(av.attrib["max"])
                    anchor_attrs[attr_name] = {
                        "idx": idx,
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                anchor_spec["attributes"] = anchor_attrs
                # Check if more than 1 attr, select must be mentioned
                if len(anchor_attrs) > 1 and anchor_spec["select"] == "first":
                    raise ValueError(f"{name}: anchor with multiple attributes requires select attribute")
                temporal_relation["anchor"] = anchor_spec

            # event
            event_el = tr_el.find("event")
            if event_el is not None:
                event_spec = {
                    "select": event_el.attrib.get("select", "first")
                }
                event_attrs = {}
                for attr_el in event_el.findall("attribute"):
                    attr_name = attr_el.attrib["name"]
                    idx = int(attr_el.attrib.get("idx", 0))
                    allowed_values = set()
                    min_val = None
                    max_val = None
                    for av in attr_el.findall("allowed-value"):
                        if "equal" in av.attrib:
                            allowed_values.add(av.attrib["equal"])
                        if "min" in av.attrib:
                            min_val = float(av.attrib["min"])
                        if "max" in av.attrib:
                            max_val = float(av.attrib["max"])
                    event_attrs[attr_name] = {
                        "idx": idx,
                        "allowed_values": allowed_values,
                        "min": min_val,
                        "max": max_val
                    }
                event_spec["attributes"] = event_attrs
                # Check if more than 1 attr, select must be mentioned
                if len(event_attrs) > 1 and event_spec["select"] == "first":
                    raise ValueError(f"{name}: event with multiple attributes requires select attribute")
                temporal_relation["event"] = event_spec

            rule_spec["temporal_relation"] = temporal_relation

            # --- compliance-function (optional) ---
            cf_el = rule_el.find("compliance-function")
            if cf_el is not None:
                compliance_function = {}

                # time-constraint-compliance
                tcc_el = cf_el.find("time-constraint-compliance")
                if tcc_el is not None:
                    func_el = tcc_el.find("function")
                    if func_el is not None:
                        func_name = func_el.attrib["name"]
                        # Check function is in REPO
                        if func_name not in REPO:
                            raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions. Ensure it's declared in REPO parameter.")
                        trapez = (func_el.find("trapeze").attrib["trapezeA"], func_el.find("trapeze").attrib["trapezeB"], func_el.find("trapeze").attrib["trapezeC"], func_el.find("trapeze").attrib["trapezeD"])
                        if len(trapez) != 4:
                            raise ValueError(f"{name}: trapez must have 4 values")
                        # Check trapez values are all time or all numeric
                        time_like = []
                        for val in trapez:
                            try:
                                parse_duration(val)
                                time_like.append(True)
                            except:
                                time_like.append(False)
                        if not all(time_like) and not all(not t for t in time_like):
                            raise ValueError(f"{name}: trapez values must be all time representations or all numeric")
                        time_spec = {
                            "function_name": func_name,
                            "trapez": trapez
                        }
                        parameters_el = func_el.find("parameters")
                        if parameters_el is not None:
                            params = []
                            for p in parameters_el.findall("parameter"):
                                ref = p.attrib.get("ref")
                                if ref not in [p["name"] for p in parameters]:
                                    raise ValueError(f"{name}: parameter ref '{ref}' not found in declared parameters")
                                params.append({"ref": ref})
                            time_spec["parameters"] = params
                        compliance_function["time_constraint"] = time_spec

                # value-constraint-compliance
                vcc_el = cf_el.find("value-constraint-compliance")
                if vcc_el is not None:
                    func_el = vcc_el.find("function")
                    if func_el is not None:
                        func_name = func_el.attrib["name"]
                        if func_name not in REPO:
                            raise ValueError(f"{name}: compliance function '{func_name}' not found in external functions")
                        trapez = (func_el.find("trapeze").attrib["trapezeA"], func_el.find("trapeze").attrib["trapezeB"], func_el.find("trapeze").attrib["trapezeC"], func_el.find("trapeze").attrib["trapezeD"])
                        if len(trapez) != 4:
                            raise ValueError(f"{name}: trapez must have 4 values")
                        time_like = []
                        for val in trapez:
                            try:
                                parse_duration(val)
                                time_like.append(True)
                            except:
                                time_like.append(False)
                        if not all(time_like) and not all(not t for t in time_like):
                            raise ValueError(f"{name}: trapez values must be all time representations or all numeric")
                        value_spec = {
                            "function_name": func_name,
                            "trapez": trapez
                        }
                        parameters_el = func_el.find("parameters")
                        if parameters_el is not None:
                            params = []
                            for p in parameters_el.findall("parameter"):
                                ref = p.attrib.get("ref")
                                if ref not in [p["name"] for p in parameters]:
                                    raise ValueError(f"{name}: parameter ref '{ref}' not found in declared parameters")
                                params.append({"ref": ref})
                            value_spec["parameters"] = params
                        compliance_function["value_constraint"] = value_spec

                rule_spec["compliance_function"] = compliance_function

            abstraction_rules.append(rule_spec)

        pattern = cls(
            name=name,
            categories=cats,
            description=desc,
            derived_from=derived_from,
            parameters=parameters,
            abstraction_rules_operator=abstraction_rules_operator,
            abstraction_rules=abstraction_rules
        )
        pattern.validate()
        return pattern

    def validate(self) -> None:
        repo = get_tak_repository()
        
        # Early validation: derived_from and parameters must have required fields
        for df in self.derived_from:
            if not df["name"] or not df["tak"]:
                raise ValueError(f"{self.name}: derived_from must have non-empty 'name' and 'tak'")
        
        for param in self.parameters:
            if not param["name"] or not param["tak"] or not param["ref"] or not param["default"]:
                raise ValueError(f"{self.name}: parameters must have non-empty 'name', 'tak', 'ref', 'default'")
            # idx is mandatory only if parameter is a raw-concept
            tak = repo.get(param["tak"])
            if tak is None:
                raise ValueError(f"{self.name}: parameter TAK '{param['tak']}' not found")
            if isinstance(tak, RawConcept) and tak.concept_type == "raw":
                if "idx" not in param or param["idx"] is None:
                    raise ValueError(f"{self.name}: parameter '{param['name']}' references raw-concept '{param['tak']}', so idx is mandatory")
        
        # 1) All derived-from TAKs must exist
        for df in self.derived_from:
            tak = repo.get(df["name"])
            if tak is None:
                raise ValueError(f"{self.name}: derived_from TAK '{df['name']}' not found in repository")
            if df["tak"] == "raw-concept":
                # Must mention idx in abstraction rules if referenced
                if not any(
                    any(attr["name"] == df["name"] and "idx" in attr for tr in rule["temporal_relation"] for attr in tr.get("anchor", {}).get("attributes", {}).values() or tr.get("event", {}).get("attributes", {}).values())
                    for rule in self.abstraction_rules
                ):
                    raise ValueError(f"{self.name}: raw-concept '{df['name']}' must be referenced with idx in abstraction rules")
        
        # 2) Must have abstraction rules
        if not self.abstraction_rules:
            raise ValueError(f"{self.name}: LocalPattern must define abstraction rules")
        
        # 3) Each rule must have at least one temporal-relation
        for rule in self.abstraction_rules:
            if "temporal_relation" not in rule:
                raise ValueError(f"{self.name}: abstraction rule missing temporal-relation")
            tr = rule["temporal_relation"]
            if "anchor" not in tr and "event" not in tr:
                raise ValueError(f"{self.name}: temporal-relation missing anchor or event")
            for attr in tr.get("anchor", {}).keys():
                tak = repo.get(attr)
                if tak is None:
                    raise ValueError(f"{self.name}: anchor attribute TAK '{attr}' not found")
                if attr not in [d["name"] for d in self.derived_from]:
                    raise ValueError(f"{self.name}: anchor attribute TAK '{attr}' not found in derived-from")
            for attr in tr.get("event", {}).get("attributes", {}).keys():
                tak = repo.get(attr)
                if tak is None:
                    raise ValueError(f"{self.name}: event attribute TAK '{attr}' not found")
                if attr not in [d["name"] for d in self.derived_from]:
                    raise ValueError(f"{self.name}: event attribute TAK '{attr}' not found in derived-from")
                # Only raw-concept numeric can use min/max constraints
                for constraint in tr["event"]["attributes"][attr]["allowed"]:
                    if constraint["type"] in ("min", "max"):
                        if not (isinstance(tak, RawConcept) and any(a["type"] == "numeric" for a in tak.attributes)):
                            raise ValueError(f"{self.name}: min/max constraint only allowed for numeric raw-concept attributes")
        # Add validation for context, anchor, event attributes
        derived_from_names = {df["name"] for df in self.derived_from}
        for rule in self.abstraction_rules:
            if "context" in rule:
                for attr_name in rule["context"]:
                    if attr_name not in derived_from_names:
                        raise ValueError(f"{self.name}: context attribute '{attr_name}' not in derived_from")
            tr = rule["temporal_relation"]
            for attr_name in tr.get("anchor", {}).get("attributes", {}):
                if attr_name not in derived_from_names:
                    raise ValueError(f"{self.name}: anchor attribute '{attr_name}' not in derived_from")
                # Check allowed values
                tak = repo.get(attr_name)
                attr_spec = tr["anchor"]["attributes"][attr_name]
                if isinstance(tak, (Context, Event)) or (isinstance(tak, RawConcept) and tak.concept_type in ("raw-nominal", "raw-boolean")):
                    if not attr_spec["allowed_values"].issubset(set(tak.attributes[0]["allowed"]) if tak.attributes else set()):
                        raise ValueError(f"{self.name}: anchor allowed values for '{attr_name}' not in TAK's allowed values")
                elif isinstance(tak, RawConcept) and any(a["type"] == "numeric" for a in tak.attributes):
                    if attr_spec["min"] is not None and (tak.attributes[0]["min"] is not None and attr_spec["min"] < tak.attributes[0]["min"]):
                        raise ValueError(f"{self.name}: anchor min for '{attr_name}' below TAK's min")
                    if attr_spec["max"] is not None and (tak.attributes[0]["max"] is not None and attr_spec["max"] > tak.attributes[0]["max"]):
                        raise ValueError(f"{self.name}: anchor max for '{attr_name}' above TAK's max")
            for attr_name in tr.get("event", {}).get("attributes", {}):
                if attr_name not in derived_from_names:
                    raise ValueError(f"{self.name}: event attribute '{attr_name}' not in derived_from")
                # Similar checks for event
                tak = repo.get(attr_name)
                attr_spec = tr["event"]["attributes"][attr_name]
                if isinstance(tak, (Context, Event)) or (isinstance(tak, RawConcept) and tak.concept_type in ("raw-nominal", "raw-boolean")):
                    if not attr_spec["allowed_values"].issubset(set(tak.attributes[0]["allowed"]) if tak.attributes else set()):
                        raise ValueError(f"{self.name}: event allowed values for '{attr_name}' not in TAK's allowed values")
                elif isinstance(tak, RawConcept) and any(a["type"] == "numeric" for a in tak.attributes):
                    if attr_spec["min"] is not None and (tak.attributes[0]["min"] is not None and attr_spec["min"] < tak.attributes[0]["min"]):
                        raise ValueError(f"{self.name}: event min for '{attr_name}' below TAK's min")
                    if attr_spec["max"] is not None and (tak.attributes[0]["max"] is not None and attr_spec["max"] > tak.attributes[0]["max"]):
                        raise ValueError(f"{self.name}: event max for '{attr_name}' above TAK's max")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LocalPattern to find and rate pattern instances.
        Input: Patient data (from derived_from TAKs)
        Output: Intervals with Value=True/Partial/False (False only for QA)
        """
        if df.empty:
            return pd.DataFrame(columns=["PatientId","ConceptName","StartDateTime","EndDateTime","Value","AbstractionType"])

        logger.info("[%s] apply() start | input_rows=%d", self.name, len(df))

        # Group by patient
        patients = df['PatientId'].unique()
        results = []
        for pid in patients:
            patient_df = df[df['PatientId'] == pid]
            patient_patterns = self._find_patterns(patient_df)
            for pattern in patient_patterns:
                compliance = self._rate_compliance(pattern)
                if compliance != 'False':  # Only True/Partial to OutputPatientData
                    results.append({
                        'PatientId': pid,
                        'ConceptName': self.name,
                        'StartDateTime': pattern['start'],
                        'EndDateTime': pattern['end'],
                        'Value': compliance,
                        'AbstractionType': self.family
                    })
                # For QA: insert False as well, but separate table

        out_df = pd.DataFrame(results)
        logger.info("[%s] apply() end | output_rows=%d", self.name, len(out_df))
        return out_df

    def _find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find pattern instances using rules and chaining for operator="and".
        """
        patterns = []
        used_anchors = set()
        used_events = set()

        for rule in self.abstraction_rules:
            tr = rule['temporal_relation']
            anchor_df = self._filter_df(df, tr.get('anchor', {}))
            event_df = self._filter_df(df, tr.get('event', {}))

            rule_matches = []
            for match in TemporalRelationRule(tr.get('anchor', {}), tr.get('event', {}), tr['how'], tr.get('max_distance')).matches(anchor_df, event_df):
                if match['anchor_row'].name not in used_anchors and match['event_row'].name not in used_events:
                    rule_matches.append(match)
                    used_anchors.add(match['anchor_row'].name)
                    used_events.add(match['event_row'].name)

            if self.abstraction_rules_operator == 'and':
                # Chain with previous rules
                if patterns:
                    chained = []
                    for prev_pattern in patterns:
                        for match in rule_matches:
                            if self._can_chain(prev_pattern, match, rule):
                                chained.append(self._merge_patterns(prev_pattern, match))
                    patterns = chained
                else:
                    patterns = rule_matches
            else:
                patterns.extend(rule_matches)

        return patterns

    def _can_chain(self, prev_pattern: Dict[str, Any], match: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        # Check distance between prev event and current anchor/event
        prev_end = prev_pattern['end']
        curr_start = match['start']
        max_dist = parse_duration(rule['temporal_relation'].get('max_distance', '0h'))
        return (curr_start - prev_end) <= max_dist

    def _merge_patterns(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
        # Merge start/end
        return {
            'start': min(prev['start'], curr['start']),
            'end': max(prev['end'], curr['end']),
            'anchor_row': curr['anchor_row'],  # Update to latest
            'event_row': curr['event_row']
        }

    def _rate_compliance(self, pattern: Dict[str, Any]) -> str:
        # Use QATPRule
        compliance_spec = {}  # From rule's compliance_function
        return QATPRule(compliance_spec).matches(pattern)

    def _filter_df(self, df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
        # Filter df based on spec (allowed_values, min/max)
        # Implement filtering logic
        pass
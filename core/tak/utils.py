from datetime import timedelta
from dataclasses import dataclass
from typing import Union, Dict, TYPE_CHECKING
import logging

from .external_functions import REPO

if TYPE_CHECKING:
    from .repository import TAKRepository

logger = logging.getLogger(__name__)


def _map_tak_to_concepts(
    repo: "TAKRepository",
    input_concepts: set,
    global_clippers: Dict[str, str],
) -> None:
    """
    Purpose: Detect concept-name mismatches between the input table and the TAK
             repository early, so misconfigured raw concepts or stale input columns
             surface before patient processing starts.
    Method: Build the set of concept names referenced by non-parameterized
            RawConcept attributes (these are the names actually queried against
            InputPatientData) plus the global clippers, then symmetric-diff it
            against the set of distinct ConceptName values in the input table and
            log a warning per mismatch.

    Args:
        repo (TAKRepository): The finalized TAK repository.
        input_concepts (set): Distinct ConceptName values present in InputPatientData.
        global_clippers (Dict[str, str]): {clipper_name: 'START' | 'END'} — these
            are queried from InputPatientData directly, not through a raw concept.
    """
    # Local imports to avoid circular dependencies
    from .raw_concept import RawConcept, ParameterizedRawConcept

    # Build the set of all attribute names referenced by non-parameterized raw concepts
    # (used to detect "wasted" concept names in InputPatientData below).
    all_referenced_attrs: set = set()
    raw_concepts = []
    for tak in repo.taks.values():
        if isinstance(tak, RawConcept) and not isinstance(tak, ParameterizedRawConcept):
            raw_concepts.append(tak)
            for attr in tak.attributes:
                all_referenced_attrs.add(attr["name"])
    all_referenced_attrs.update(global_clippers.keys())

    # 1) In input but referenced by NO raw concept (and no global clipper) → wasted data / typo
    for cn in sorted(input_concepts - all_referenced_attrs):
        logger.warning(
            f"Concept '{cn}' is in InputPatientData but is not referenced by any raw-concept attribute "
            f"(or global clipper). It will not be used by the pipeline."
        )

    # 2) Raw concept will NOT be calculable: ALL of its attribute names are missing from InputPatientData.
    #    (Partial misses are normal — only a fully-missing raw concept breaks downstream TAKs.)
    for tak in raw_concepts:
        attr_names = [a["name"] for a in tak.attributes]
        missing = [n for n in attr_names if n not in input_concepts]
        if missing and len(missing) == len(attr_names):
            logger.warning(
                f"Raw concept '{tak.name}' cannot be calculated: none of its attributes "
                f"{attr_names} appear in InputPatientData. Any TAK depending on it will receive empty input."
            )

    # 3) Global clippers missing from input → emit per missing clipper (each is its own concept)
    for cn in sorted(set(global_clippers.keys()) - input_concepts):
        logger.warning(
            f"Global clipper '{cn}' does not appear in InputPatientData. "
            f"Clipping for this boundary will be skipped."
        )


@dataclass(frozen=True)
class FuzzyLogicTrapez:
    """Immutable trapezoid node for compliance scoring.
    
    Supports both:
    - Time-based: timedelta values (A, B, C, D all as timedelta)
    - Value-based: float values (A, B, C, D all as float)

    Additional Attributes:
        is_time (bool): True if time-constraint (timedelta), False if value-constraint (float).
        missing_score (float): Score to assign if for unmatched anchors/ 'False' rows. Default is 0.0.
    
    Order: A <= B <= C <= D (validated at parse time).
    """
    A: Union[float, timedelta]
    B: Union[float, timedelta]
    C: Union[float, timedelta]
    D: Union[float, timedelta]
    is_time: bool = False
    missing_score: float = 0.0
    
    def __post_init__(self):
        """Post-initialization to validate and convert types."""
        # Convert A, B, C, D to correct type
        if self.is_time:
            object.__setattr__(self, 'A', parse_duration(self.A) if isinstance(self.A, str) else self.A)
            object.__setattr__(self, 'B', parse_duration(self.B) if isinstance(self.B, str) else self.B)
            object.__setattr__(self, 'C', parse_duration(self.C) if isinstance(self.C, str) else self.C)
            object.__setattr__(self, 'D', parse_duration(self.D) if isinstance(self.D, str) else self.D)
        else:
            object.__setattr__(self, 'A', float(self.A))
            object.__setattr__(self, 'B', float(self.B))
            object.__setattr__(self, 'C', float(self.C))
            object.__setattr__(self, 'D', float(self.D))
        
        self.validate()
        # Compute missing_score: score for a value for missed anchors
        object.__setattr__(self, 'missing_score', self.compliance_direction())
    
    def validate(self) -> None:
        """Ensure trapez is well-formed: A <= B <= C <= D."""
        if not (self.A <= self.B <= self.C <= self.D):
            raise ValueError(
                f"Invalid trapez order: A={self.A}, B={self.B}, C={self.C}, D={self.D}. "
                f"Must satisfy A <= B <= C <= D."
            )
    
    def compliance_direction(self) -> float:
        """
        Determine compliance direction based on trapez shape.
        Trapez is always flat-topped between B and C.
        
        Returns:
            float: 1.0 if higher values are better, 0.0 if lower values are better.
        """
        if self.C < self.D and self.A == self.B:
            if not self.is_time:
                # Perfect compliance at left edge, so lower to None values for value compliance are better.
                return 1.0
            # Later -> deminishing returns: higher values for time compliance are worse, missing score 0.0
            # Or, if value constraint but not A==B<>0 then we can't tell direction, so default to 0.0 
            return 0.0
        elif self.A < self.B and self.C == self.D:
            # Later -> increasing returns: higher values are better, missing score 1.0
            # Can only happen for time-constraint where missing event can count as longer time between events
            # For value-constraint, this shape is nonsensical, since higher values are not compareable to missing value, but missing values are like lower values
            if self.is_time:
                return 1.0
            else:
                return 0.0
        elif self.A <= self.B and self.C <= self.D:
            # Accuracy matters: anything between B and C is best (1.0), missing score 0.0
            return 0.0
        else:
            # Mixed or undefined shape
            return 0.0
        
    def compliance_score(self, value: Union[float, timedelta]) -> float:
        """
        Compute compliance score for a given value using piecewise-linear interpolation.
        
        Score is 1.0 (100%) between B and C.
        Score is 0.0 (0%) outside [A, D].
        Score linearly interpolates:
          - [A, B]: 0 → 1
          - [B, C]: 1 (constant)
          - [C, D]: 1 → 0
        
        Args:
            value: The actual measured value (timedelta for time-constraint, float for value-constraint)
        
        Returns:
            float: Compliance score in [0, 1]
        """
        # Convert timedeltas to seconds for uniform comparison
        if isinstance(value, timedelta):
            val = value.total_seconds()
            a, b, c, d = self.A.total_seconds(), self.B.total_seconds(), self.C.total_seconds(), self.D.total_seconds()
        else:
            val = float(value)
            a, b, c, d = float(self.A), float(self.B), float(self.C), float(self.D)
        
        if val < a or val > d:
            # Outside [A, D]
            return self.missing_score
        
        if b <= val <= c:
            # Between [B, C] - flat-top, score is 1.0
            return 1.0

        if a <= val <= b:
            if b == a:
                # Value exactly at A and B are the same, meaning full compliance
                return 1.0
            # Linear interpolation between A and B assuming A < B
            return (val - a) / (b - a)
        
        if c <= val <= d:
            if d == c:
                # Value exactly at C and D are the same, meaning full compliance
                return 1.0
            # Linear interpolation between C and D assuming C < D
            return (d - val) / (d - c)
        
        # Fallback (should not reach here)
        return 0.0

        
def parse_duration(duration_str):
    """
    Convert a compact duration string (e.g. '72h', '2d', '15m') into a timedelta object.

    Args:
        duration_str (str): A string with a number followed by a unit character:
                            - 's' for seconds
                            - 'm' for minutes
                            - 'h' for hours
                            - 'd' for days
                            - 'w' for weeks
                            - 'M' for months (approximated as 30 days)
                            - 'y' for years (approximated as 365 days)

    Returns:
        timedelta: A timedelta representing the duration.

    Raises:
        ValueError: If the input format is invalid or unit is unsupported.
    """
    if not duration_str or len(duration_str) < 2:
        raise ValueError(f"Invalid duration format: '{duration_str}'")
    
    try:
        duration_str = duration_str.strip()
        value = int(duration_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid numeric value in duration: '{duration_str}'")
    
    unit = duration_str[-1]
    
    # Map units to timedelta kwargs (months/years converted to days)
    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    elif unit == 'M':
        return timedelta(days=value * 30)  # approximate month as 30 days
    elif unit == 'y':
        return timedelta(days=value * 365)  # approximate year as 365 days
    else:
        raise ValueError(f"Unsupported duration unit: '{unit}'. Use s, m, h, d, w, M, or y.")
    

def apply_external_function(func_name: str, value: Union[float, str], *args) -> Union[float, str]:
    """
    Apply an external function by name to a single value, passing any additional *args.
    Used for 2 purposes:
        1. For parameterized-raw-concept values (string or float), meaning depends on data -> calculated on apply()
        2. For trapezoid values in compliance functions, meaning the nodes are static and known at parse time, but the parameter depends on the data -> calculated on apply().

    Args:
        func_name (str): The name of the external function to apply.
        value (Union[float, str]): The value to apply the function to.
        *args: Additional arguments to pass to the function.

    Returns:
        Union[float, str]: The result of applying the external function.

    Raises:
        ValueError: If the function name is not recognized or wrong number of parameters.
    """
    func = REPO.get(func_name)
    if func is None:
        raise ValueError(f"External function '{func_name}' not found in repository.")
    
    try:
        return func(value, *args)
    except TypeError as e:
        raise ValueError(f"Function '{func_name}' called with wrong number of parameters: {e}")
    except Exception as e:
        raise ValueError(f"Error applying function '{func_name}' to value '{value}': {e}")


def apply_external_function_on_trapez(func_name: str, trapez: FuzzyLogicTrapez, constraint_type: str, *args) -> FuzzyLogicTrapez:
    """
    Apply an external function by name to each value in the trapez tuple, passing any additional *args.
    For time-constraint, values are parsed as durations, converted to seconds, passed to function,
    then converted back to timedelta objects.

    Args:
        func_name (str): The name of the external function to apply.
        trapez (FuzzyLogicTrapez): Tuple of values to apply the function to.
        constraint_type (str): "time-constraint" or "value-constraint".
        *args: Additional arguments to pass to the function.

    Returns:
        FuzzyLogicTrapez: Finalized trapezoid node (A, B, C, D values as timedelta or float), after applying the external function.
        Does not modify the input trapez, returns a new one.

    Raises:
        ValueError: If the function name is not recognized or wrong number of parameters.
    """
    # Preprocess all values based on type
    if constraint_type == "time-constraint":
        # Parse all as durations and convert to seconds
        seconds = [val.total_seconds() for val in [trapez.A, trapez.B, trapez.C, trapez.D]]
        results = []
        for sec in seconds:
            res = apply_external_function(func_name, sec, *args)
            results.append(res)
        
        # Validate ordering
        if not (results[0] <= results[1] <= results[2] <= results[3]):
            raise ValueError(
                f"Function '{func_name}' did not return ordered trapez values: "
                f"Results: {results}"
            )
        
        # Convert back to timedeltas and return FuzzyLogicTrapez
        return FuzzyLogicTrapez(
            A=timedelta(seconds=results[0]),
            B=timedelta(seconds=results[1]),
            C=timedelta(seconds=results[2]),
            D=timedelta(seconds=results[3]),
            is_time=True
        )
    
    else:  # "value-constraint"
        processed = [trapez.A, trapez.B, trapez.C, trapez.D]
        results = []
        for val in processed:
            res = apply_external_function(func_name, val, *args)
            results.append(res)
        
        # Validate ordering
        if not (results[0] <= results[1] <= results[2] <= results[3]):
            raise ValueError(
                f"Function '{func_name}' did not return ordered trapez values: "
                f"Results: {results}"
            )
        
        # Return FuzzyLogicTrapez with float values
        return FuzzyLogicTrapez(A=results[0], B=results[1], C=results[2], D=results[3], is_time=False)
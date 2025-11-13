from datetime import timedelta
from dataclasses import dataclass
from typing import Union
from .external_functions import REPO


@dataclass(frozen=True)
class FuzzyLogicTrapez:
    """Immutable trapezoid node for compliance scoring.
    
    Supports both:
    - Time-based: timedelta values (A, B, C, D all as timedelta)
    - Value-based: float values (A, B, C, D all as float)
    
    Order: A <= B <= C <= D (validated at parse time).
    """
    A: Union[float, timedelta]
    B: Union[float, timedelta]
    C: Union[float, timedelta]
    D: Union[float, timedelta]
    
    def validate(self) -> None:
        """Ensure trapez is well-formed: A <= B <= C <= D."""
        if not (self.A <= self.B <= self.C <= self.D):
            raise ValueError(
                f"Invalid trapez order: A={self.A}, B={self.B}, C={self.C}, D={self.D}. "
                f"Must satisfy A <= B <= C <= D."
            )
    
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
        else:
            val = float(value)
        
        if isinstance(self.A, timedelta):
            a = self.A.total_seconds()
            b = self.B.total_seconds()
            c = self.C.total_seconds()
            d = self.D.total_seconds()
        else:
            a = float(self.A)
            b = float(self.B)
            c = float(self.C)
            d = float(self.D)
        
        if val < a or val > d:
            return 0.0
        
        if b <= val <= c:
            return 1.0
        
        if a <= val < b:
            if b == a:
                return 0.0
            return (val - a) / (b - a)
        
        if c < val <= d:
            if d == c:
                return 0.0
            return (d - val) / (d - c)
        
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
    

def apply_external_function(func_name: str, trapez: tuple, constraint_type: str, *args) -> FuzzyLogicTrapez:
    """
    Apply an external function by name to each value in the trapez tuple, passing any additional *args.
    For time-constraint, values are parsed as durations, converted to seconds, passed to function,
    then converted back to timedelta objects.

    Args:
        func_name (str): The name of the external function to apply.
        trapez (tuple): Tuple of values to apply the function to.
        constraint_type (str): "time-constraint" or "value-constraint".
        *args: Additional arguments to pass to the function.

    Returns:
        FuzzyLogicTrapez: Finalized trapezoid node (A, B, C, D values as timedelta or float)

    Raises:
        ValueError: If the function name is not recognized or wrong number of parameters.
    """
    func = REPO.get(func_name)
    if func is None:
        raise ValueError(f"External function '{func_name}' not found in repository.")

    # Preprocess all values based on type
    if constraint_type == "time-constraint":
        # Parse all as durations and convert to seconds
        seconds = [parse_duration(val).total_seconds() for val in trapez]
        results = []
        for sec in seconds:
            try:
                res = func(sec, *args)
                results.append(res)
            except TypeError as e:
                raise ValueError(f"Function '{func_name}' called with wrong number of parameters: {e}")
            except Exception as e:
                raise ValueError(f"Error applying function '{func_name}' to value '{sec}': {e}")
        
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
            D=timedelta(seconds=results[3])
        )
    
    else:  # "value-constraint"
        processed = list(trapez)
        results = []
        for val in processed:
            try:
                res = func(val, *args)
                results.append(res)
            except TypeError as e:
                raise ValueError(f"Function '{func_name}' called with wrong number of parameters: {e}")
            except Exception as e:
                raise ValueError(f"Error applying function '{func_name}' to value '{val}': {e}")
        
        # Validate ordering
        if not (results[0] <= results[1] <= results[2] <= results[3]):
            raise ValueError(
                f"Function '{func_name}' did not return ordered trapez values: "
                f"Results: {results}"
            )
        
        # Return FuzzyLogicTrapez with float values
        return FuzzyLogicTrapez(A=results[0], B=results[1], C=results[2], D=results[3])
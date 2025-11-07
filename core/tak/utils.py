from datetime import timedelta
from .tak import TrapezNode
from .external_functions import REPO


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
    

def apply_external_function(func_name: str, trapez: tuple, constraint_type: str, *args) -> TrapezNode:
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
        TrapezNode: Finalized trapezoid node (A, B, C, D values as timedelta or float)

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
        
        # Convert back to timedeltas and return TrapezNode
        return TrapezNode(
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
        
        # Return TrapezNode with float values
        return TrapezNode(A=results[0], B=results[1], C=results[2], D=results[3])
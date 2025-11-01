from datetime import timedelta

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
    

def apply_external_function(func_name: str, trapez: tuple, type: str, *args):
    """
    Apply an external function by name to each value in the trapez tuple, passing any additional *args.
    For time-constraint, values are parsed as durations and converted to seconds before applying the function.

    Args:
        func_name (str): The name of the external function to apply.
        trapez (tuple): Tuple of values to apply the function to.
        type (str): "time-constraint" or "value-constraint".
        *args: Additional arguments to pass to the function.

    Returns:
        tuple: Ordered tuple of results from applying the function to each trapez value.

    Raises:
        ValueError: If the function name is not recognized or wrong number of parameters.
    """
    func = REPO.get(func_name)
    if func is None:
        raise ValueError(f"External function '{func_name}' not found in repository.")

    # Preprocess all values based on type
    if type == "time-constraint":
        # Parse all as durations and convert to seconds
        processed = [int(parse_duration(val).total_seconds()) for val in trapez]
    else:
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
    return tuple(results)
from datetime import timedelta


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
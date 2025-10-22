from datetime import timedelta


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
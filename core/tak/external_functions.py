from typing import Dict, Callable

# Repository of external functions
REPO: Dict[str, Callable] = {}


def register(name: str, raw_concept_only: bool = False):
    """
    Decorator to register external functions.

    Args:
        name: Key under which the function is stored in REPO.
        raw_concept_only: If True, the function is only valid in parameterized-raw-concept
            context and will raise if called via apply_external_function_on_trapez.

    Usage:
        @register("my_function")
        def my_function(x, *params):
            return x * params[0]
    """
    def wrapper(func: Callable):
        func._raw_concept_only = raw_concept_only
        REPO[name] = func
        return func
    return wrapper


# --- Built-in Functions ---

@register("id")
def identity(x, *args):
    """Identity function: returns input unchanged."""
    return x


@register("mul")
def multiply(x, *args):
    """
    Multiply trapez value by all parameters.
    
    Args:
        x: Base trapez value
        *args: Parameters to multiply with
    
    Returns:
        Product of x and all parameters
    
    Example:
        mul(10, 72) → 720
        mul(0.5, 72, 1.2) → 43.2
    """
    result = x
    for arg in args:
        result *= arg
    return result


@register("div")
def divide(x, *args):
    """
    Divide trapez value by all parameters.
    
    Args:
        x: Base trapez value
        *args: Parameters to divide by
    
    Returns:
        Result of dividing x by all parameters
    
    Example:
        div(10, 2) → 5
        div(43.2, 72, 1.2) → 0.5
    """
    result = x
    for arg in args:
        if arg == 0:
            raise ValueError("Division by zero in 'div' function.")
        result /= arg
    return result


@register("add")
def add(x, *args):
    """
    Add all parameters to trapez value.
    
    Args:
        x: Base trapez value
        *args: Parameters to add
    
    Returns:
        Sum of x and all parameters
    
    Example:
        add(10, 5) → 15
        add(10, 5, 3) → 18
    """
    result = x
    for arg in args:
        result += arg
    return result

@register("subtract")
def subtract(x, *args):
    """
    Subtract all parameters from trapez value.
    
    Args:
        x: Base trapez value
        *args: Parameters to subtract
    
    Returns:
        Result of subtracting all parameters from x
    
    Example:
        add(10, 5) → 15
        add(10, 5, 3) → 18
    """
    result = x
    for arg in args:
        result -= arg
    return result


@register("id_if_thresh_met", raw_concept_only=True)
def id_if_thresh_met(value, prev_value, threshold=180, op="ge"):
    """
    Purpose: Gate function for consecutive-state detection. Returns the current value
             only if the preceding measurement satisfies the threshold condition.
             Returns None to signal that the current row should be skipped.

    Args:
        value (numeric): The current row's value (passed through if gate passes).
        prev_value (numeric): The preceding measurement's value (the gate check).
        threshold (numeric): The threshold to compare against. Defaults to 180.
        op (str): Comparison operator — 'ge' (>=) or 'le' (<=). Defaults to 'ge'.

    Returns:
        numeric: value unchanged if condition met, else None (skip signal).
    """
    if prev_value is None:
        return None
    prev = float(prev_value)
    thresh = float(threshold)
    if op == "ge":
        return value if prev >= thresh else None
    elif op == "le":
        return value if prev <= thresh else None
    else:
        raise ValueError(f"id_if_thresh_met: unknown operator '{op}', must be 'ge' or 'le'")


# --- Custom Functions (Example) ---
# Users can add their own functions below using the @register() decorator
# Functions are meant to edit the trapez node value based on additional/ dynamic parameters

# @register("custom_dosage")
# def custom_dosage_adjustment(x, *params):
#     """
#     Example custom function: adjust dosage based on weight and age.
#     Args:
#         x: Base dosage (one of the trapez nodes)
#         *params: [weight (kg), age (years)]
#     Returns:
#         Adjusted dosage
#     """
#     weight = params[0]
#     age = params[1]
#     adjusted = x * weight
#     if age > 65:
#         adjusted *= 0.8  # 20% reduction for elderly patients
#     return adjusted
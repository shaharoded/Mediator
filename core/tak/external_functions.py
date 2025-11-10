from typing import Dict, Callable

# Repository of external functions
REPO: Dict[str, Callable] = {}


def register(name: str):
    """
    Decorator to register external functions for compliance calculations.
    
    Usage:
        @register("my_function")
        def my_function(x, *params):
            return x * params[0]
    """
    def wrapper(func: Callable):
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
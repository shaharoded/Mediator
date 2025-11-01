"""
TO-DO:
- build a wrapper that checks their number of arguments compared to TAK
- build a wrapper (utils.py) that can handle the application of them on numeric values or time values
"""

def id(x):
    """
    Identity function that returns the argument as is.
    """
    return x

def mul(*args):
    """
    Multiplies all the arguments together.
    """
    result = 1
    for arg in args:
        result *= arg
    return result

def add(*args):
    """
    Adds all the arguments together.
    """
    result = 0
    for arg in args:
        result += arg
    return result

REPO = {
    "id": id,
    "mul": mul,
    "add": add
}
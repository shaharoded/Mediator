"""
Shared utilities for unit tests.

Provides common functions for:
- Writing XML files to disk
- Creating test timestamps
- Building test DataFrames
"""
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    """Write XML string to disk and return path."""
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


def make_ts(hhmm: str, day: int = 0) -> datetime:
    """
    Build timestamp: 2024-01-01 + day offset + HH:MM.
    
    Args:
        hhmm: Time string in "HH:MM" format (e.g., "08:00")
        day: Day offset from base date (default: 0)
        
    Returns:
        datetime object
        
    Example:
        make_ts("08:00") -> 2024-01-01 08:00:00
        make_ts("14:30", day=1) -> 2024-01-02 14:30:00
    """
    base = datetime(2024, 1, 1) + timedelta(days=day)
    hh, mm = map(int, hhmm.split(":"))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)

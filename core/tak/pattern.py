from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .tak import TAK, get_tak_repository
from .event import Event
from .context import Context
from .utils import parse_duration


class Pattern(TAK):
    pass
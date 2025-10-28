import os

# Get project root (two levels up from core/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Knowledge Base (TAK definitions)
TAK_FOLDER = os.path.join(PROJECT_ROOT, 'core', 'knowledge-base')

# XSD Schema (required for TAK validation)
TAK_SCHEMA_PATH = os.path.join(TAK_FOLDER, 'tak_schema.xsd')
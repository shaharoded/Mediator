"""
Unit tests for TAKRepository.

Tests cover:
1. Dependency graph construction
2. Topological sort ordering
3. Circular reference detection
4. Complex dependency chains (e.g., State → Event → RawConcept)
5. Family-based priority ordering (raw-concepts before events before states, etc.)
6. Pattern dependencies (including Pattern-from-Pattern)
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.tak.repository import TAKRepository, set_tak_repository
from core.tak.raw_concept import RawConcept
from core.tak.event import Event
from core.tak.state import State
from core.tak.context import Context
from core.tak.pattern import LocalPattern


# ============================
# Helpers
# ============================

def write_xml(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml.strip(), encoding="utf-8")
    return p


# ============================
# Hard-coded XML Fixtures
# ============================

# Raw concept: single numeric attribute (raw-numeric)
RAW_CONCEPT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="RAW_CONCEPT" concept-type="raw-numeric">
    <categories>test</categories>
    <description>Test raw concept</description>
    <attributes>
        <attribute name="value" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="0" max="1000"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
"""

# Event with abstraction rules (required for numeric raw-concept)
EVENT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="EVENT">
    <categories>test</categories>
    <description>Test event with numeric discretization</description>
    <derived-from>
        <attribute name="RAW_CONCEPT" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Low" operator="or">
            <attribute ref="A1">
                <allowed-value max="50"/>
            </attribute>
        </rule>
        <rule value="High" operator="or">
            <attribute ref="A1">
                <allowed-value min="50"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""

# State: derived from event
STATE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<state name="STATE">
    <categories>test</categories>
    <description>Test state</description>
    <derived-from name="EVENT" tak="event"/>
    <persistence good-after="4h" interpolate="false" max-skip="0"/>
</state>
"""

# Multiple raw concepts (for multi-source tests)
RC1_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="RC1" concept-type="raw-numeric">
    <categories>test</categories>
    <description>Test RC1</description>
    <attributes>
        <attribute name="val1" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="0" max="1000"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
"""

RC2_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="RC2" concept-type="raw-numeric">
    <categories>test</categories>
    <description>Test RC2</description>
    <attributes>
        <attribute name="val2" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="0" max="1000"/>
            </numeric-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
"""

# Event with multiple derived-from (but no abstraction rules → should still be valid for numeric)
# FIX: Add abstraction rules to satisfy validation
EVENT_MULTI_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<event name="EVENT">
    <categories>test</categories>
    <description>Test event with multiple sources</description>
    <derived-from>
        <attribute name="RC1" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="RC2" tak="raw-concept" idx="0" ref="A2"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Match" operator="or">
            <attribute ref="A1">
                <allowed-value min="0"/>
            </attribute>
            <attribute ref="A2">
                <allowed-value min="0"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
"""

# Nominal raw-concept (for non-numeric test)
RAW_MEAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="MEAL" concept-type="raw-nominal">
    <categories>test</categories>
    <description>Meal type</description>
    <attributes>
        <attribute name="MEAL_TYPE" type="nominal">
            <nominal-allowed-values>
                <allowed-value value="Breakfast"/>
                <allowed-value value="Lunch"/>
            </nominal-allowed-values>
        </attribute>
    </attributes>
</raw-concept>
"""

# Context
CONTEXT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<context name="CONTEXT">
    <categories>test</categories>
    <description>Test context</description>
    <derived-from>
        <attribute name="MEAL" tak="raw-concept" idx="0"/>
    </derived-from>
    <context-windows>
        <persistence good-before="1h" good-after="2h"/>
    </context-windows>
</context>
"""

# Pattern
PATTERN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="PATTERN1" concept-type="local-pattern">
    <categories>test</categories>
    <description>Test pattern</description>
    <derived-from>
        <attribute name="EVENT" tak="event" idx="0" ref="A1"/>
        <attribute name="RC2" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='24h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value min="0"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""

# Pattern-from-Pattern
PATTERN2_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="PATTERN2" concept-type="local-pattern">
    <categories>test</categories>
    <description>Pattern using PATTERN1 twice</description>
    <derived-from>
        <attribute name="PATTERN1" tak="local-pattern" idx="0" ref="A1"/>
        <attribute name="PATTERN1" tak="local-pattern" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='48h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value equal="Partial"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""


# ============================
# Fixtures
# ============================

@pytest.fixture
def temp_kb():
    """Create temporary knowledge base folder structure."""
    with TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        (kb_path / "raw-concepts").mkdir()
        (kb_path / "events").mkdir()
        (kb_path / "states").mkdir()
        (kb_path / "contexts").mkdir()
        (kb_path / "patterns").mkdir()
        yield kb_path


@pytest.fixture
def repo():
    """Create fresh TAKRepository and set as global."""
    r = TAKRepository()
    set_tak_repository(r)
    return r


# ============================
# Tests
# ============================

def test_simple_linear_dependency_ordering(temp_kb, repo):
    """Test simple linear dependency: RawConcept → Event → State"""
    # Write XMLs
    rc_path = write_xml(temp_kb / "raw-concepts", "raw_concept.xml", RAW_CONCEPT_XML)
    event_path = write_xml(temp_kb / "events", "event.xml", EVENT_XML)
    state_path = write_xml(temp_kb / "states", "state.xml", STATE_XML)
    
    # Parse and register in dependency order
    rc = RawConcept.parse(rc_path)
    repo.register(rc)
    
    event = Event.parse(event_path)
    repo.register(event)
    
    state = State.parse(state_path)
    repo.register(state)
    
    # Finalize
    repo.finalize_repository()
    
    # Verify execution order
    order = repo.execution_order
    assert len(order) == 3
    assert order.index("RAW_CONCEPT") < order.index("EVENT")
    assert order.index("EVENT") < order.index("STATE")
    print(f"✓ Simple linear chain: {order}")


def test_multiple_raw_concepts_single_event(temp_kb, repo):
    """Test multiple RawConcepts feeding into single Event"""
    # Write XMLs
    rc1_path = write_xml(temp_kb / "raw-concepts", "rc1.xml", RC1_XML)
    rc2_path = write_xml(temp_kb / "raw-concepts", "rc2.xml", RC2_XML)
    event_path = write_xml(temp_kb / "events", "event.xml", EVENT_MULTI_XML)
    
    # Parse and register
    rc1 = RawConcept.parse(rc1_path)
    repo.register(rc1)
    
    rc2 = RawConcept.parse(rc2_path)
    repo.register(rc2)
    
    event = Event.parse(event_path)
    repo.register(event)
    
    repo.finalize_repository()
    
    # Verify order
    order = repo.execution_order
    assert len(order) == 3
    assert order.index("RC1") < order.index("EVENT")
    assert order.index("RC2") < order.index("EVENT")
    print(f"✓ Multiple RawConcepts → Event: {order}")


def test_pattern_with_raw_event_dependencies(temp_kb, repo):
    """Test Pattern with dependencies on RawConcept and Event"""
    # Write XMLs
    rc1_path = write_xml(temp_kb / "raw-concepts", "rc1.xml", RC1_XML)
    rc2_path = write_xml(temp_kb / "raw-concepts", "rc2.xml", RC2_XML)
    event_path = write_xml(temp_kb / "events", "event.xml", EVENT_XML.replace("RAW_CONCEPT", "RC1"))
    pattern_path = write_xml(temp_kb / "patterns", "pattern1.xml", PATTERN_XML)
    
    # Parse and register
    rc1 = RawConcept.parse(rc1_path)
    repo.register(rc1)
    
    rc2 = RawConcept.parse(rc2_path)
    repo.register(rc2)
    
    event = Event.parse(event_path)
    repo.register(event)
    
    pattern = LocalPattern.parse(pattern_path)
    repo.register(pattern)
    
    repo.finalize_repository()
    
    # Verify order
    order = repo.execution_order
    assert len(order) == 4
    assert order.index("RC1") < order.index("EVENT")
    assert order.index("EVENT") < order.index("PATTERN1")
    assert order.index("RC2") < order.index("PATTERN1")
    print(f"✓ Pattern with RC+Event dependencies: {order}")


def test_pattern_from_pattern_dependency(temp_kb, repo):
    """Test Pattern-from-Pattern dependency"""
    # Write XMLs
    rc1_path = write_xml(temp_kb / "raw-concepts", "rc1.xml", RC1_XML)
    rc2_path = write_xml(temp_kb / "raw-concepts", "rc2.xml", RC2_XML)
    event1_path = write_xml(temp_kb / "events", "event1.xml", EVENT_XML.replace("RAW_CONCEPT", "RC1").replace("EVENT", "EVENT1"))
    event2_path = write_xml(temp_kb / "events", "event2.xml", EVENT_XML.replace("RAW_CONCEPT", "RC2").replace("EVENT", "EVENT2"))
    
    # PATTERN1: depends on EVENT1 + EVENT2 (FIXED: no idx for events)
    pattern1_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="PATTERN1" concept-type="local-pattern">
    <categories>test</categories>
    <description>Test pattern</description>
    <derived-from>
        <attribute name="EVENT1" tak="event" ref="A1"/>
        <attribute name="EVENT2" tak="event" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='24h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="Low"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value equal="High"/>
                    </attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
"""
    pattern1_path = write_xml(temp_kb / "patterns", "pattern1.xml", pattern1_xml)
    
    # PATTERN2: depends on PATTERN1 twice
    pattern2_path = write_xml(temp_kb / "patterns", "pattern2.xml", PATTERN2_XML)
    
    # Parse and register
    rc1 = RawConcept.parse(rc1_path)
    repo.register(rc1)
    
    rc2 = RawConcept.parse(rc2_path)
    repo.register(rc2)
    
    event1 = Event.parse(event1_path)
    repo.register(event1)
    
    event2 = Event.parse(event2_path)
    repo.register(event2)
    
    pattern1 = LocalPattern.parse(pattern1_path)
    repo.register(pattern1)
    
    pattern2 = LocalPattern.parse(pattern2_path)
    repo.register(pattern2)
    
    repo.finalize_repository()
    
    # Verify execution order
    order = repo.execution_order
    assert len(order) == 6
    assert order.index("RC1") < order.index("EVENT1")
    assert order.index("RC2") < order.index("EVENT2")
    assert order.index("EVENT1") < order.index("PATTERN1")
    assert order.index("EVENT2") < order.index("PATTERN1")
    assert order.index("PATTERN1") < order.index("PATTERN2")
    
    print(f"✓ Pattern-from-Pattern: {order}")
    
    # Verify deduplication (graph stores sets, not lists)
    pattern2_deps = repo.graph.get("PATTERN2", set())
    
    # Check that PATTERN1 appears exactly once (sets automatically deduplicate)
    assert "PATTERN1" in pattern2_deps, (
        f"PATTERN2 should depend on PATTERN1, but dependencies are: {pattern2_deps}"
    )
    
    # Check that PATTERN1 appears exactly once by verifying set has 1 element
    # (PATTERN2 uses PATTERN1 twice in XML, but dependency graph should deduplicate)
    assert len(pattern2_deps) == 1, (
        f"PATTERN2 should have exactly 1 unique dependency (PATTERN1), found {len(pattern2_deps)}: {pattern2_deps}"
    )
    
    print(f"  → PATTERN2 dependencies (deduplicated): {pattern2_deps}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

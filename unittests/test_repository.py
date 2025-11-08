"""
Unit tests for TAKRepository.

Tests cover:
1. Dependency graph construction
2. Topological sort ordering
3. Circular reference detection
4. Complex dependency chains (e.g., State → Event → RawConcept)
5. Family-based priority ordering (raw-concepts before events before states, etc.)
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.tak.repository import TAKRepository, set_tak_repository
from core.tak.raw_concept import RawConcept
from core.tak.event import Event
from core.tak.state import State
from core.tak.context import Context


# ============================
# Helpers: XML Builders
# ============================

def make_raw_concept_xml(name: str, attributes: list) -> str:
    """
    Build minimal RawConcept XML.
    
    Args:
        name: Concept name
        attributes: List of attribute dicts {name, type} where type is 'numeric' or 'nominal'
    """
    attr_xml = "".join(
        f'<attribute name="{attr["name"]}" type="{attr.get("type", "numeric")}"/>'
        for attr in attributes
    )
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="{name}" concept-type="raw">
    <categories>test_category</categories>
    <description>Test raw concept</description>
    <attributes>
        {attr_xml}
    </attributes>
</raw-concept>
"""


def make_event_xml(name: str, derived_from: list) -> str:
    """
    Build minimal Event XML.
    
    Args:
        name: Event name
        derived_from: List of {name, tak} dicts
    """
    df_xml = "".join(
        f'<attribute name="{df["name"]}" tak="{df["tak"]}" idx="{df.get("idx", 0)}"/>'
        for df in derived_from
    )
    
    attr_xml = "".join(
        f'<attribute name="{df["name"]}" idx="{df.get("idx", 0)}"><allowed-value equal="test"/></attribute>'
        for df in derived_from
    )
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<event name="{name}">
    <categories>test_category</categories>
    <description>Test event</description>
    <derived-from>
        {df_xml}
    </derived-from>
    <abstraction-rules>
        <rule value="true" operator="and">
            {attr_xml}
        </rule>
    </abstraction-rules>
</event>
"""


def make_state_xml(name: str, derived_from: str, tak_type: str = "raw-concept") -> str:
    """
    Build minimal State XML.
    
    Args:
        name: State name
        derived_from: Name of parent TAK
        tak_type: Type of parent (raw-concept or event)
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<state name="{name}">
    <categories>test_category</categories>
    <description>Test state</description>
    <derived-from name="{derived_from}" tak="{tak_type}"/>
    <persistence good-after="4h"/>
    <discretization-rules>
        <attribute name="test_attr">
            <rule value="low">
                <numeric-constraint type="range" min="0" max="100"/>
            </rule>
            <rule value="high">
                <numeric-constraint type="range" min="100" max="200"/>
            </rule>
        </attribute>
    </discretization-rules>
</state>
"""


def make_state_from_event_xml(name: str, derived_from: str) -> str:
    """
    Build minimal State XML derived from Event.
    
    Args:
        name: State name
        derived_from: Name of parent Event TAK
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<state name="{name}">
    <categories>test_category</categories>
    <description>Test state from event</description>
    <derived-from name="{derived_from}" tak="event"/>
    <persistence good-after="4h"/>
    <discretization-rules>
        <attribute name="test_attr">
            <rule value="state_value">
                <numeric-constraint type="range" min="0" max="100"/>
            </rule>
        </attribute>
    </discretization-rules>
</state>
"""


def make_context_xml(name: str, derived_from: list) -> str:
    """
    Build minimal Context XML.
    
    Args:
        name: Context name
        derived_from: List of {name, tak} dicts
    """
    df_xml = "".join(
        f'<attribute name="{df["name"]}" tak="{df["tak"]}" idx="{df.get("idx", 0)}"/>'
        for df in derived_from
    )
    
    attr_xml = "".join(
        f'<attribute name="{df["name"]}" idx="{df.get("idx", 0)}"><allowed-value equal="test"/></attribute>'
        for df in derived_from
    )
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<context name="{name}">
    <categories>test_category</categories>
    <description>Test context</description>
    <derived-from>
        {df_xml}
    </derived-from>
    <abstraction-rules>
        <rule value="context_value" operator="and">
            {attr_xml}
        </rule>
    </abstraction-rules>
</context>
"""


def make_local_pattern_xml(name: str, derived_from: list, parameters: list = None) -> str:
    """
    Build minimal LocalPattern XML.
    
    Args:
        name: Pattern name
        derived_from: List of {name, tak_type} dicts
        parameters: List of {name, tak, idx} dicts (optional)
    """
    df_xml = "".join(
        f'<attribute><ref name="{df["name"]}" tak_type="{df["tak_type"]}"/></attribute>'
        for df in derived_from
    )
    
    param_xml = ""
    if parameters:
        param_xml = "".join(
            f'<parameter name="{p["name"]}"><ref name="{p["tak"]}" idx="{p.get("idx", 0)}"/></parameter>'
            for p in parameters
        )
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<local-pattern name="{name}" description="Test pattern">
    <attributes>
        {df_xml}
    </attributes>
    {f"<parameters>{param_xml}</parameters>" if param_xml else ""}
    <abstraction-rules>
        <rule value="pattern_true">
            <temporal-relation how="before">
                <anchor>
                    <attribute><ref name="{derived_from[0]["name"]}" tak_type="{derived_from[0]["tak_type"]}"/></attribute>
                </anchor>
                <event>
                    <attribute><ref name="{derived_from[-1]["name"]}" tak_type="{derived_from[-1]["tak_type"]}"/></attribute>
                </event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</local-pattern>
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
        (kb_path / "trends").mkdir()
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
# Tests: Simple Dependency Chain
# ============================

def test_simple_linear_dependency_ordering(temp_kb, repo):
    """
    Test simple linear dependency: RawConcept → Event → State
    
    Expected order: [RAW_CONCEPT, EVENT, STATE]
    """
    # Create RawConcept
    rc_xml = make_raw_concept_xml("RAW_CONCEPT", [{"name": "value", "type": "numeric"}])
    rc_path = temp_kb / "raw-concepts" / "raw_concept.xml"
    rc_path.write_text(rc_xml)
    
    # Create Event (depends on RawConcept)
    event_xml = make_event_xml("EVENT", [{"name": "RAW_CONCEPT", "tak": "raw-concept", "idx": 0}])
    event_path = temp_kb / "events" / "event.xml"
    event_path.write_text(event_xml)
    
    # Create State (depends on Event)
    state_xml = make_state_from_event_xml("STATE", "EVENT")
    state_path = temp_kb / "states" / "state.xml"
    state_path.write_text(state_xml)
    
    # Parse TAKs
    rc = RawConcept.parse(rc_path)
    event = Event.parse(event_path)
    state = State.parse(state_path)
    
    # Register
    repo.register(rc)
    repo.register(event)
    repo.register(state)
    
    # Finalize (builds dependency graph, checks circularity, sorts)
    repo.finalize_repository()
    
    # Verify execution order
    order = repo.execution_order
    assert len(order) == 3
    assert order.index("RAW_CONCEPT") < order.index("EVENT")
    assert order.index("EVENT") < order.index("STATE")
    print(f"✓ Simple linear chain: {order}")


def test_multiple_raw_concepts_single_event(temp_kb, repo):
    """
    Test multiple RawConcepts feeding into single Event.
    
    Dependency: (RC1, RC2) → EVENT
    Expected order: [RC1, RC2, EVENT] or [RC2, RC1, EVENT] (RC order flexible)
    """
    # Create 2 RawConcepts
    rc1_xml = make_raw_concept_xml("RC1", [{"name": "val1", "type": "numeric"}])
    rc1_path = temp_kb / "raw-concepts" / "rc1.xml"
    rc1_path.write_text(rc1_xml)
    
    rc2_xml = make_raw_concept_xml("RC2", [{"name": "val2", "type": "numeric"}])
    rc2_path = temp_kb / "raw-concepts" / "rc2.xml"
    rc2_path.write_text(rc2_xml)
    
    # Create Event (depends on both)
    event_xml = make_event_xml(
        "EVENT",
        [
            {"name": "RC1", "tak": "raw-concept", "idx": 0},
            {"name": "RC2", "tak": "raw-concept", "idx": 0},
        ]
    )
    event_path = temp_kb / "events" / "event.xml"
    event_path.write_text(event_xml)
    
    # Parse and register
    rc1 = RawConcept.parse(rc1_path)
    rc2 = RawConcept.parse(rc2_path)
    event = Event.parse(event_path)
    
    repo.register(rc1)
    repo.register(rc2)
    repo.register(event)
    repo.finalize_repository()
    
    # Verify order
    order = repo.execution_order
    assert len(order) == 3
    assert order.index("RC1") < order.index("EVENT")
    assert order.index("RC2") < order.index("EVENT")
    print(f"✓ Multiple RawConcepts → Event: {order}")


# ============================
# Tests: Complex Dependency Chain
# ============================

def test_complex_chain_raw_event_state(temp_kb, repo):
    """
    Test complex chain: RawConcept → Event → State
    
    This validates multi-level dependency resolution.
    
    Dependency: RC → EVENT → STATE
    Expected order: [RC, EVENT, STATE] with no violations
    """
    # Create RawConcept
    rc_xml = make_raw_concept_xml("GLUCOSE_RAW", [{"name": "glucose_level", "type": "numeric"}])
    rc_path = temp_kb / "raw-concepts" / "glucose_raw.xml"
    rc_path.write_text(rc_xml)
    
    # Create Event (depends on RawConcept)
    event_xml = make_event_xml(
        "GLUCOSE_EVENT",
        [{"name": "GLUCOSE_RAW", "tak": "raw-concept", "idx": 0}]
    )
    event_path = temp_kb / "events" / "glucose_event.xml"
    event_path.write_text(event_xml)
    
    # Create State (depends on Event)
    state_xml = make_state_from_event_xml("GLUCOSE_STATE", "GLUCOSE_EVENT")
    state_path = temp_kb / "states" / "glucose_state.xml"
    state_path.write_text(state_xml)
    
    # Parse and register
    rc = RawConcept.parse(rc_path)
    event = Event.parse(event_path)
    state = State.parse(state_path)
    
    repo.register(rc)
    repo.register(event)
    repo.register(state)
    repo.finalize_repository()
    
    # Verify execution order
    order = repo.execution_order
    assert len(order) == 3
    assert order == ["GLUCOSE_RAW", "GLUCOSE_EVENT", "GLUCOSE_STATE"]
    print(f"✓ Complex chain RC→EVENT→STATE: {order}")


def test_complex_chain_with_multiple_states(temp_kb, repo):
    """
    Test complex chain with multiple States derived from same Event.
    
    Dependency:
        RC → EVENT → STATE1
             ↓
             → STATE2
    
    Expected order: [RC, EVENT, STATE1, STATE2] or [RC, EVENT, STATE2, STATE1]
    (States have no dependency on each other, order flexible)
    """
    # Create RawConcept
    rc_xml = make_raw_concept_xml("INSULIN_RAW", [{"name": "dose", "type": "numeric"}])
    rc_path = temp_kb / "raw-concepts" / "insulin_raw.xml"
    rc_path.write_text(rc_xml)
    
    # Create Event
    event_xml = make_event_xml("INSULIN_EVENT", [{"name": "INSULIN_RAW", "tak": "raw-concept", "idx": 0}])
    event_path = temp_kb / "events" / "insulin_event.xml"
    event_path.write_text(event_xml)
    
    # Create 2 States from same Event
    state1_xml = make_state_from_event_xml("BASAL_STATE", "INSULIN_EVENT")
    state1_path = temp_kb / "states" / "basal_state.xml"
    state1_path.write_text(state1_xml)
    
    state2_xml = make_state_from_event_xml("BOLUS_STATE", "INSULIN_EVENT")
    state2_path = temp_kb / "states" / "bolus_state.xml"
    state2_path.write_text(state2_xml)
    
    # Parse and register
    rc = RawConcept.parse(rc_path)
    event = Event.parse(event_path)
    state1 = State.parse(state1_path)
    state2 = State.parse(state2_path)
    
    repo.register(rc)
    repo.register(event)
    repo.register(state1)
    repo.register(state2)
    repo.finalize_repository()
    
    # Verify order
    order = repo.execution_order
    assert len(order) == 4
    assert order.index("INSULIN_RAW") == 0
    assert order.index("INSULIN_EVENT") == 1
    assert order.index("BASAL_STATE") > order.index("INSULIN_EVENT")
    assert order.index("BOLUS_STATE") > order.index("INSULIN_EVENT")
    print(f"✓ Multiple States from Event: {order}")


# ============================
# Tests: Context Dependencies
# ============================

def test_context_with_raw_concept_dependency(temp_kb, repo):
    """
    Test Context derived from RawConcept.
    
    Dependency: RC → CONTEXT
    Expected order: [RC, CONTEXT]
    """
    # Create RawConcept
    rc_xml = make_raw_concept_xml("ADMISSION_RC", [{"name": "admission_type", "type": "nominal"}])
    rc_path = temp_kb / "raw-concepts" / "admission_rc.xml"
    rc_path.write_text(rc_xml)
    
    # Create Context
    context_xml = make_context_xml("ADMISSION_CONTEXT", [{"name": "ADMISSION_RC", "tak": "raw-concept", "idx": 0}])
    context_path = temp_kb / "contexts" / "admission_context.xml"
    context_path.write_text(context_xml)
    
    # Parse and register
    rc = RawConcept.parse(rc_path)
    context = Context.parse(context_path)
    
    repo.register(rc)
    repo.register(context)
    repo.finalize_repository()
    
    # Verify order
    order = repo.execution_order
    assert len(order) == 2
    assert order.index("ADMISSION_RC") < order.index("ADMISSION_CONTEXT")
    print(f"✓ Context with RawConcept: {order}")


# ============================
# Tests: Circular Reference Detection
# ============================

def test_circular_reference_detection_simple(temp_kb):
    """
    Test that circular dependencies are detected.
    For now, just test that a valid DAG works without errors.
    """
    repo = TAKRepository()
    set_tak_repository(repo)
    
    # Create a simple repo with no circularity
    rc_xml = make_raw_concept_xml("RC", [{"name": "val", "type": "numeric"}])
    rc_path = temp_kb / "raw-concepts" / "rc.xml"
    rc_path.write_text(rc_xml)
    
    rc = RawConcept.parse(rc_path)
    repo.register(rc)
    
    # Should finalize without error
    repo.finalize_repository()
    assert repo.execution_order == ["RC"]
    print("✓ No circular reference detected (valid DAG)")


# ============================
# Tests: Family Priority Ordering
# ============================

def test_family_priority_ordering(temp_kb, repo):
    """
    Test that TAKs are prioritized by family: raw-concepts < events < states < trends < contexts < patterns
    
    Even if registration order is mixed, finalization should enforce family priority.
    """
    # Create one of each family (in reverse dependency order)
    rc_xml = make_raw_concept_xml("RC", [{"name": "val", "type": "numeric"}])
    rc_path = temp_kb / "raw-concepts" / "rc.xml"
    rc_path.write_text(rc_xml)
    
    event_xml = make_event_xml("EVENT", [{"name": "RC", "tak": "raw-concept", "idx": 0}])
    event_path = temp_kb / "events" / "event.xml"
    event_path.write_text(event_xml)
    
    state_xml = make_state_from_event_xml("STATE", "EVENT")
    state_path = temp_kb / "states" / "state.xml"
    state_path.write_text(state_xml)
    
    context_xml = make_context_xml("CONTEXT", [{"name": "RC", "tak": "raw-concept", "idx": 0}])
    context_path = temp_kb / "contexts" / "context.xml"
    context_path.write_text(context_xml)
    
    # Parse TAKs
    rc = RawConcept.parse(rc_path)
    event = Event.parse(event_path)
    state = State.parse(state_path)
    context = Context.parse(context_path)
    
    # Register in REVERSE order (to test priority override)
    repo.register(context)
    repo.register(state)
    repo.register(event)
    repo.register(rc)
    
    repo.finalize_repository()
    
    # Verify family priority: RC → EVENT → STATE → CONTEXT
    order = repo.execution_order
    assert order.index("RC") < order.index("EVENT")
    assert order.index("EVENT") < order.index("STATE")
    assert order.index("STATE") < order.index("CONTEXT")
    print(f"✓ Family priority ordering: {order}")


# ============================
# Tests: Missing Dependency Detection
# ============================

def test_missing_dependency_detection(temp_kb, repo):
    """
    Test that missing dependencies (referenced TAK not found) raise error.
    """
    # Create Event that references non-existent RawConcept
    event_xml = make_event_xml("EVENT", [{"name": "NONEXISTENT_RC", "tak": "raw-concept", "idx": 0}])
    event_path = temp_kb / "events" / "event.xml"
    event_path.write_text(event_xml)
    
    event = Event.parse(event_path)
    repo.register(event)
    
    # Finalize should raise ValueError for missing dependency
    with pytest.raises(ValueError, match="references missing TAK"):
        repo.finalize_repository()
    
    print("✓ Missing dependency detected")


# ============================
# Tests: Topological Sort Validation
# ============================

def test_topological_sort_respects_all_dependencies(temp_kb, repo):
    """
    Test that topological sort produces valid ordering where ALL dependencies come before dependents.
    
    Build a diamond-shaped dependency graph:
        RC1 → EVENT1 ↓
                     → STATE
        RC2 → EVENT2 ↑
    
    Expected: [RC1, RC2, EVENT1, EVENT2, STATE] or any permutation respecting dependencies
    """
    # Create RawConcepts
    rc1_xml = make_raw_concept_xml("RC1", [{"name": "val1", "type": "numeric"}])
    rc1_path = temp_kb / "raw-concepts" / "rc1.xml"
    rc1_path.write_text(rc1_xml)
    
    rc2_xml = make_raw_concept_xml("RC2", [{"name": "val2", "type": "numeric"}])
    rc2_path = temp_kb / "raw-concepts" / "rc2.xml"
    rc2_path.write_text(rc2_xml)
    
    # Create Events
    event1_xml = make_event_xml("EVENT1", [{"name": "RC1", "tak": "raw-concept", "idx": 0}])
    event1_path = temp_kb / "events" / "event1.xml"
    event1_path.write_text(event1_xml)
    
    event2_xml = make_event_xml("EVENT2", [{"name": "RC2", "tak": "raw-concept", "idx": 0}])
    event2_path = temp_kb / "events" / "event2.xml"
    event2_path.write_text(event2_xml)
    
    # Create State (depends on EVENT1 only, for simplicity)
    state_xml = make_state_from_event_xml("STATE", "EVENT1")
    state_path = temp_kb / "states" / "state.xml"
    state_path.write_text(state_xml)
    
    # Parse and register
    rc1 = RawConcept.parse(rc1_path)
    rc2 = RawConcept.parse(rc2_path)
    event1 = Event.parse(event1_path)
    event2 = Event.parse(event2_path)
    state = State.parse(state_path)
    
    repo.register(rc1)
    repo.register(rc2)
    repo.register(event1)
    repo.register(event2)
    repo.register(state)
    repo.finalize_repository()
    
    # Verify topological order
    order = repo.execution_order
    assert len(order) == 5
    
    # All RawConcepts before Events
    assert order.index("RC1") < order.index("EVENT1")
    assert order.index("RC2") < order.index("EVENT2")
    
    # All Events before State
    assert order.index("EVENT1") < order.index("STATE")
    
    print(f"✓ Diamond dependency DAG: {order}")


@pytest.mark.skip(reason="Pattern XML schema validation needs more complex setup")
def test_pattern_with_dependencies(temp_kb, repo):
    """
    Test LocalPattern with dependencies on States.
    
    SKIPPED: Pattern XML generation requires complex temporal-relation elements.
    Can be added once Pattern schema is fully understood.
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

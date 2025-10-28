# TAK (Temporal Abstraction Knowledge) Documentation

## Table of Contents
1. [Overview](#overview)
2. [TAK Families](#tak-families)
3. [Schema Reference](#schema-reference)
4. [Usage Examples](#usage-examples)
5. [Validation](#validation)

---

## Overview

TAKs (Temporal Abstraction Knowledge) are XML-based definitions that transform raw clinical data into higher-level abstractions. The Mediator processes TAKs in dependency order:

```
Raw Concepts → Events → States → Trends → Contexts → Patterns
```

Each TAK family has specific XML schema requirements and semantic rules.

---

## TAK Families

### 1. Raw Concepts
**Purpose:** Bridge between InputPatientData and abstraction pipeline.

**Types:**
- `raw` — Multi-attribute concepts with tuple merging (e.g., medication dosage + route)
- `raw-numeric` — Single numeric attributes with range validation
- `raw-nominal` — Single nominal attributes with allowed values
- `raw-boolean` — Boolean flags (presence/absence)

**Key Parameters:**
- `concept-type` — One of: `raw`, `raw-numeric`, `raw-nominal`, `raw-boolean`
- `<tuple-order>` — (raw only) Defines attribute order in output tuples
- `<merge require-all>` — (raw only) Whether to require all attributes for tuple emission
- `<attributes>` - Define the str names of values in column **ConceptName** which will be converted to this raw concept.

---

### 2. Events
**Purpose:** Point-in-time occurrences derived from raw-concepts.

**Key Parameters:**
- `<derived-from>` — List of raw-concepts (can bridge multiple sources)
- `<abstraction-rules>` — Optional constraints: `equal`, `min`, `max`, `min+max` (range)
- `operator` — `or` (default) or `and` for multi-attribute rules

**Output:** Point events (StartDateTime = EndDateTime)

---

### 3. States
**Purpose:** Symbolic intervals derived from numeric/nominal concepts via discretization.

**Key Parameters:**
- `<derived-from>` — Single raw-concept or event
- `<discretization-rules>` — Map numeric ranges → discrete labels (e.g., "Low", "High")
- `<abstraction-rules>` — Combine discrete attributes → final state labels
- `<persistence>` — Interval merging: `good-after`, `interpolate`, `max-skip`

**Output:** Intervals with symbolic values

---

### 4. Trends
**Purpose:** Compute local slopes (Increasing/Decreasing/Steady) over time windows.

**Key Parameters:**
- `<derived-from>` — Single raw-concept (numeric attribute)
- `significant-variation` — Threshold for trend detection
- `<time-steady>` — Lookback window for slope calculation. All points in this window are collected.
- `<persistence good-after>` — Maximum gap for interval stretching

generally, `<time-steady>` << `<persistence good-after>`.

An interval is not steady if it's linear regression slope * time_steady > significant-variation.

**Output:** Intervals with trend labels

---

### 5. Contexts
**Purpose:** Background facts with interval windowing and clipping. Very similar to Events otherwise.

**Key Parameters:**
- `<derived-from>` — List of raw-concepts
- `<context-windows>` — Per-value windows: `good-before`, `good-after`
- `<clippers>` — External events that trim intervals (e.g., ADMISSION, DEATH)

**Output:** Windowed intervals (can vary by abstraction value)

---

## Schema Reference

### Duration Strings
Format: `<number><unit>` (e.g., `15m`, `24h`, `3d`)

**Units:**
- `s` — seconds
- `m` — minutes
- `h` — hours
- `d` — days
- `w` — weeks
- `M` — months (30 days)
- `y` — years (365 days)

---

### Common Attributes

#### `<attribute>` (in raw-concepts)
```xml
<attribute name="GLUCOSE_VALUE" type="numeric">
    <numeric-allowed-values>
        <allowed-value min="0" max="600"/>
    </numeric-allowed-values>
</attribute>
```

**Parameters:**
- `name` — Unique identifier
- `type` — `numeric`, `nominal`, or `boolean`
- `min` / `max` — (numeric) Valid range
- `<allowed-value value="...">` — (nominal) Enumerated values

---

#### `<abstraction-rules>` (Events/Contexts)
```xml
<abstraction-rules>
    <rule value="Hypoglycemia" operator="or">
        <attribute name="GLUCOSE_MEASURE" idx="0">
            <allowed-value max="70"/>  <!-- <= 70 -->
        </attribute>
        <attribute name="HYPOGLYCEMIA" idx="0">
            <allowed-value equal="True"/>  <!-- exact match -->
        </attribute>
    </rule>
</abstraction-rules>
```

**Constraint types:**
- `equal="value"` — Exact match
- `min="value"` — `>= value`
- `max="value"` — `<= value`
- `min + max` — Range `[min, max]`

---

#### `<persistence>` (States/Trends)
```xml
<persistence good-after="24h" interpolate="true" max-skip="1"/>
```

**Parameters:**
- `good-after` — Maximum gap for interval merging
- `interpolate` — Allow skipping outliers (requires `max-skip > 0`)
- `max-skip` — Maximum consecutive outliers to skip

---

## Usage Examples

### Example 1: Raw Concept (Multi-Attribute)

**File:** `knowledge-base/raw-concepts/BASAL_BITZUA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<raw-concept name="BASAL_BITZUA" concept-type="raw">
    <categories>Medications</categories>
    <description>Basal insulin administration (dosage + route)</description>
    
    <attributes>
        <attribute name="BASAL_DOSAGE" type="numeric">
            <numeric-allowed-values>
                <allowed-value min="0" max="100"/>
            </numeric-allowed-values>
        </attribute>
        <attribute name="BASAL_ROUTE" type="nominal">
            <nominal-allowed-values>
                <allowed-value value="SubCutaneous"/>
                <allowed-value value="IntraVenous"/>
            </nominal-allowed-values>
        </attribute>
    </attributes>
    
    <tuple-order>
        <attribute name="BASAL_DOSAGE"/>
        <attribute name="BASAL_ROUTE"/>
    </tuple-order>
    
    <merge require-all="false"/>
</raw-concept>
```

**Output:** Tuples like `(15, "SubCutaneous")` or `(20, None)` if route missing.

---

### Example 2: Event (Multi-Source)

**File:** `knowledge-base/events/DISGLYCEMIA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<event name="DISGLYCEMIA_EVENT">
    <categories>Events</categories>
    <description>Dysglycemia event (hypo or hyper)</description>
    
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
        <attribute name="HYPERGLYCEMIA" tak="raw-concept" idx="0"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute name="GLUCOSE_MEASURE" idx="0">
                <allowed-value max="70"/>
            </attribute>
            <attribute name="HYPOGLYCEMIA" idx="0">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
        <rule value="Hyperglycemia" operator="or">
            <attribute name="GLUCOSE_MEASURE" idx="0">
                <allowed-value min="250"/>
            </attribute>
            <attribute name="HYPERGLYCEMIA" idx="0">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
```

**Logic:** If glucose <= 70 OR HYPOGLYCEMIA flag → emit "Hypoglycemia" event.

---

### Example 3: State (with Discretization)

**File:** `knowledge-base/states/GLUCOSE_MEASURE.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Glucose state abstraction</description>
    
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    
    <discretization-rules>
        <attribute idx="0">
            <rule value="Severe hypoglycemia" min="0" max="54"/>
            <rule value="Hypoglycemia" min="54" max="70"/>
            <rule value="Low glucose" min="70" max="140"/>
            <rule value="Normal glucose" min="140" max="180"/>
            <rule value="High glucose" min="180" max="250"/>
            <rule value="Hyperglycemia" min="250"/>
        </attribute>
    </discretization-rules>
    
    <!-- No abstraction rules → outputs discrete string directly -->
    <abstraction-rules/>
</state>
```

**Output:** Intervals like `[08:00 → 14:00, "Hypoglycemia"]`.

---

### Example 4: Trend

**File:** `knowledge-base/trends/GLUCOSE_MEASURE.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<trend name="GLUCOSE_MEASURE_TREND">
    <categories>Measurements</categories>
    <description>Glucose trend detection</description>
    
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" significant-variation="40"/>
    
    <time-steady value="12h"/>
    <persistence good-after="24h"/>
</trend>
```

**Logic:** If slope over 12h window exceeds ±40 → "Increasing"/"Decreasing", else "Steady".

---

### Example 5: Context (with Clippers)

**File:** `knowledge-base/contexts/BASAL_BITZUA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<context name="BASAL_BITZUA_CONTEXT">
    <categories>Medicine</categories>
    <description>Basal insulin influence context</description>
    
    <derived-from>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Low" operator="or">
            <attribute name="BASAL_BITZUA" idx="0">
                <allowed-value min="0" max="20"/>
            </attribute>
        </rule>
        <rule value="High" operator="or">
            <attribute name="BASAL_BITZUA" idx="0">
                <allowed-value min="40"/>
            </attribute>
        </rule>
    </abstraction-rules>
    
    <context-windows>
        <persistence value="Low" good-before="0h" good-after="12h"/>
        <persistence value="High" good-before="0h" good-after="24h"/>
    </context-windows>
    
    <clippers>
        <clipper name="DEATH" tak="raw-concept" clip-before="0s" clip-after="120y"/>
        <clipper name="RELEASE" tak="raw-concept" clip-before="0s" clip-after="120y"/>
    </clippers>
</context>
```

**Logic:** "Low" dosage extends 12h after administration; "High" extends 24h. DEATH/RELEASE trim intervals.

---

## Validation

### XSD Schema Validation
The Mediator automatically validates TAK XML files against `knowledge-base/tak_schema.xsd` during parsing.

**Validation happens for ALL TAK families:**
- ✅ Raw Concepts
- ✅ Events
- ✅ States
- ✅ Trends
- ✅ Contexts

**Installation:**
```bash
pip install -r requirements.txt  # Includes lxml
```

**Manual validation (command-line):**
```bash
# Validate all TAKs in knowledge-base
xmllint --schema knowledge-base/tak_schema.xsd knowledge-base/raw-concepts/*.xml
xmllint --schema knowledge-base/tak_schema.xsd knowledge-base/events/*.xml
xmllint --schema knowledge-base/tak_schema.xsd knowledge-base/states/*.xml
xmllint --schema knowledge-base/tak_schema.xsd knowledge-base/trends/*.xml
xmllint --schema knowledge-base/tak_schema.xsd knowledge-base/contexts/*.xml
```

**What gets validated:**
- ✅ XML syntax (well-formed)
- ✅ Required elements/attributes (e.g., `<categories>`, `<description>`, `name` attribute)
- ✅ Attribute types (e.g., `concept-type` must be `raw|raw-numeric|raw-nominal|raw-boolean`)
- ✅ Element order (e.g., `<derived-from>` must come before `<abstraction-rules>`)
- ✅ Duration format (e.g., `15m`, not `15 min`)
- ✅ Numeric types (e.g., `min`/`max` must be decimals)

**What gets validated by business logic (NOT in XSD):**
- ❌ Parent TAK existence (e.g., `derived-from="NONEXISTENT"` → runtime error)
- ❌ Attribute index bounds (e.g., `idx="5"` when tuple size is 3 → runtime error)
- ❌ Discretization coverage (e.g., gaps in numeric ranges → warning)
- ❌ Abstraction rule coverage (e.g., uncovered discrete values → warning)

---

## Tips

### 1. Naming Conventions
- **Raw Concepts:** Match InputPatientData `ConceptName` (e.g., `GLUCOSE_LAB_MEASURE`)
- **Abstractions:** Use `_EVENT`, `_STATE`, `_TREND`, `_CONTEXT` suffixes
- **File names:** Match TAK name (e.g., `BASAL_BITZUA.xml` for `<raw-concept name="BASAL_BITZUA">`)

### 2. Testing TAKs
```bash
# Process single patient with debug logs
python -m core.mediator --patients 1000 --log-level DEBUG

# Check output
sqlite3 backend/data/mediator.db "SELECT * FROM OutputPatientData WHERE PatientId=1000 AND ConceptName='GLUCOSE_MEASURE_STATE';"
```

### 3. Common Pitfalls
- ❌ **Tuple mismatch:** `<tuple-order>` must list ALL attributes (not subset)
- ❌ **Missing discretization:** Numeric attributes need rules before abstraction
- ❌ **Invalid durations:** Use `15m` not `15 min` (no spaces)
- ❌ **Wrong `idx`:** Tuple indices are 0-based (first attribute = idx 0)

---

## Support

**Questions?** Check:
1. [Main README](../../README.md) — System overview
2. [Unit Tests](../../unittests/) — Working examples
3. [Log File](../../backend/data/mediator_run.log) — Validation warnings
# TAK (Temporal Abstraction Knowledge) Documentation

## Table of Contents
1. [Overview](#overview)
2. [TAK Families](#tak-families)
3. [XML Schema Reference](#xml-schema-reference)
4. [Algorithms & Implementation](#algorithms--implementation)
5. [Validation Rules](#validation-rules)
6. [Usage Examples](#usage-examples)
7. [Common Issues](#common-issues)

---

## Overview

TAKs (Temporal Abstraction Knowledge) are XML-based definitions that transform raw clinical data into higher-level abstractions. The Mediator processes TAKs in dependency order:

```
Raw Concepts → Events → States → Trends → Contexts → Patterns
```

Each TAK family has specific:
- **XML schema requirements** (enforced by `tak_schema.xsd`)
- **Business logic validation** (parent dependencies, coverage checks)
- **Computational algorithms** (discretization, merging, slope calculation)

---

## TAK Families

### 1. Raw Concepts

**Purpose:** Bridge between `InputPatientData` and abstraction pipeline.

**Types:**
- `raw` — Multi-attribute concepts with tuple merging (e.g., medication dosage + route)
- `raw-numeric` — Single numeric attributes with range validation
- `raw-nominal` — Single nominal attributes with allowed values
- `raw-boolean` — Boolean flags (presence/absence)

**Key Parameters:**
- `concept-type` — One of: `raw`, `raw-numeric`, `raw-nominal`, `raw-boolean`
- `<tuple-order>` — (raw only) Defines attribute order in output tuples
- `<merge require-all>` — (raw only) Whether to require all attributes for tuple emission
- `<attributes>` — Define the string names of values in column `ConceptName` which will be converted to this raw concept

**Algorithm (raw concept-type):**
1. **Filter** input rows to only those matching declared attribute names
2. **Validate** values per-attribute type (numeric ranges, nominal allowed values)
3. **Group** by exact timestamp (nanosecond precision)
4. **Merge** attributes into tuples following `<tuple-order>`
5. **Filter** partial tuples if `require-all="true"`

**Output:** Tuples at exact timestamps (e.g., `(15, "SubCutaneous")`)

---

### 2. Events

**Purpose:** Point-in-time occurrences derived from one or more raw-concepts.

**Key Parameters:**
- `<derived-from>` — List of raw-concepts (can bridge multiple sources)
- `<abstraction-rules>` — Optional constraints: `equal`, `min`, `max`, `min+max` (range)
- `operator` — `or` (default) or `and` for multi-attribute rules

**Algorithm:**
1. **Filter** input to relevant raw-concepts
2. **Apply abstraction rules:**
   - For each rule, check constraints against input row
   - `operator="or"`: match if ANY attribute satisfies
   - `operator="and"`: match if ALL attributes satisfy
3. **Emit** point events (StartDateTime = EndDateTime)

**Constraint Types:**
```xml
<allowed-value equal="True"/>                <!-- exact match -->
<allowed-value min="70"/>                    <!-- >= 70 -->
<allowed-value max="180"/>                   <!-- <= 180 -->
<allowed-value min="70" max="180"/>          <!-- range [70, 180] -->
```

**Output:** Point events with symbolic labels (e.g., "Hypoglycemia")

---

### 3. States

**Purpose:** Symbolic intervals derived from numeric/nominal concepts via discretization.

**Key Parameters:**
- `<derived-from>` — Single raw-concept or event
- `<abstraction-rules>` — Combine attributes → final state labels, including numeric discretization.
- `<persistence>` — Interval merging: `good-after`, `interpolate`, `max-skip`

**Algorithm:**
1. **Discretize:** Map raw numeric values → discrete labels using range rules
   ```
   [0, 70) → "Hypoglycemia"
   [70, 180) → "Normal"
   [180, ∞) → "Hyperglycemia"
   ```

2. **Abstract:** Apply abstraction rules to discrete tuples/ Will return first matching rule

3. **Merge:** Concatenate adjacent identical states
   - **Same value + within `good_after` window** → merge
   - **Interpolation (`interpolate=true`):** Skip up to `max_skip` outliers if next point returns to original value
   - **Interval extension:** EndDateTime = last_merged_sample_time + good_after (or next sample's start, whichever is earlier)

**Output:** Symbolic intervals (e.g., `[08:00 → 14:00, "Hypoglycemia"]`)

---

### 4. Trends

**Purpose:** Compute local slopes (Increasing/Decreasing/Steady) over time windows.

**Key Parameters:**
- `<derived-from>` — Single raw-concept (numeric attribute)
- `significant-variation` — Threshold for trend detection
- `<time-steady>` — Lookback window for slope calculation (all points in this window are collected)
- `<persistence good-after>` — Maximum gap for interval stretching

**Algorithm:**
1. **Slope Calculation (per point `i`):**
   - Collect all points in window `[t_i - time_steady, t_i]`
   - Compute OLS (Ordinary Least Squares) linear regression slope
   - **Optimizations:**
     - Binary search for window boundaries (O(log n))
     - Pre-convert timestamps to float seconds (stable numerics)
     - Vectorized numpy operations (no loops)
   - **Total variation:** `slope * time_steady.total_seconds()`

2. **Classification:**
   ```
   if total_var >= significant_variation → "Increasing"
   elif total_var <= -significant_variation → "Decreasing"
   else → "Steady"
   ```

3. **Interval Building (anchor-based):**
   - **Anchor rule:** First point sets anchor (no interval yet)
   - **Next point:** Stretch back to current anchor if `(t_i - anchor) <= good_after`
   - **Gap exceeds window:** Emit hole interval `[anchor, t_i]` with `Value=None`, reset anchor
   - **Consecutive identical labels:** Merge by extending EndDateTime

**Note:** Generally, `time_steady << good_after` (e.g., 12h lookback, 24h merge window)

**Output:** Trend intervals (e.g., `[08:00 → 14:00, "Increasing"]`)

---

### 5. Contexts

**Purpose:** Background facts with interval windowing and clipping. Very similar to Events otherwise.

**Key Parameters:**
- `<derived-from>` — List of raw-concepts
- `<context-windows>` — Per-value windows: `good-before`, `good-after`
- `<clippers>` — External events that trim intervals (e.g., ADMISSION, DEATH)

**Algorithm:**
1. **Apply abstraction rules** (same as Events)

2. **Context Windowing:**
   - Original interval: `[t_i, t_i]` (point-in-time from raw-concept)
   - Windowed interval: `[t_i - good_before, t_i + good_after]`
   - **Value-specific windows:** Each abstraction value can have different windows
   - **Default window:** Applies to values without specific windows

3. **Clipping:**
   - For each clipper event (e.g., DEATH at `t_death`):
     - **clip-before:** If context starts before clipper, trim start to `t_clipper + clip_before`
     - **clip-after:** If context overlaps clipper, delay start to `t_clipper_end + clip_after`
   - **Invalid intervals** (start >= end) are removed

**Output:** Windowed intervals (e.g., `[08:00 → 20:00, "Low Basal"]`)

---

## XML Schema Reference

### Duration Strings

Format: `<number><unit>` (e.g., `15m`, `24h`, `3d`)

**Units:**
```
s → seconds
m → minutes
h → hours
d → days
w → weeks
M → months (30 days)
y → years (365 days)
```

---

### Critical Element Order Rules

**The XSD schema enforces STRICT element ordering. Wrong order = validation failure.**

#### Raw Concepts
```xml
<raw-concept name="..." concept-type="raw">
    <categories>...</categories>
    <description>...</description>
    <attributes>...</attributes>
    <tuple-order>...</tuple-order>  <!-- BEFORE merge -->
    <merge require-all="..."/>
</raw-concept>
```

#### Events/Contexts
```xml
<event name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from>...</derived-from>
    <abstraction-rules>...</abstraction-rules>  <!-- BEFORE context-windows (contexts only) -->
    <context-windows>...</context-windows>      <!-- contexts only -->
    <clippers>...</clippers>                    <!-- contexts only, always last -->
</event>
```

#### States
```xml
<state name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from name="..." tak="..."/>
    <persistence good-after="..." interpolate="..." max-skip="..."/>
    <discretization-rules>...</discretization-rules>  <!-- BEFORE abstraction-rules -->
    <abstraction-rules order="...">...</abstraction-rules>
</state>
```

---

## Algorithms & Implementation

### State Merging Algorithm

**Input:** Sorted list of (StartDateTime, Value) tuples  
**Output:** List of merged intervals with extended EndDateTime

```python
# Pseudocode
merged_intervals = []
anchor = first_sample_time
current_value = first_sample_value
merged_samples = [anchor]
skip_count = 0

for next_sample in samples[1:]:
    time_gap = next_sample.time - anchor
    same_value = (next_sample.value == current_value)
    
    if same_value and time_gap <= good_after:
        # Merge into current interval
        merged_samples.append(next_sample.time)
        skip_count = 0
    elif not same_value and interpolate and skip_count < max_skip:
        # Check if next-next sample returns to original value
        if peek_next_sample.value == current_value and peek_gap <= good_after:
            # Skip this outlier
            skip_count += 1
            continue
        else:
            # Break interval
            emit_interval(anchor, last_merged_time + good_after, current_value)
            anchor = next_sample.time
            current_value = next_sample.value
            merged_samples = [anchor]
            skip_count = 0
    else:
        # Break interval
        emit_interval(anchor, last_merged_time + good_after, current_value)
        anchor = next_sample.time
        current_value = next_sample.value
        merged_samples = [anchor]
        skip_count = 0

# Emit last interval
emit_interval(anchor, last_merged_time + good_after, current_value)
```

**Key Properties:**
- **Interval end capped at next sample's start** (avoids overlaps)
- **Interpolation requires 3 points** (before, outlier, after)
- **Same value + within window → always merge** (even if gap > 0)

---

### Trend Slope Calculation

**OLS (Ordinary Least Squares) Linear Regression:**

```python
# Collect points in window [t_i - time_steady, t_i]
window_times = [t_j for t_j in times if t_i - time_steady <= t_j <= t_i]
window_values = [v_j for corresponding times]

# Convert to seconds (stable numerics)
times_sec = [(t - window_times[0]).total_seconds() for t in window_times]

# OLS regression
x_mean = mean(times_sec)
y_mean = mean(window_values)
numerator = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(times_sec, window_values))
denominator = sum((x_i - x_mean)^2 for x_i in times_sec)
slope = numerator / denominator  # units: value per second

# Total variation over window
total_var = slope * time_steady.total_seconds()

# Classify
if total_var >= significant_variation:
    label = "Increasing"
elif total_var <= -significant_variation:
    label = "Decreasing"
else:
    label = "Steady"
```

**Optimizations:**
- Binary search for window boundaries (O(log n) vs O(n))
- Pre-convert timestamps to float seconds (avoid repeated conversions)
- Vectorized numpy operations (dot products instead of loops)

---

## Validation Rules

### XSD Schema Validation (Structural)

**Enforced by `tak_schema.xsd`:**
- ✅ XML well-formed
- ✅ Required elements present (`<categories>`, `<description>`, ...)
- ✅ Element order correct (see Critical Element Order Rules)
- ✅ Attribute types valid (`concept-type`, `tak`, `operator`, ...)
- ✅ Duration format valid (`15m`, not `15 min`)
- ✅ Constraint types valid (`equal` XOR `min`/`max`)

---

### Business Logic Validation (Semantic)

**Enforced by TAK.validate():**

#### All TAKs
- ✅ Parent TAK exists in repository
- ✅ Parent TAK has correct type (e.g., States can only derive from RawConcept or Event)

#### Raw Concepts
- ✅ No duplicate attribute names
- ✅ Nominal attributes have non-empty allowed values
- ✅ Numeric attributes have valid ranges (min < max)
- ✅ `tuple-order` lists exactly all declared attributes (for `concept-type="raw"`)

#### Events/Contexts
- ✅ Derived-from TAKs exist and are RawConcepts
- ✅ `operator="and"` requires all attributes from same source
- ✅ Constraint values match parent attribute's allowed values (for nominal)

#### States
- ✅ Derived-from is RawConcept or Event
- ✅ Discretization rule indices within tuple bounds
- ✅ Numeric attributes have discretization before abstraction
- ⚠️ **Warn:** Discretization coverage gaps (uncovered ranges)
- ⚠️ **Warn:** Abstraction coverage gaps (uncovered discrete values)

#### Trends
- ✅ Derived-from is RawConcept with numeric attribute
- ✅ Attribute index within tuple bounds
- ✅ `good_after > 0`

#### Contexts
- ✅ Clippers exist and are RawConcepts
- ✅ Context windows match abstraction rule values (bidirectional check)
- ✅ Default window exists OR value-specific windows cover all rules

---

### Context Window Validation (Bidirectional)

**Rule:** Every abstraction rule value must have a window definition (value-specific OR default).

```xml
<!-- ✅ Valid: value-specific windows -->
<abstraction-rules>
    <rule value="Low">...</rule>
    <rule value="High">...</rule>
</abstraction-rules>
<context-windows>
    <persistence value="Low" good-before="0h" good-after="12h"/>
    <persistence value="High" good-before="0h" good-after="24h"/>
</context-windows>

<!-- ✅ Valid: default window covers all -->
<abstraction-rules>
    <rule value="Low">...</rule>
    <rule value="High">...</rule>
</abstraction-rules>
<context-windows>
    <persistence good-before="0h" good-after="12h"/>  <!-- no value attr -->
</context-windows>

<!-- ❌ Invalid: typo in value -->
<abstraction-rules>
    <rule value="Low">...</rule>
</abstraction-rules>
<context-windows>
    <persistence value="LOW" good-before="0h" good-after="12h"/>  <!-- case mismatch! -->
</context-windows>
```

---

## Usage Examples

### Example 1: Raw Concept (Multi-Attribute Tuple)

**File:** `raw-concepts/BASAL_BITZUA.xml`

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
    
    <merge require-all="false"/>  <!-- allow partial tuples -->
</raw-concept>
```

**InputPatientData:**
```
PatientId | ConceptName   | StartDateTime       | Value
1000      | BASAL_DOSAGE  | 2024-01-01 21:00:00 | 15
1000      | BASAL_ROUTE   | 2024-01-01 21:00:00 | SubCutaneous
```

**OutputPatientData:**
```
PatientId | ConceptName  | StartDateTime       | Value                | AbstractionType
1000      | BASAL_BITZUA | 2024-01-01 21:00:00 | (15, SubCutaneous)   | raw-concept
```

---

### Example 2: Event (Multi-Source with OR logic)

**File:** `events/DISGLYCEMIA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<event name="DISGLYCEMIA_EVENT">
    <categories>Events</categories>
    <description>Dysglycemia event (hypo or hyper)</description>
    
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute name="GLUCOSE_MEASURE" idx="0">
                <allowed-value max="70"/>  <!-- <= 70 mg/dL -->
            </attribute>
            <attribute name="HYPOGLYCEMIA" idx="0">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
```

**Logic:** Emit "Hypoglycemia" if **either** glucose <= 70 **OR** HYPOGLYCEMIA flag is True.

---

### Example 3: State (Discretization + Merging)

**File:** `states/GLUCOSE_MEASURE.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Glucose state abstraction</description>
    
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    
    <discretization-rules>
        <attribute idx="0">
            <rule value="Hypoglycemia" min="0" max="70"/>
            <rule value="Normal" min="70" max="180"/>
            <rule value="Hyperglycemia" min="180"/>
        </attribute>
    </discretization-rules>
</state>
```

**Input (GLUCOSE_MEASURE raw-concept):**
```
Time  | Value
08:00 | 60   → Hypoglycemia
12:00 | 65   → Hypoglycemia
14:00 | 200  → Hyperglycemia (outlier)
18:00 | 62   → Hypoglycemia
```

**Output (GLUCOSE_MEASURE_STATE):**
```
StartDateTime | EndDateTime | Value
08:00         | 18:00+24h   | Hypoglycemia  (merged with interpolation, skipped outlier at 14:00)
```

---

### Example 4: Trend (Slope-Based)

**File:** `trends/GLUCOSE_MEASURE.xml`

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

**Logic:**
- For each point, compute slope over **12h lookback window**
- If `slope * 12h >= 40` → "Increasing"
- If `slope * 12h <= -40` → "Decreasing"
- Otherwise → "Steady"

**Input:**
```
Time  | Value
00:00 | 100
02:00 | 150  (slope from [00:00-02:00] ≈ +25/h → total_var = 25*12 = 300 > 40 → Increasing)
04:00 | 200
06:00 | 210
```

**Output:**
```
StartDateTime | EndDateTime | Value
00:00         | 06:00       | Increasing  (merged 4 consecutive "Increasing" classifications)
```

---

### Example 5: Context (Windowing + Clipping)

**File:** `contexts/BASAL_BITZUA.xml`

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
    </abstraction-rules>
    
    <context-windows>
        <persistence value="Low" good-before="0h" good-after="12h"/>
    </context-windows>
    
    <clippers>
        <clipper name="DEATH" tak="raw-concept" clip-before="0s" clip-after="120y"/>
    </clippers>
</context>
```

**Input (BASAL_BITZUA):**
```
Time  | Dosage | Route         → Abstraction
21:00 | 15     | SubCutaneous  → "Low"
```

**Windowed:**
```
StartDateTime | EndDateTime
21:00         | 21:00+12h = 09:00 (next day)
```

**Clipping (if DEATH at 05:00):**
```
StartDateTime | EndDateTime
21:00         | 05:00  (clipped by DEATH event)
```

---

## Common Issues

### Issue 1: Element Order Violation

**Symptom:**
```
ValueError: XML validation failed:
Element 'tuple-order': This element is not expected. Expected is ( merge ).
```

**Cause:** Wrong element order in raw-concept XML.

**Fix:** Reorder to match schema:
```xml
<attributes>...</attributes>
<tuple-order>...</tuple-order>  <!-- BEFORE merge -->
<merge require-all="..."/>
```

---

### Issue 2: Context Window Mismatch

**Symptom:**
```
ValueError: context-window for value='LOW' does not match any abstraction rule value (possible typo)
```

**Cause:** Case mismatch between rule value and window value.

**Fix:**
```xml
<!-- ❌ Wrong -->
<rule value="Low">...</rule>
<persistence value="LOW" .../>

<!-- ✅ Correct -->
<rule value="Low">...</rule>
<persistence value="Low" .../>
```
# TAK (Temporal Abstraction Knowledge) Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Assumptions & Process](#data-assumptions--process)
3. [TAK Families](#tak-families)
   - [Raw Concepts](#1-raw-concepts)
   - [Parameterized Raw Concepts](#2-parameterized-raw-concepts)
   - [Events](#3-events)
   - [States](#4-states)
   - [Trends](#5-trends)
   - [Contexts](#6-contexts)
   - [Patterns (Local)](#7-patterns-local)
4. [XML Schema Reference](#xml-schema-reference)
5. [Algorithms & Implementation](#algorithms--implementation)
6. [Validation Rules](#validation-rules)
7. [External Functions](#external-functions)
8. [Usage Examples](#usage-examples)
9. [Pattern Design Best Practices](#pattern-design-best-practices)

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

## Data Assumptions & Process

### Input Data Structure

**The Mediator only accepts temporal data in a standardized format:**

```sql
CREATE TABLE IF NOT EXISTS InputPatientData (
    RowId INTEGER PRIMARY KEY AUTOINCREMENT, -- automated, not in input.
    PatientId INTEGER NOT NULL,
    ConceptName TEXT NOT NULL,
    StartDateTime TEXT NOT NULL,
    EndDateTime TEXT NOT NULL,
    Value TEXT NOT NULL,
    Unit TEXT,
    UNIQUE (PatientId, ConceptName, StartDateTime)
);
```

**Key Requirements:**
- **All data must be temporal:** Every record must have `StartDateTime` and `EndDateTime` (even for point-in-time measurements)
- **ConceptName standardization:** All relevant IDs/codes from the original coding system (e.g., ICD, LOINC) must be mapped to a single `ConceptName` column
- **Value normalization:** Units and dosages must be pre-normalized to a single scale before input
- **Multi-attribute concepts:** Handled via RawConcept type `raw` (e.g., medication dosage + route are separate input rows, merged by Mediator)

---

### Research Context: Diabetes in Hospitalization

This TAK knowledge base was developed for analyzing diabetes management during hospital admissions. Key assumptions:

#### Patient ID = Admission ID
- Each `PatientId` in the dataset represents a **unique hospital admission** (not a unique person)
- A single patient can have multiple admissions (multiple `PatientId` values)
- This design allows per-admission analysis and simplifies temporal reasoning

#### Admission/Release/Death Events
- **All admissions have:**
  - **`ADMISSION` event** at admission time (marks episode start)
  - **`RELEASE` or `DEATH` event** at episode end
- **Death handling:**
  - If death occurs ≤ 30 days after `RELEASE`, death event **replaces** release
  - Rationale: Late deaths are considered admission-related outcomes

#### State Threshold Discovery
**Discretization thresholds were data-driven, not arbitrary:**
- **Method:** K-means clustering + Kernel Density Estimation (KDE)
- **Process:** Search for significant peaks/valleys in concept value distributions
- **Outcome:** Empirically grounded state boundaries (e.g., glucose ranges)

**Examples:**
- **Simple concepts (e.g., `GLUCOSE_MEASURE`):** Single-attribute thresholds directly applied
- **Complex concepts (e.g., `STEROIDS`):** Pre-discretized at medication level before Mediator input
  - Rationale: Too many distinct values (drug-dose combinations) for in-Mediator discretization
  - Solution: External pre-processing reduces to categorical labels (Low/Medium/High)

#### Meal Event Assumptions
- **MEAL events are NOT in raw data**
- **Assumption:** Meals occur at predefined standard hospital times:
  - Breakfast: 08:00
  - Lunch: 12:00
  - Dinner: 18:00
  - Snack: 21:00 (if applicable)
- **Implementation:** Synthetic MEAL_EVENT records injected during data preprocessing

#### Prior Hospitaslization Events Assumptions
- **AdmissionID as PatientID restricts information availability from prior admissions**
- **Solution:** Relevant measurements (specific concepts and specific time ranges) can be synthetically added to the input data with StartDateTime == Admission time.

#### Medication Orders vs. Actual Administration
- **Available:** Actual medication administration records only
- **NOT available:** Prescription orders or instructions
- **Implication:** Cannot analyze prescription-administration gaps

---

### Design Patterns: When to Use Each TAK Family

#### Events & Contexts: Semantic Unification
**Use Case:** Multiple concepts represent the same clinical meaning

**Events:** Point-in-time unification
```xml
<!-- DISGLYCEMIA can come from glucose lab value OR hypoglycemia flag -->
<event name="DISGLYCEMIA_EVENT">
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0" ref="A2"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute ref="A1"><allowed-value max="70"/></attribute>
            <attribute ref="A2"><allowed-value equal="True"/></attribute>
        </rule>
    </abstraction-rules>
</event>
```

**Contexts:** Interval-based unification with temporal windows
```xml
<!-- BASAL influence can come from SubCutaneous OR IntraVenous routes -->
<context name="BASAL_BITZUA_CONTEXT">
    <derived-from>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>
    <abstraction-rules>
        <rule value="Low" operator="or">
            <attribute ref="A1"><allowed-value min="0" max="20"/></attribute>
        </rule>
    </abstraction-rules>
    <context-windows>
        <persistence value="Low" good-before="0h" good-after="12h"/>
    </context-windows>
</context>
```

**Key Difference from Patterns:**
- Events/Contexts: Unify attributes at **exact same time** (no temporal gap) using `operator="and"` (if attributes belong to the same RawConcept) or simply pick one of K concepts that occured using `operator="or"` (not limited by number of concepts).
- Patterns: Relate concepts with **temporal separation** (before/overlap relationships)

#### Patterns: Multi-Concept Temporal Relationships
**Use Case:** Detect relationships between 2+ concepts across time

**Only Patterns can:**
- Define temporal relationships (before/overlap with max-distance)
- Combine concepts from different time points
- Score compliance with clinical guidelines using fuzzy time/value windows

```xml
<!-- Pattern: Glucose measured AFTER admission, within reasonable time -->
<pattern name="GLUCOSE_MEASURE_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <derived-from>
        <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
    <abstraction-rules>
        <rule>
            <temporal-relation how='before' max-distance='12h'>
                <anchor><attribute ref="A1">...</attribute></anchor>
                <event select='first'><attribute ref="E1">...</attribute></event>
            </temporal-relation>
        </rule>
    </abstraction-rules>
</pattern>
```

**Design Principle:**
- **Events/Contexts:** "What happened?" (semantic abstraction)
- **Patterns:** "What happened **after** what?" (temporal reasoning)

---

### Data Preprocessing Pipeline

**Typical workflow for this research:**

1. **Extract raw EHR data** → ICD codes, lab values, medication records
2. **Normalize concepts:**
   - Map all glucose-related LOINC codes → `GLUCOSE_MEASURE`
   - Convert all insulin dosages to standard units (e.g., IU)
3. **Discretize complex concepts** (e.g., steroids) → categorical labels
4. **Inject synthetic events** (e.g., MEAL times)
5. **Handle Death events** (replace RELEASE if death ≤ 30 days)
6. **Load to InputPatientData table** (temporal format)
7. **Run Mediator** → apply TAKs in dependency order
8. **Output to OutputPatientData** → abstracted intervals + QA scores
9. **Analize & Predict** → Analyze abstracted intervals (e.g. TIRP discovery) + QA scores per Pattern / group of Patterns, and predict using the abstracted data and/or discovered patterns.

---

## TAK Families

### 1. Raw Concepts

**Purpose:** Bridge between `InputPatientData` and abstraction pipeline.

**Types:**
- `raw` — Multi-attribute concepts with tuple merging (e.g., medication dosage + route)
- `raw-numeric` — Single numeric attributes with range validation
- `raw-nominal` — Single nominal attributes with allowed values
- `raw-boolean` — Boolean flags (presence only), will automatically assign str "True" as one nominal value.

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

### 2. Parameterized Raw Concepts

**Purpose:**  
Parameterized Raw Concepts allow you to define a new raw concept as a function of an existing raw concept and one or more parameters (such as the first value of another measurement, or a patient-specific attribute). This enables dynamic calculation of derived values at the raw abstraction level, before higher-level abstractions like events, states, or patterns.

**Key Features:**
- **Derived-from:** References a parent raw concept (e.g., GLUCOSE_MEASURE).
- **Parameters:** References additional raw concepts or constants, resolved per patient and per row (e.g., FIRST_GLUCOSE_MEASURE, WEIGHT_MEASURE).
- **Functions:** Specifies how to combine the parent value and parameters using a named function (e.g., division, multiplication).
- **Default values:** Each parameter must have a default value, used if no matching row is found for the patient.

**Algorithm:**
1. For each row of the parent raw concept, resolve parameter values:
    - If a matching parameter row exists (by name and closest in time), use its value.
    - Otherwise, use the parameter's default value.
2. Apply the specified function (e.g., `div`) to the parent value and parameter(s).
3. Emit a new row with the result as the value, and the same temporal columns as the parent.

**Output:**  
A DataFrame with the same shape and columns as the parent raw concept, but with the modified value in-place.

**Use Cases:**
- Calculating ratios (e.g., glucose divided by first glucose measurement)
- Normalizing measurements by patient-specific attributes (e.g., dosage per kg)
- Any derived raw value that can be expressed as a function of other raw values and parameters

**Notes:**
- Parameterized raw concepts are resolved and emitted before events, states, trends, contexts, and patterns.
- All parameters must have a default value to ensure robust calculation.
- Functions are extensible and can be registered in the external functions module.
---

### 3. Events

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

### 4. States

**Purpose:** Symbolic intervals derived from numeric/nominal concepts via discretization.

**Key Parameters:**
- `<derived-from>` — Single raw-concept or event
- `<abstraction-rules>` — Combine attributes → final state labels, including numeric discretization
- `<persistence>` — Interval merging: `good-after`, `interpolate`, `max-skip`

**Algorithm:**
1. **Discretize:** Map raw numeric values → discrete labels using range rules
   ```
   [0, 70) → "Hypoglycemia"
   [70, 180) → "Normal"
   [180, ∞) → "Hyperglycemia"
   ```

2. **Abstract:** Apply abstraction rules to discrete tuples (returns first matching rule)

3. **Merge:** Concatenate adjacent identical states
   - **Same value + within `good_after` window** → merge
   - **Interpolation (`interpolate=true`):** Skip up to `max_skip` outliers if next point returns to original value
   - **Interval extension:** EndDateTime = last_merged_sample_time + good_after (or next sample's start, whichever is earlier)

**Output:** Symbolic intervals (e.g., `[08:00 → 14:00, "Hypoglycemia"]`)

---

### 5. Trends

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

### 6. Contexts

**Purpose:** Background facts with interval windowing and clipping (similar to Events with temporal extension).

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
   - Clippers can be all TAK objects.
   - For each clipper event (e.g., DEATH at `t_death`):
     - **clip-before:** If context starts before clipper, trim start to `t_clipper + clip_before`
     - **clip-after:** If context overlaps clipper, delay start to `t_clipper_end + clip_after`
   - **Invalid intervals** (start >= end) are removed

**Output:** Windowed intervals (e.g., `[08:00 → 20:00, "Low Basal"]`)

NOTE: If 2 contexts's windows overlap, the later one automatically clips the earlier one's window.

---

### 7. Patterns (Local)

**Purpose:** Detect complex temporal relationships between multiple TAKs with optional fuzzy compliance scoring.

**Key Parameters:**
- `<derived-from>` — List of TAKs (raw-concepts, events, states, contexts) with **ref** identifiers
- `<parameters>` — Optional external values (e.g., patient weight) for compliance functions
- `<abstraction-rules>` — One or more pattern detection rules (each rule represents an independent pattern instance)
- `<compliance-function>` — Optional fuzzy scoring using trapezoidal membership functions

#### Ref Mechanism

Patterns use **ref-based indexing** to reference attributes/parameters (similar to Events/Contexts):

```xml
<derived-from>
    <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
    <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
</derived-from>

<!-- Later reference by ref in rules: -->
<anchor>
    <attribute ref="A1">...</attribute>
</anchor>
<event>
    <attribute ref="E1">...</attribute>
</event>
```

#### Algorithm

1. **Candidate Extraction:**
   - For each rule, extract **anchor** and **event** candidates from input data
   - Filter candidates by attribute constraints (equal/min/max)
   - Sort by `select` preference (first/last based on StartDateTime)

2. **Temporal Matching (Vectorized Pre-filtering):**
   - **before:** `anchor.end < event.start` AND `(event.start - anchor.end) <= max-distance`
   - **overlap:** Intervals overlap (not disjoint)
   - **Optimization:** Vectorized pandas masks reduce O(N²) nested loop to O(N×M) (where M is # of valid pairs)

3. **Context Checking (Optional):**
   - Check if context interval overlaps `[min(anchor.start, event.start), max(anchor.end, event.end)]`
   - Context must exist and overlap the pattern timeframe

4. **One-to-One Pairing:**
   - Track used anchor/event indices (no reuse across rules)
   - For each anchor, find first matching event (by sort order)
   - Break after first match (greedy pairing)

5. **Compliance Scoring (Optional):**
   - **Parameters resolved ONCE per patient** (closest to pattern start time, or default)
   - **Time-constraint:** Score actual time gap using trapezoidal function
   - **Value-constraint:** Score target attribute values (from anchor/event)
   - **Combined score:** Average of time + value scores (allows partial compliance on one dimension)
   - **Classification:**
     ```
     score == 1.0 → "True"
     score > 0.0  → "Partial"
     score == 0.0 → "False"
     ```

6. **Output Generation:**
    - **Input NOT found:** Output for the pattern is only calculated if the patient has any records related to that pattern (anchor / event / parameters / context) otherwise - an empty df is returned.
   - **Pattern found:** One or more intervals with `Value="True"/"Partial"/"False"`, compliance scores in separate columns
   - **Pattern NOT found:** Single row with `Value="False"`, `StartDateTime/EndDateTime=NaT`,
   compliance scores in seperate columns will be 0, if compliance function is defined for the Pattern.

#### Compliance Functions

Patterns support **trapezoidal membership functions** for fuzzy temporal/value constraints:

```
Score
1.0 |        ____________________
    |       /                    \
0.5 |      /                      \
    |     /                        \
0.0 |____/                          \_________
        A    B                  C    D  (time or value)
```
- **[A, B]:** Score linearly increases 0 → 1
- **[B, C]:** Score = 1 (full compliance)
- **[C, D]:** Score linearly decreases 1 → 0
- **Outside [A, D]:** Score = 0

**Types:**
- **Time-constraint:** Trapez values are duration strings (e.g., `"0h"`, `"8h"`)
  - Scores actual time gap: `event.start - anchor.end`
- **Value-constraint:** Trapez values are numeric (floats)
  - Scores target attribute values (from anchor/event rows)

#### External Functions for Compliance

Compliance functions can transform trapez values dynamically using external functions:

- **`id`** — Identity (trapez values used as-is)
  ```xml
  <function name="id">
      <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
  </function>
  ```

- **`mul`** — Multiply trapez by parameter (e.g., dosage per kg body weight)
  ```xml
  <function name="mul">
      <parameter ref="P1"/>  <!-- e.g., weight -->
      <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
      <!-- Result: [0, 0.2*weight, 0.6*weight, 1*weight] -->
  </function>
  ```

- **Custom functions:** Register in `external_functions.py` (see [External Functions](#external-functions) section)

#### Parameter Types

Parameters can be numeric, time-duration strings, or arbitrary strings (from all TAK types, points to "Value" column):

- **Numeric:** `default="72"` → 72.0
- **Time-duration:** `default="1h"` → 3600.0 seconds (converted by `parse_duration()`)
- **String:** Passed as-is to external function (function validates)

**Resolution Strategy:**
- Use **closest record to pattern start time** (minimize time distance)
- Fallback to `default` value if no data found for patient

---

## XML Schema Reference

### Duration Strings

Format: `<number><unit>` (e.g., `15m`, `24h`, `3d`)

**Supported Units:**
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

#### Parameterized Raw Concepts
```xml
<raw-concept name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from name="..." tak="..."/>         <!-- Single TAK reference, no idx -->
    <parameters>...</parameters>
    <functions>...</functions>
</raw-concept>
```

#### Events
```xml
<event name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from>...</derived-from>            <!-- Multiple attributes possible -->
    <abstraction-rules>...</abstraction-rules>  <!-- Optional -->
</event>
```

#### States
```xml
<state name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from name="..." tak="..."/>         <!-- Single TAK reference, no idx -->
    <persistence good-after="..." interpolate="..." max-skip="..."/>
    <abstraction-rules>...</abstraction-rules>   <!-- Optional -->
</state>
```

#### Contexts
```xml
<context name="...">
    <categories>...</categories>
    <description>...</description>
    <derived-from>...</derived-from>            <!-- Multiple attributes possible -->
    <clippers>...</clippers>                    <!-- Optional, BEFORE abstraction-rules -->
    <abstraction-rules>...</abstraction-rules>  <!-- Optional, BEFORE context-windows -->
    <context-windows>...</context-windows>      <!-- Required -->
</context>
```

#### Patterns
```xml
<pattern name="..." concept-type="local-pattern">
    <categories>...</categories>
    <description>...</description>
    <derived-from>...</derived-from>            <!-- Multiple attributes with refs -->
    <parameters>...</parameters>                <!-- Optional, BEFORE abstraction-rules -->
    <abstraction-rules>...</abstraction-rules>  <!-- One or more rules -->
</pattern>
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

### Pattern Performance Optimizations

**Implemented Optimizations:**
- **Vectorized temporal filtering:** Pre-filter anchor-event pairs using pandas masks (reduces O(N²) to O(N×M))
- **One-time parameter resolution:** Resolve all parameters once per patient (not per-rule or per-instance)
- **Early exits:** Skip rules with no anchor/event candidates
- **Binary search for trapez lookup:** O(log n) compliance score computation

**Typical Performance:**
- **Per-patient (100 records, 3 rules, 2 parameters):** ~10-20ms
- **Throughput:** ~50-100 patients/second (single-threaded)

---

## Validation Rules

### XSD Schema Validation (Structural)

**Enforced by `tak_schema.xsd`:**
- ✅ XML well-formed
- ✅ Required elements present (`<categories>`, `<description>`, ...)
- ✅ Element order correct (see [Critical Element Order Rules](#critical-element-order-rules))
- ✅ Attribute types valid (`concept-type`, `tak`, `operator`, ...)
- ✅ Duration format valid (`15m`, not `15 min`)
- ✅ Constraint types valid (`equal` XOR `min`/`max`)

---

### Business Logic Validation (Semantic)

**Enforced by TAK.parse() and TAK.validate():**

#### All TAKs
- ✅ Parent TAK exists in repository
- ✅ Parent TAK has correct type (e.g., States can only derive from RawConcept or Event)

#### Raw Concepts
- ✅ No duplicate attribute names
- ✅ Nominal attributes have non-empty allowed values
- ✅ Numeric attributes have valid ranges (min < max)
- ✅ `tuple-order` lists exactly all declared attributes (for `concept-type="raw"`)

#### Parameterized Raw Concepts
- ✅ The parent TAK referenced in `<derived-from>` must exist and be a raw concept.
- ✅ All parameters in `<parameters>` must reference valid TAKs or constants and have a `default` value.
- ✅ Each `<function>` must reference valid indices and parameter refs.
- ✅ The function name in `<function name="...">` must be registered and available.
- ✅ No duplicate parameter refs or names.
- ✅ Output shape and columns must match the parent raw concept (except for ConceptName/Value).
- ✅ All parameterized raw concepts must be registered before any TAKs that depend on them (e.g., states, events).

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

#### Patterns
- ✅ All derived-from TAKs exist in repository
- ✅ `idx` values within bounds for raw-concept tuples
- ✅ `time-constraint-compliance` only valid for `how='before'`
- ✅ `max-distance >= trapezeD` (pattern captures all valid instances)
- ✅ Value-constraint targets must reference **anchor or event** (not context/parameter)
- ✅ Compliance function names exist in `external_functions.REPO`
- ✅ Parameter refs declared in `<parameters>` block
- ✅ Numeric attributes use `min`/`max` constraints (not `equal`)
- ✅ Nominal/boolean attributes use `equal` constraints (not `min`/`max`)

---

## External Functions

**Purpose:** Transform compliance function trapezoid values dynamically using patient-specific parameters.

**Registration:** All external functions are registered in `core/tak/external_functions.py` using the `@register()` decorator:

```python
# filepath: core/tak/external_functions.py
from typing import Dict, Callable

# Repository of external functions
REPO: Dict[str, Callable] = {}

def register(name: str):
    """Decorator to register external function."""
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
    """Multiply x by all parameters."""
    result = x
    for arg in args:
        result *= arg
    return result

@register("add")
def add(x, *args):
    """Add all parameters to x."""
    result = x
    for arg in args:
        result += arg
    return result
```

### Creating Custom External Functions

**Step 1: Define and register the function**

```python
# filepath: core/tak/external_functions.py

@register("custom_dosage")
def custom_dosage_adjustment(x, *args):
    """
    Custom dosage adjustment based on weight and age.
    
    The function is called ONCE per trapez value (A, B, C, or D).
    Parameters are passed as *args in the order declared in XML.
    
    Args:
        x: Single trapez value (e.g., trapezeA=0 or trapezeB=0.2)
        *args: Variable parameters in declaration order
               args[0] = weight (from P1)
               args[1] = age (from P2)
    
    Returns:
        Transformed trapez value (must preserve ordering: A <= B <= C <= D)
    
    Example:
        If trapez=(0, 0.2, 0.6, 1) and weight=72, age=70:
        - custom_dosage(0, 72, 70) → 0 * 72 * 0.8 = 0
        - custom_dosage(0.2, 72, 70) → 0.2 * 72 * 0.8 = 11.52
        - custom_dosage(0.6, 72, 70) → 0.6 * 72 * 0.8 = 34.56
        - custom_dosage(1, 72, 70) → 1 * 72 * 0.8 = 57.6
        Result trapez: (0, 11.52, 34.56, 57.6)
    """
    # Extract parameters (function trusts caller to provide correct order/count)
    if len(args) < 2:
        raise ValueError("custom_dosage requires 2 parameters: weight, age")
    
    weight = args[0]
    age = args[1]
    
    # Apply transformation
    adjusted = x * weight
    if age > 65:
        adjusted *= 0.8  # 20% reduction for elderly patients
    
    return adjusted
```

**That's it!** The `@register()` decorator automatically adds the function to `REPO`. No manual registration needed.

**Step 2: Use in pattern XML**

```xml
<value-constraint-compliance>
    <target>
        <attribute ref="E1"/>  <!-- BASAL dosage -->
    </target>
    <function name="custom_dosage">
        <parameter ref="P1"/>  <!-- Weight (args[0]) -->
        <parameter ref="P2"/>  <!-- Age (args[1]) -->
        <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
    </function>
</value-constraint-compliance>
```

**Step 3: Declare parameters**

```xml
<parameters>
    <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
    <parameter name="AGE" tak="raw-concept" idx="0" ref="P2" default="50"/>
</parameters>
```

### External Function Contract

**Function Signature:**

```python
def my_function(trapez_value: float, *params) -> float:
    """
    External function contract for compliance calculations.
    
    The function is called FOUR times per compliance calculation:
    - Once for trapezeA
    - Once for trapezeB
    - Once for trapezeC
    - Once for trapezeD
    
    Args:
        trapez_value: Single trapez value (A, B, C, or D)
        *params: Resolved parameter values in XML declaration order
                 (patient-specific values or defaults)
    
    Returns:
        Transformed trapez value
    
    Requirements:
        1. Must accept (trapez_value, *params) signature
        2. Return type must be float (or time-duration compatible)
        3. Must preserve ordering: f(A) <= f(B) <= f(C) <= f(D)
        4. Should validate parameter count/types internally
    """
    pass
```

**Requirements:**
- Function must accept `(trapez_value, *params)` signature
- Return type must be `float` (for value-constraint) or compatible with `parse_duration()` (for time-constraint, returned as float seconds)
- **Ordering constraint:** `f(A) <= f(B) <= f(C) <= f(D)` (enforced at validation time)
- **Parameter trust model:** The function is responsible for:
  - Checking parameter count (`len(params)`)
  - Extracting parameters in correct order (`weight = params[0], age = params[1]`)
  - Type validation (if needed)

**Error Handling:**
- Raise `ValueError` if parameter count is incorrect
- Raise `TypeError` if parameter types are incompatible
- All exceptions are caught by `apply_external_function()` and logged

### Testing External Functions

**Unit test example:**

```python
# filepath: unittests/test_external_functions.py
from core.tak.external_functions import REPO

def test_custom_dosage_adjustment():
    func = REPO["custom_dosage"]
    
    # Test normal case
    assert func(10, 70, 30) == 700  # 10 * 70, age < 65
    
    # Test elderly adjustment
    assert func(10, 70, 70) == 560  # 10 * 70 * 0.8
    
    # Test ordering preservation
    results = [func(x, 72, 50) for x in [0, 0.2, 0.6, 1]]
    assert results == sorted(results), "Function must preserve ordering"
    
    # Test parameter validation
    import pytest
    with pytest.raises(ValueError, match="requires 2 parameters"):
        func(10, 70)  # Missing age parameter
```

**Key Testing Points:**
1. ✅ Correct transformation for normal cases
2. ✅ Correct transformation for edge cases (e.g., elderly)
3. ✅ Ordering preservation: `f(A) <= f(B) <= f(C) <= f(D)`
4. ✅ Parameter validation (count/type checks)

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

### Example 2: Raw Concept (Parameterized)

**File:** `parameterized-raw-concepts/M-SHR_MEASURE.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<parameterized-raw-concept name="M-SHR_MEASURE">
    <categories>Measurements</categories>
    <description>Measurement of M-SHR ratio (glucose / first glucose measure)</description>
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    <parameters>
        <parameter name="FIRST_GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="P1" default="120"/>
    </parameters>
    <functions>
        <function name="div">
            <value idx="0"/>
            <parameter ref="P1"/>
        </function>
    </functions>
</parameterized-raw-concept>
```

**Input from cache:**
```
PatientId | ConceptName           | StartDateTime       | Value
1000      | GLUCOSE_MEASURE       | 2024-01-01 08:00:00 | (100,)
1000      | FIRST_GLUCOSE_MEASURE | 2024-01-01 07:00:00 | (50,)
```

**OutputPatientData:**
```
PatientId | ConceptName     | StartDateTime       | Value   | AbstractionType
1000      | M-SHR_MEASURE   | 2024-01-01 08:00:00 | (2.0,)  | raw-concept
```

---

### Example 3: Event (Multi-Source with OR logic)

**File:** `events/DISGLYCEMIA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<event name="DISGLYCEMIA_EVENT">
    <categories>Events</categories>
    <description>Dysglycemia event (hypo or hyper)</description>
    
    <derived-from>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="A1"/>
        <attribute name="HYPOGLYCEMIA" tak="raw-concept" idx="0" ref="A2"/>
    </derived-from>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="or">
            <attribute ref="A1">
                <allowed-value max="70"/>  <!-- <= 70 mg/dL -->
            </attribute>
            <attribute ref="A2">
                <allowed-value equal="True"/>
            </attribute>
        </rule>
    </abstraction-rules>
</event>
```

**Logic:** Emit "Hypoglycemia" if **either** glucose <= 70 **OR** HYPOGLYCEMIA flag is True.

---

### Example 4: State (Discretization + Merging)

**File:** `states/GLUCOSE_MEASURE.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<state name="GLUCOSE_MEASURE_STATE">
    <categories>Measurements</categories>
    <description>Glucose state abstraction</description>
    
    <derived-from name="GLUCOSE_MEASURE" tak="raw-concept"/>
    
    <persistence good-after="24h" interpolate="true" max-skip="1"/>
    
    <abstraction-rules>
        <rule value="Hypoglycemia" operator="and">
            <attribute idx="0">
                <allowed-value min="0" max="70"/>
            </attribute>
        </rule>
        ...
    <abstraction-rules>
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

### Example 5: Trend (Slope-Based)

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

### Example 6: Context (Windowing + Clipping)

**File:** `contexts/BASAL_BITZUA.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<context name="BASAL_BITZUA_CONTEXT">
    <categories>Medicine</categories>
    <description>Basal insulin influence context</description>
    
    <derived-from>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="A1"/>
    </derived-from>

    <clippers>
        <clipper name="DEATH" tak="raw-concept" clip-before="0s" clip-after="120y"/>
    </clippers>
    
    <abstraction-rules>
        <rule value="Low" operator="or">
            <attribute ref="A1">
                <allowed-value min="0" max="20"/>
            </attribute>
        </rule>
    </abstraction-rules>
    
    <context-windows>
        <persistence value="Low" good-before="0h" good-after="12h"/>
    </context-windows>
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

### Example 7: Pattern (Glucose Measure on Admission)

**File:** `patterns/GLUCOSE_MEASURE_ON_ADMISSION.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="GLUCOSE_MEASURE_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Captures if glucose was measured within reasonable time of admission</description>
    
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
        <attribute name="GLUCOSE_MEASURE" tak="raw-concept" idx="0" ref="E1"/>
    </derived-from>
 
    <abstraction-rules>
        <rule>
            <!-- Context: Patient must have diabetes diagnosis -->
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>

            <!-- Temporal relation: Glucose measure AFTER admission, within 8h -->
            <temporal-relation how='before' max-distance='8h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>

            <!-- Compliance: Ideal 0-8h, acceptable up to 12h -->
            <compliance-function>
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
                    </function>
                </time-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
```

**Scenario 1: Full Compliance (within 4h)**

```
Input:
  - ADMISSION @ 08:00
  - GLUCOSE_MEASURE @ 10:00 (value=120)
  - DIABETES_DIAGNOSIS_CONTEXT: [06:00-20:00, "True"]

Output:
  PatientId | ConceptName                        | StartDateTime | EndDateTime | Value | TimeConstraintScore | ValueConstraintScore
  1000      | GLUCOSE_MEASURE_ON_ADMISSION_PAT. | 08:00         | 10:00       | True  | 1.0                 | None
  
# Time score = 1.0 (gap = 2h, within [0h, 8h])
# Value score = None (no value-constraint defined)
# Combined = 1.0 → "True"
```

**Scenario 2: Partial Compliance (10h gap)**

```
Input:
  - ADMISSION @ 08:00
  - GLUCOSE_MEASURE @ 18:00 (value=120)
  - DIABETES_DIAGNOSIS_CONTEXT: [06:00-20:00, "True"]

Output:
  PatientId | ... | Value   | TimeConstraintScore | ...
  1000      | ... | Partial | 0.5                 | ...
  
# Time score = 0.5 (gap = 10h, in [8h, 12h] → linear decay)
#   Score = (12h - 10h) / (12h - 8h) = 2/4 = 0.5
# Combined = 0.5 → "Partial"
```

**Scenario 3: Pattern Not Found (no glucose measure within 12h)**

```
Input:
  - ADMISSION @ 08:00
  - (no glucose measure within 12h)

Output:
  PatientId | ... | StartDateTime | EndDateTime | Value | TimeConstraintScore | ...
  1000      | ... | NaT           | NaT         | False | None                | ...
  
# Pattern not detected → return False with NaT times
```

---

### Example 8: Pattern with Value Compliance (Insulin Dosage)

**File:** `patterns/INSULIN_ON_ADMISSION.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<pattern name="INSULIN_ON_ADMISSION_PATTERN" concept-type="local-pattern">
    <categories>Admission</categories>
    <description>Insulin (BASAL/BOLUS) within reasonable time + appropriate dosage</description>
    
    <derived-from>
        <attribute name="DIABETES_DIAGNOSIS_CONTEXT" tak="context" ref="C1"/>
        <attribute name="ADMISSION_EVENT" tak="event" ref="A1"/>
        <attribute name="BASAL_BITZUA" tak="raw-concept" idx="0" ref="E1"/>
        <attribute name="BOLUS_BITZUA" tak="raw-concept" idx="0" ref="E2"/>
    </derived-from>

    <!-- Parameter: Patient weight (used for dosage/kg calculation) -->
    <parameters>
        <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
    </parameters>

    <abstraction-rules>
        <rule>
            <context>
                <attribute ref="C1">
                    <allowed-value equal="True"/>
                </attribute>
            </context>

            <temporal-relation how='before' max-distance='72h'>
                <anchor>
                    <attribute ref="A1">
                        <allowed-value equal="True"/>
                    </attribute>
                </anchor>
                <event select='first'>
                    <attribute ref="E1">
                        <allowed-value min="0"/>
                    </attribute>
                    <attribute ref="E2">
                        <allowed-value min="0"/>
                    </attribute>
                </event>
            </temporal-relation>

            <compliance-function>
                <!-- Time compliance: 0-48h ideal, up to 72h acceptable -->
                <time-constraint-compliance>
                    <function name="id">
                        <trapeze trapezeA="0h" trapezeB="0h" trapezeC="48h" trapezeD="72h"/>
                    </function>
                </time-constraint-compliance>
                
                <!-- Value compliance: Dosage should be 0.2-0.6 units/kg -->
                <value-constraint-compliance>
                    <target>
                        <attribute ref="E1"/>  <!-- BASAL dosage -->
                        <attribute ref="E2"/>  <!-- BOLUS dosage -->
                    </target>
                    <function name="mul">
                        <parameter ref="P1"/>  <!-- Weight -->
                        <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
                    </function>
                </value-constraint-compliance>
            </compliance-function>
        </rule>
    </abstraction-rules>
</pattern>
```

**Scenario: Full Compliance**

```
Input:
  - ADMISSION @ 08:00
  - BASAL_BITZUA @ 10:00 (dosage=25 units)
  - WEIGHT_MEASURE @ 07:00 (value=72 kg)
  - DIABETES_DIAGNOSIS_CONTEXT: [06:00-20:00, "True"]

Calculation:
  - Time score: 2h gap → score = 1.0 (within [0h, 48h])
  - Value score:
    - External function: mul(72, [0, 0.2, 0.6, 1]) → [0, 14.4, 43.2, 72]
    - Actual dosage = 25 units → within [14.4, 43.2] → score = 1.0
  - Combined: (1.0 + 1.0) / 2 = 1.0 → "True"

Output:
  PatientId | ... | Value | TimeConstraintScore | ValueConstraintScore
  1000      | ... | True  | 1.0                 | 1.0
```

**Scenario: Partial Compliance (Dosage Too High)**

```
Input:
  - ADMISSION @ 08:00
  - BASAL_BITZUA @ 10:00 (dosage=60 units)
  - WEIGHT_MEASURE @ 07:00 (value=72 kg)

Calculation:
  - Time score: 1.0 (within ideal window)
  - Value score:
    - Trapez = [0, 14.4, 43.2, 72]
    - Actual = 60 units → in [43.2, 72] → linear decay
    - Score = (72 - 60) / (72 - 43.2) = 12 / 28.8 ≈ 0.42
  - Combined: (1.0 + 0.42) / 2 = 0.71 → "Partial"

Output:
  PatientId | ... | Value   | TimeConstraintScore | ValueConstraintScore
  1000      | ... | Partial | 1.0                 | 0.42
```

---

## Pattern Design Best Practices

### 1. Choose Appropriate `max-distance`

**Rule:** `max-distance` should be **≥ trapezeD** (if using time-constraint compliance).

**Rationale:** Pattern matching uses `max-distance` to filter candidates. If `max-distance < trapezeD`, valid instances may be missed.

```xml
<!-- ❌ Bad: max-distance too small -->
<temporal-relation how='before' max-distance='24h'>...</temporal-relation>
<time-constraint-compliance>
    <trapeze trapezeA="0h" trapezeB="0h" trapezeC="24h" trapezeD="48h"/>
    <!-- Pattern will NEVER find instances in [24h, 48h] gap! -->
</time-constraint-compliance>

<!-- ✅ Good: max-distance covers full trapez -->
<temporal-relation how='before' max-distance='48h'>...</temporal-relation>
<time-constraint-compliance>
    <trapeze trapezeA="0h" trapezeB="0h" trapezeC="24h" trapezeD="48h"/>
</time-constraint-compliance>
```

### 2. Use `select="first"` for Clinical Guidelines

**Use Case:** Detect compliance with **first** action after triggering event.

```xml
<!-- Example: First glucose measure after admission -->
<event select='first'>
    <attribute ref="E1">
        <allowed-value min="0"/>
    </attribute>
</event>
```

**Alternative:** Use `select="last"` for end-of-episode checks.

### 3. Design Trapez for Clinical Thresholds

**Time-Constraint Example (Glucose on Admission):**

```xml
<!-- Clinical guideline: Glucose within 8h ideal, up to 12h acceptable -->
<trapeze trapezeA="0h" trapezeB="0h" trapezeC="8h" trapezeD="12h"/>
```

**Value-Constraint Example (Insulin Dosage):**

```xml
<!-- Clinical guideline: 0.2-0.6 units/kg ideal, up to 1.0 acceptable -->
<function name="mul">
    <parameter ref="P1"/>  <!-- Weight -->
    <trapeze trapezeA="0" trapezeB="0.2" trapezeC="0.6" trapezeD="1"/>
</function>
```

### 4. Context for Population Filtering

Use `<context>` to restrict patterns to relevant patient subgroups:

```xml
<!-- Only detect pattern in diabetic patients -->
<context>
    <attribute ref="C1">
        <allowed-value equal="True"/>
    </attribute>
</context>
```

**Note:** Context must **overlap** the interval `[anchor.start, event.start]` (or `[anchor.start, event.end]` for overlap patterns).

### 5. Parameters for Patient-Specific Thresholds

Use `<parameters>` to incorporate patient attributes into compliance scoring:

```xml
<!-- Weight-based dosage threshold -->
<parameters>
    <parameter name="WEIGHT_MEASURE" tak="raw-concept" idx="0" ref="P1" default="72"/>
</parameters>
```

**Resolution Strategy:**
- Use **closest record to pattern start time** (minimize `|time_diff|`)
- Fallback to `default` if no data found


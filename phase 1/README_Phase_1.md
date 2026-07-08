# Phase 1: Data Ingestion & Domain-First Sequence Isolation

This document covers **Phase 1** of our Heimes (2008) Turbofan Remaining Useful Life (RUL) Rebuild project using the **NASA C-MAPSS FD001** dataset.

---

## 1. Domain Challenge & Engineering Rationale

### Why Standard Tabular Pipelines Fail
In standard tabular machine learning, datasets are frequently loaded into a global `DataFrame` and shuffled randomly across train/test splits. 

In Engine Health Management (EHM), **row shuffling destroys physical causality**.
- Each C-MAPSS FD001 unit represents an individual two-spool turbofan engine operating from an initial healthy state until catastrophic failure.
- Sensor observations at cycle $t$ depend cumulatively on mechanical wear from cycles $1 \dots t-1$.
- Consequently, **temporal order within each engine unit is immutable**.

### Stripping the Explicit Clock (`cycle_number`)
We deliberately exclude the `cycle_number` from the input feature matrix (`setting_1` to `setting_3` + `sensor_1` to `sensor_21`).
- **Why?** Feeding `cycle_number` directly into a recurrent model allows the network to take an unphysical shortcut: counting elapsed cycles rather than learning actual degradation symptoms.
- By omitting the clock, the neural network **must** infer health degradation implicitly from shifting thermodynamic gas-path variables (such as high-pressure compressor discharge temperature and core speed).

---

## 2. Ingestion & Transformation Sequence (`phase_1.py`)

```mermaid
flowchart LR
    A[train_FD001.txt<br/>Space-Separated] --> B[Schema Assignment<br/>26 Named Columns]
    B --> C[Fleet Sequence Grouping<br/>engine_id: 1..100]
    C --> D[Feature Matrix Extraction<br/>24 Continuous Channels]
```

1. **Raw Ingestion**: Loads space-separated data arrays without native headers (`train_FD001.txt`).
2. **Schema Assignment**: Maps explicit 26-column headers (`engine_id`, `cycle_number`, `setting_1..3`, `sensor_1..21`).
3. **Chronological Sequence Grouping**: Iterates through `engine_id` $1 \dots 100$, isolating each engine's lifetime into an independent NumPy sequence matrix.
4. **Feature Selection**: Truncates metadata (`engine_id`, `cycle_number`), yielding a clean `[Sequence_Length, 24]` tensor per engine.

---

## 3. Verified C-MAPSS FD001 Invariants

Running `phase_1.py` confirms that all physical and structural dataset invariants are satisfied:

```text
Total Number of engines:    100
Shortest engine sequence:   128 cycles (Engine #39)
Longest engine sequence:    362 cycles (Engine #69)
Feature channels per unit:  24 channels (3 settings + 21 sensors)
```

---

## 4. Execution

To run Phase 1 data ingestion and verify engine sequence isolation locally in Spyder or terminal:
```bash
python phase_1.py
```

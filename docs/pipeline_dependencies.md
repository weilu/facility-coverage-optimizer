# Pipeline Dependency Analysis

## Complete Pipeline Dependency Graph

```
EXTRACT PHASE
═════════════════════════════════════════════════════════════════════════════

[Independent - can run in parallel]
┌────────────────────────┐     ┌────────────────────────┐
│ E1: WorldPop Raster    │     │ E2: WB Boundaries      │
│     Download           │     │     Cache (Admin0/1/2) │
│     ~2 min             │     │     ~5 min (one-time)  │
└───────────┬────────────┘     └───────────┬────────────┘
            │                              │
            ▼                    ┌─────────┴─────────┐
┌────────────────────────┐      │                   │
│ E3: Population Table   │      ▼                   ▼
│     (country-level)    │   ┌─────────────┐   ┌─────────────┐
│     ~7 min             │   │ E4: LGU     │   │ E5: Province│
└────────────────────────┘   │ Boundaries  │   │ Boundaries  │──────┐
                             │ (country)   │   │ (per prov)  │      │
                             └─────────────┘   └─────────────┘      │
                                                                    ▼
                                                           ┌─────────────────┐
                                                           │ E6: Facilities  │
                                                           │ (per province)  │
                                                           │ [OSM or File]   │
                                                           └─────────────────┘

TRANSFORM PHASE (per province × distance combination)
═════════════════════════════════════════════════════════════════════════════

[Load from Extract outputs]
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ T1: Load       │  │ T2: Load       │  │ T3: Load       │  │ T4: Load LGU   │
│ Boundaries     │  │ Facilities     │  │ Population     │  │ Boundaries     │
│ (E5)           │  │ (E6)           │  │ (E3)           │  │ (E4)           │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │                   │
        ▼                   │                   │                   │
┌────────────────┐          │                   │                   │
│ T5: Filter Pop │◄─────────┼───────────────────┘                   │
│ to AOI (H3)    │          │                                       │
│ ~4 min         │          │                                       │
└───────┬────────┘          │                                       │
        │                   │                                       │
        ├───────────────────┼───────────────────────────────────────┤
        │                   │                                       │
        ▼                   ▼                                       │
┌────────────────┐  ┌────────────────┐                              │
│ T6: Generate   │  │ T7: Facilities │                              │
│ Potential Locs │  │ to Spark + H3  │                              │
│ (grid/kmeans)  │  └───────┬────────┘                              │
└───────┬────────┘          │                                       │
        │                   │                                       │
        ▼                   │                                       │
┌────────────────┐          │                                       │
│ T8: Potential  │          │                                       │
│ Locs + H3      │          │                                       │
└───────┬────────┘          │                                       │
        │                   │                                       │
        ├───────────────────┤                                       │
        │                   │                                       │
        ▼                   ▼                                       │
┌────────────────┐  ┌────────────────┐                              │
│ T9: Compute    │  │ T10: Compute   │                              │
│ Potential      │  │ Facility       │                              │
│ Coverage ~8min │  │ Coverage ~8min │                              │
└───────┬────────┘  └───────┬────────┘                              │
        │                   │                                       │
        └─────────┬─────────┘                                       │
                  │                                                 │
                  ▼                                                 │
        ┌────────────────┐                                          │
        │ T11: Analyze   │                                          │
        │ Current Access │                                          │
        └───────┬────────┘                                          │
                │                                                   │
                ▼                                                   │
        ┌────────────────┐                                          │
        │ T12: Prepare   │                                          │
        │ Optimization   │                                          │
        │ Inputs         │                                          │
        └───────┬────────┘                                          │
                │                                                   │
                ▼                                                   │
        ┌────────────────┐                                          │
        │ T13: Greedy    │                                          │
        │ MCLP Optimize  │                                          │
        │ (sequential)   │                                          │
        └───────┬────────┘                                          │
                │                                                   │
                ├───────────────────────────────────────────────────┤
                │                                                   │
                ▼                                                   ▼
        ┌────────────────┐                              ┌────────────────┐
        │ T14: Visualize │                              │ T15: Build     │
        │ Pareto + Maps  │                              │ H3→LGU Mapping │
        └────────────────┘                              └───────┬────────┘
                                                                │
                                                                ▼
                                                        ┌────────────────┐
                                                        │ T16: Compute   │
                                                        │ LGU Metrics    │
                                                        │ (per greedy    │
                                                        │  step)         │
                                                        └───────┬────────┘
                                                                │
                                                                ▼
                                                        ┌────────────────┐
                                                        │ T17: Write     │
                                                        │ LGU Results    │
                                                        └────────────────┘
```

## Summary Table

| Step | Depends On | Output | Parallelizable? |
|------|-----------|--------|-----------------|
| **EXTRACT** |
| E1: WorldPop Raster | - | `.tif` file | Yes (with E2) |
| E2: WB Cache | - | `.geojson` files | Yes (with E1) |
| E3: Population Table | E1 | UC table | Yes (with E4-E6) |
| E4: LGU Boundaries | E2 | UC table | Yes (with E3, E5-E6) |
| E5: Province Boundaries | E2 | UC table (per prov) | Yes (across provinces) |
| E6: Facilities | E5 | UC table (per prov) | Yes (across provinces) |
| **TRANSFORM** |
| T1-T4: Load data | Extract outputs | DataFrames | Yes (parallel loads) |
| T5: Filter Pop to AOI | T1, T3 | `population_aoi` table | - |
| T6: Generate Potential | T1, T5 | DataFrame | Yes (with T7) |
| T7: Facilities to Spark | T2 | DataFrame | Yes (with T6) |
| T8: Add H3 to Potential | T6 | DataFrame | Yes (with T7) |
| T9: Potential Coverage | T5, T8 | UC table | Yes (with T10) |
| T10: Facility Coverage | T5, T7 | UC table | Yes (with T9) |
| T11: Analyze Current | T5, T10 | metrics | - |
| T12: Prepare Optim | T9, T10 | dicts | - |
| T13: Greedy MCLP | T12 | results list | **Sequential** |
| T14: Visualize | T13 | maps | Yes (with T15-T17) |
| T15: H3→LGU Mapping | T4, T5 | DataFrame | - |
| T16: LGU Metrics | T13, T15 | DataFrame | - |
| T17: Write Results | T16 | UC table | - |

## Cross-Phase Dependencies (Extract → Transform)

```
Transform Step          Requires Extract Steps
─────────────────────────────────────────────
T1: Load Boundaries  →  E2 → E5
T2: Load Facilities  →  E2 → E5 → E6
T3: Load Population  →  E1 → E3
T4: Load LGU         →  E2 → E4
```

**Minimum extract completion before transform can start:**
- Path 1: E1 → E3 (population)
- Path 2: E2 → E5 → E6 (boundaries + facilities)
- Path 3: E2 → E4 (LGU - only needed for T15+)

Note: E4 (LGU) is only needed late in transform (T15), so transform could
theoretically start before E4 completes if restructured.

## Key Bottlenecks

1. **T13 (Greedy MCLP)** - Inherently sequential; each step depends on previous
2. **T9/T10 (Coverage computation)** - Slowest steps (~8 min each) but can run in parallel
3. **E3 (Population Table)** - ~7 min, blocks T3/T5

## Parallelization Opportunities

### Within Extract:
- E1 || E2 (fully parallel)
- E3 || E4 || E5 (after E1/E2 respectively)
- E5(province A) || E5(province B) || ... (across provinces)
- E6(province A) || E6(province B) || ... (across provinces, after respective E5)

### Within Transform:
- T6 || T7 (generate potential || convert facilities)
- T9 || T10 (coverage computations)
- T14 || T15 (visualization || LGU mapping)

### Across Province×Distance Combinations:
- Each (province, distance) combination is independent
- See Phase 4 TODO for Spark parallelization investigation

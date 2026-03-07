# Optimization Approach: Greedy vs MIP Solvers

## Problem Overview

The Maximum Covering Location Problem (MCLP) selects facility locations to maximize population coverage within a distance threshold.

**Our problem size:**
- ~70,000 facilities (1,258 existing + ~69,000 potential grid locations)
- ~200,000 demand points (H3 cells after aggregation from 10M+ population pixels)
- Millions of coverage pairs (facility → H3 cell relationships)

## Approaches Considered

### 1. Gurobi (Commercial MIP Solver)

**Pros:**
- Exact optimal solution
- Industry-leading performance (10-100x faster than open-source)
- Good handling of large-scale MIP problems

**Cons:**
- Commercial license required (~$10k+/year)
- License management complexity in Databricks
- Still requires collecting all data to driver memory

**Verdict:** Could potentially solve this problem in 10-60 minutes on a high-memory machine, but license cost and infrastructure complexity make it impractical for this use case.

### 2. CBC/PuLP (Open-Source MIP Solver)

**Pros:**
- Free and open-source
- No license management
- Bundled with PuLP Python library

**Cons:**
- 10-100x slower than Gurobi
- Higher memory footprint
- Crashed kernel on our problem size (~270k binary variables)

**Verdict:** Not viable for problems of this scale.

### 3. Greedy Approximation (Selected Approach)

**Pros:**
- Runs in seconds (O(p × |candidates|) complexity)
- Minimal memory footprint - processes incrementally
- No external dependencies
- Proven (1 - 1/e) ≈ 63.2% worst-case approximation guarantee
- In practice achieves 90%+ of optimal for spatial coverage problems

**Cons:**
- Not guaranteed optimal
- No optimality gap certificate

**Verdict:** Best fit for production use at this scale.

## Why Greedy Works Well Here

1. **Theoretical guarantee:** The 63.2% bound is a worst-case lower bound. The greedy algorithm for submodular maximization (which includes max coverage) is provably optimal among polynomial-time algorithms unless P=NP.

2. **Practical performance:** For facility location with smooth spatial coverage patterns, greedy typically finds near-optimal solutions. The worst-case instances are contrived and rarely occur in real geographic data.

3. **Incremental results:** Greedy naturally produces a Pareto frontier (results for p=1, 2, 3, ... facilities) in a single pass, whereas MIP would need to solve separately for each p.

4. **Robustness:** No solver crashes, no memory issues, deterministic results.

## Infrastructure Challenges

Even with a capable solver, we faced infrastructure bottlenecks:

| Issue | Cause | Impact |
|-------|-------|--------|
| Driver OOM | Collecting 10M+ population points | Crashed before optimization |
| Executor OOM | Caching large coverage tables | Killed executor processes |
| Kernel crash | CBC building 270k-variable model | Unresponsive notebook |

The greedy approach avoids all of these by:
- Aggregating to H3 cells (~200k vs 10M points)
- Processing candidates incrementally
- Using simple Python data structures

## Recommendation

Use the greedy approximation for production. If exact optimality is required for specific analyses:

1. Reduce problem size (coarser grid, fewer candidates)
2. Run Gurobi on a dedicated high-memory machine outside Databricks
3. Consider regional decomposition (solve smaller subproblems)

## References

- Nemhauser, Wolsey, Fisher (1978): "An analysis of approximations for maximizing submodular set functions" - proves the (1-1/e) bound
- Feige (1998): "A threshold of ln(n) for approximating set cover" - proves the bound is optimal unless P=NP

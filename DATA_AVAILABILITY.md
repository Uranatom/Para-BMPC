# Data and reproducibility

The exported data are frozen measurements from the manuscript studies. Outcomes, failures, null findings and every retained policy seed are preserved. CSV files contain scientific identifiers needed for pairing, not neural-network weights or action-level training archives.

## Dataset map

| Directory | Content |
| --- | --- |
| `results/controller_comparison` | 6,144 primary rows, 96 isolated replays represented by per-decision samples, and tuned-reference contrast checks |
| `results/learning_curves` | 7,800 per-cohort records for 26 independently trained runs; three SAC seeds and two DDPG/PPO seeds |
| `results/ablations` | 16,384 unique primary rows covering action, structural and horizon studies, timing summaries, model probes, 160 contrasts and 64 equivalence tests |
| `results/solver_horizon` | 3,968 solver-study rows, timing, 360 paired contrasts and descriptive summaries |
| `results/deployment_diagnostics` | 6,656 merged rows, including reused reference conditions; 4,608 trajectories and 72 isolated replays are additional to the main evidence |
| `results/fixed_selection` | Development grid and disjoint merged-confirmation records for the return-selected constant controller |

These counts overlap across studies and must not be added as independent safety replications. The common formal starts comprise 128 observations per scenario/population cell, 512 overall. `stress` in identifiers denotes the manuscript's feasible-boundary population. Initial manifests contain the environment's realized SV mode to reproduce traffic generation; that field is not an input to any released controller.

## Statistical conventions

Main displayed values average the retained seed-specific means equally. Paired-start intervals average seeds within each of 128 shared starts. SAC training-seed intervals use three seed means (df=2). DDPG/PPO learning bands use two seeds (df=1); SAC learning bands use df=2. Twenty-cohort trailing smoothing occurs within each training seed before averaging.

Main fixed-reference contrasts use separate twenty-comparison Bonferroni families. Action contrasts use 60 comparisons; structural contrasts use 80 and the secondary fixed-risk comparison 20. Horizon equivalence uses 64 TOST comparisons, with intervals compatible with two one-sided tests. Solver contrast families are recorded per CSV row (60/240/60). Deployment diagnostics use a separate exploratory family of 120 comparisons on reused starts and frozen policies. An unresolved difference is not evidence of equivalence.

Table III subtracts unrounded equal-seed means from the tuned fixed reference, then rounds. Rate differences are percentage points. Mean seed-specific isolated p99 in the table differs from the pooled isolated-decision CDF. Solver Table I averages case outcomes; latency spans four case-specific p99 values. Table II takes the median of four case-specific p99 values, never a pooled percentile.

Learning weight-only merging policies remain ineligible under their original selection criterion, despite later collision-free physical completion. Publication learning axes emphasize mean trends and may clip uncertainty bands; the raw curves are retained. Fixed search is finite and scenario-specific; the merged confirmation pool preserves earlier confirmed finalists. Formal outcomes did not select constants.

No trained policy is distributed at this stage. Recomputing recorded results and running baseline shared-model examples do not reproduce proposed-policy rollouts, training or fresh CarSim validation. Independent CarSim use requires a separately licensed installation and vehicle setup. `configs/` provides the numerical common model and recorded start definitions.

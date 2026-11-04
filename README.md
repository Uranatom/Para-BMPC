# Para-BMPC review materials

Review-stage partial release for **Parameter-Adaptive Risk-Aware Branch Model Predictive Control for Interactive Autonomous Driving via Reinforcement Learning**.

This repository is private during review. Access requires authorization from the authors. It provides frozen benchmark results, result-reproduction scripts and baseline implementations. The complete author-developed training/evaluation code, trained policies, configurations and supporting reproducibility materials will be released after publication. Third-party components retain their original licenses; CarSim requires a separate license.

## What can be reproduced now

- Tables I–III, the main fixed-reference contrasts, all 160 ablation contrasts, 64 horizon equivalence tests, 360 solver contrasts and 120 deployment contrasts can be recomputed from exported measurements.
- Result plots for Figures 4–9 can be redrawn from the exported records. Reproduced plot formatting may differ from the publication PDFs in `paper_figures/`; numeric data and uncertainty definitions are retained.
- Fixed nominal/tuned BMPC, FOT-PID, CVaR-TMPC and Era-BMPC adaptations, plus shooting/CEM/weighted MPPI/LTV-QP study implementations, run in the shared CPU model.
- Proposed-policy training, policy inference and fresh CarSim validation require components outside this review release. Recorded CarSim results are distinct from the CPU examples.

## Recompute the results

Use Python 3.10 or newer; the verified environment uses Python 3.11 and the pinned NumPy/SciPy/Matplotlib versions. This path needs neither PyTorch, a GPU, CarSim nor policy weights.

```bash
python -m pip install -r requirements-results.txt
python analysis/reproduce.py
```

Outputs go to `generated/`. `generated/reproduction_report.json` records numerical checks. Pass `--no-plots` for statistics only or `--output PATH` for another output directory. Expected publication numbers are used for validation, not as substitutes for calculations from episode records.

## Run baseline CPU examples

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install -e . --no-deps
python -m review_bmpc.demo --method fixed_tuned --scenario merge --steps 5
python -m review_bmpc.demo --method era --scenario intersection --steps 5
python -m review_bmpc.demo --method branch --scenario merge --boundary --steps 5
python -m review_bmpc.demo --method fot_pid --scenario merge --steps 5
python -m pytest -q
```

Run the editable installation from this checkout; the examples find configuration and pinned reference files relative to the source tree. CPU PyTorch is sufficient. A CUDA-enabled PyTorch build also works, but these commands explicitly select CPU. Supported methods are `fixed_nominal`, `fixed_tuned`, `fot_pid`, `era`, `branch`, `shooting`, `cem`, `mppi` and `ltv`. `branch` is the CVaR-TMPC display method. `--index 0..127` selects a recorded start; `--boundary` selects the tighter avoidable-start population. The fixed-controller solver examples support `--horizon 10/15/20/30`; tuned/published/FOT settings remain at N10. Examples are short shared-model demonstrations, not population or real-time claims.

The baseline controller API is `solve(scenario, ego, other, predictions, probabilities, parameters, record_plan=False)`. It receives observable states and hypothesis beliefs, never the realized SV mode. `DATA_AVAILABILITY.md` describes pairing, seed aggregation and diagnostic reuse. `THIRD_PARTY.md` records original methods and source attribution. `RELEASE_SCOPE.md` specifies the protected boundary and future release.

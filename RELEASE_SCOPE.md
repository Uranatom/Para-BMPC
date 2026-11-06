# Release scope

This repository is a public, open-source partial release. Anyone may access the included materials; author-owned files are available under the MIT License without prior author approval. The release covers the files present here.

Included: common vehicle/interaction/rectangle models, fixed predictive-control and solver implementations, adapted published comparators, initial-state manifests, fixed-controller search/selection evidence, recorded outcomes for every retained seed, recorded learning curves, isolated timing, analysis and baseline examples. Shared controller code is necessary to make the baseline adaptations operational.

Withheld until publication: actor/critic networks and learning updates, categorical clearance replay/bootstrap machinery, PID-Lagrangian training integration, auxiliary training collectors, GPU learner/training orchestration, policy/critic weights, optimizer/recovery states and replay buffers. The pure measured-deficit formula is shared as a benchmark metric and contains no learned model.

The complete author-developed training and evaluation code, trained policies, configurations and required reproducibility materials will be released after publication. The MIT License applies to the author-owned materials currently included here; the complete release will include its applicable license notices. Third-party code retains its own terms; commercial CarSim binaries, native databases and credentials are not distributed.

No reviewer correspondence, private revision ledgers, original Git history, access credentials or machine-local paths are part of this package. This repository begins with an independent history. Access to these public materials does not require an invitation or author approval.

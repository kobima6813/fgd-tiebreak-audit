# FGD's reported advantage depends on score rounding and name-order tie-breaking

A reproduction study of *Fragmentation Gradient Descent* (FGD), the GPU-sharing scheduler from
"Beware of Fragmentation" (Weng et al., USENIX ATC '23), on the Alibaba `cluster-trace-gpu-v2023`
trace (1,213 nodes, 6,212 GPUs, 8,152 tasks).

**Write-up:** a short blog-style summary of the findings is in [BLOG.md](BLOG.md).

**Summary**

1. The released FGD code gets its result from two implementation details that the paper does not
   describe: scores are rounded to integers in 0..100, and ties are broken by lowest node name.
   With the random tie-breaking that upstream Kubernetes uses, the same code leaves **10.3%** of GPU
   capacity unallocated instead of **5.0%**, which is worse than the plain GPU-packing baseline (5.8%).
2. About 81% of the capacity that released FGD leaves unallocated sits in slivers smaller than
   200 milli-GPU. Most of it comes from one request size (810 milli, 13% of tasks). No placement
   policy can recover it, because the scheduler may not refuse a task that fits.
3. A 48-weight linear scoring rule found by evolutionary search leaves **4.2%** unallocated at full
   load without relying on tie-breaking, and beats released FGD on 22 of 24 workloads. It is worse
   than FGD below about 95% load.
4. A public project that reports an LLM-written policy beating FGD by 31% is compared against a
   weaker FGD than the authors' code. Against the released FGD semantics, that policy is slightly
   worse on the production mix and much worse on workloads with GPU-model constraints.

Everything here is simulation on one public trace. See [Limitations](#limitations).

## Setup and validation

`fgdsim.py` is a from-scratch simulator (Python + Numba) that follows the semantics of the authors'
simulator: CPU, memory and GPU-model feasibility, whole and fractional GPUs, the fragmentation
measure in `pkg/utils/frag.go`, the FGD score in `pkg/simulator/plugin/fgd_score.go`, and workload
inflation by resampling tasks up to 130% of cluster GPU capacity. One run takes 0.1 to 0.3 seconds.

On the scenario the authors publish as sample output (FGD, inflation 1.3, seed 42), their simulator
reports 95.3% GPU, 84.0% CPU and 62.3% memory allocated. This simulator reports 95.6%, 84.6% and
62.9% with a different random task order. An independent re-implementation
([MixSched](https://github.com/athish-shubhan/mixsched)) also reports 5.0% unallocated for FGD at
100% arrived demand.

The metric throughout is the paper's: unallocated GPU capacity when the cumulative GPU demand that
has arrived equals 100% of cluster capacity.

## Finding 1: where FGD's advantage comes from

The paper's Algorithm 1 picks the node that minimises the increase in fragmentation, a real-valued
quantity. The released code does two more things:

- `fgd_score.go` converts the score to an integer: `int64(sigmoid(delta/1000) * 100)`. With only
  about a hundred possible values, hundreds of nodes tie on most decisions.
- The vendored `selectHost` in `generic_scheduler.go` is patched to pick the lexicographically
  smallest node name among ties. A code comment says this is for reproducibility. The reservoir
  sampling that upstream kube-scheduler uses is commented out.

Together these fill nodes in a fixed order, which is first-fit packing layered on top of FGD.

Default workload, 10 task orders (seeds 1001 to 1010):

| Policy | Unallocated at 100% | sd | Unallocated at 130% |
|---|---|---|---|
| FGD as released (integer scores, ties by node name) | 5.04% | 0.17 | 4.84% |
| FGD as released, node names shuffled | 4.77% | 0.08 | 4.60% |
| FGD real-valued, exact ties kept, ties by node name | 5.91% | 0.13 | 5.68% |
| FGD real-valued (plain floats), ties by node name | 7.39% | 0.35 | 7.16% |
| FGD as released, random tie-breaking | 10.28% | 0.34 | 10.04% |
| FGD real-valued, random tie-breaking | 10.48% | 0.36 | 10.24% |
| GPU packing | 5.81% | 0.16 | 5.60% |
| Best-fit | 7.12% | 0.26 | 6.91% |
| Random | 13.04% | 0.41 | 12.23% |

What this shows:

- Tie-breaking is the dominant factor. Any fixed node order works (shuffling the names changes
  little); random selection among ties roughly doubles the waste.
- Integer rounding is worth about another 0.9 points. Real-valued FGD with exact ties kept is on par
  with GPU packing (5.91% vs 5.81%).
- A team that implements FGD from the paper inside stock Kubernetes should not expect the paper's
  numbers unless it also reproduces these two details.

The paper does not mention tie-breaking or score rounding. A short literature and repository search
found no earlier report of this dependence; that is not proof that none exists. The observation
needs confirmation from the authors or an independent reproduction.

## Finding 2: most of the remaining waste is not a placement problem

At the end of one run of released FGD (seed 1001), 297 GPUs' worth of capacity is unallocated:

| Where | GPUs |
|---|---|
| slivers under 100 milli | 37.8 |
| slivers of 100 to 200 milli | 203.1 |
| slivers of 200 to 400 milli | 23.6 |
| slivers of 400 milli or more | 12.7 |
| fully idle GPUs | 20 |

13.2% of tasks request exactly 810 milli-GPU. Each leaves 190 milli, and only 1.3% of tasks are
small enough to use such a remainder. The evolved rule below removes nearly everything else: of
its 254 unallocated GPUs, 97% are slivers under 200 milli and no GPU is fully idle. The headroom
left for placement policies on this benchmark is therefore small.

## Finding 3: an evolved scoring rule

`evolve.py` runs CMA-ES over a linear rule with 4 task classes x 12 features (48 weights), starting
from real-valued FGD and evaluated with random tie-breaking. It trained on 5 workloads and 6 task
orders; the weights in `weights/evolved.json` were selected on separate validation orders after
75 generations (7,200 simulations, about 14 minutes on 2 cores).

- Default workload, 10 held-out task orders: 4.19% unallocated vs 5.04% for released FGD
  (difference 0.85 points, standard error 0.06, better in 10 of 10).
- All 24 published workloads, 3 held-out task orders each: better by more than 0.05 points on 22,
  worse on 2 (`cpu300` by 0.42, `gpushare100` by 0.18). Mean 5.79% vs 6.67%. 19 of the 24 workloads
  were never seen in training.
- The result does not depend on tie-breaking (4.19% with random or name-order ties).

Cost: the rule was optimised only for 100% load and above. When arrivals reach 90% of capacity it
has already failed to place 1.23% of capacity, against 0.01% for released FGD, and it rejects more
multi-GPU tasks. It is not a drop-in replacement for a cluster that runs below about 95% load.

Using search to discover scheduling policies on this trace is not new; see finding 4.

## Finding 4: independent check of a published "beats FGD" claim

[Backfill](https://github.com/techstar9797/GPUoptimizer) (published 11 October 2026) has an LLM
write scoring policies and evaluates them on the same trace. Its README reports 311 idle GPUs for
the agent's policy against 452 for FGD on the production mix.

Observations from the project's own files:

- Its FGD baseline is the real-valued score, without integer rounding or name-order tie-breaking.
- Its results file lists 401 and 423 idle GPUs for that baseline on the production mix. The 452 in
  the README is close to the project's random baseline (450 and 456).

Independent comparison in this simulator, with arrivals sampled until demand reaches 100% of
capacity (idle GPUs, lower is better; 10 task orders for the production mix, 4 for the others):

| Workload | Backfill, as published | Backfill rule, re-implemented | FGD as released | Evolved rule |
|---|---|---|---|---|
| default (production mix) | 311 | 325 | 307 | 262 |
| gpushare40 | 395 | 401 | 379 | 345 |
| multigpu20 | 270 | 287 | 289 | 233 |
| gpuspec33 | 1309 | 1259 | 738 | 653 |
| gpushare80 | 665 | 669 | 690 | 672 |
| multigpu50 | 260 | 239 | 178 | 166 |
| gpuspec20 | 728 | 727 | 358 | 271 |
| gpushare60 | 538 | 542 | 560 | 518 |
| mean | 560 | 556 | 437 | 390 |

On the production mix the Backfill rule is worse than released FGD in 10 of 10 task orders, by
18 GPUs on average. It is better on the two heavy-sharing workloads and far worse where tasks are
pinned to GPU models.

Method note: Backfill's code was **not executed**. The policy was re-implemented from the source
text recorded in the project's results snapshot, and the re-implementation reproduces the project's
published per-workload numbers closely (first two columns). The project's FGD baseline is
reproduced less closely, because its tie-breaking (by sorted node state) differs from this
simulator's. This is a check of one headline comparison, not a judgement of the project.

## Limitations

- Simulation only. Nothing was run on a real cluster.
- One trace from one operator. No evidence that the numbers transfer.
- Tasks only arrive and never finish, as in the original evaluation.
- The simulator matches the authors' published sample output to within 0.3 to 0.6 points, not exactly.
  It has not been reviewed by a second person.
- The baselines other than FGD (packing, best-fit, random) are re-implementations and use random
  tie-breaking.

## Reproduce

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/alibaba/clusterdata.git
(cd clusterdata && git sparse-checkout set cluster-trace-gpu-v2023)
pip install -r requirements.txt

python experiments.py validate         # comparison with the authors' sample output
python experiments.py mechanism        # finding 1 (about 30 s)
python experiments.py leftover         # finding 2
python experiments.py generalization   # finding 3 (about 1 min)
python experiments.py audit            # finding 4 (about 3 min)
python evolve.py 900                   # optional: rerun the search
```

The trace is expected in `./clusterdata/cluster-trace-gpu-v2023/csv`, or set `FGD_TRACE_DIR`.
Outputs of the runs reported above are in `results/`.

## Provenance

The experiments, the simulator and this write-up were produced by Claude (an AI model made by
Anthropic) in a single working session, at the direction of the repository owner. Corrections are
welcome: open an issue.

## References

- Q. Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation
  Gradient Descent", USENIX ATC 2023. https://www.usenix.org/conference/atc23/presentation/weng
- Authors' simulator: https://github.com/hkust-adsl/kubernetes-scheduler-simulator
- Trace: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
- Backfill: https://github.com/techstar9797/GPUoptimizer
- MixSched: https://github.com/athish-shubhan/mixsched

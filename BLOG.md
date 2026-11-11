# FGD's reported advantage depends on score rounding and name-order tie-breaking

Fragmentation Gradient Descent (FGD), the GPU-sharing scheduler from Weng et al. (USENIX ATC '23), reports a clear edge over packing on the Alibaba `cluster-trace-gpu-v2023` workload. A reproduction on that same trace finds that the edge depends on two implementation details the paper does not describe.

In the released code, `fgd_score.go` rounds the score to an integer in 0..100, and a patched `selectHost` in the vendored `generic_scheduler.go` breaks ties by lowest node name. The reservoir sampling used by upstream Kubernetes is commented out. With the random tie-breaking that stock kube-scheduler uses, the same released FGD leaves **10.3%** of GPU capacity unallocated instead of **5.0%**, which is worse than the plain GPU-packing baseline (**5.8%**). Confirmation from the original authors is pending.

The numbers below are from a public reproduction, [fgd-tiebreak-audit](https://github.com/kobima6813/fgd-tiebreak-audit). Everything here is simulation on one public trace.

## What was reproduced

FGD schedules GPU-sharing tasks by picking the node that minimises the increase in fragmentation. The authors released a Kubernetes-based simulator and evaluated it on Alibaba's 2023 GPU cluster trace: 1,213 nodes, 6,212 GPUs, and 8,152 tasks. The paper's metric is unallocated GPU capacity when cumulative arrived demand equals 100% of cluster capacity.

The reproduction is a from-scratch simulator that follows the authors' semantics, including CPU, memory and GPU-model feasibility, whole and fractional GPUs, the fragmentation measure, the FGD score, and workload inflation by resampling tasks. On the scenario the authors publish as sample output (FGD, inflation 1.3, seed 42), their simulator reports 95.3% GPU, 84.0% CPU and 62.3% memory allocated. This simulator reports 95.6%, 84.6% and 62.9% with a different random task order. An independent re-implementation, [MixSched](https://github.com/athish-shubhan/mixsched), also reports 5.0% unallocated for FGD at 100% arrived demand.

## Two details the paper does not describe

Algorithm 1 in the paper picks the node that minimises a real-valued increase in fragmentation. The released code does two more things.

First, `fgd_score.go` converts the score to an integer: `int64(sigmoid(delta/1000) * 100)`. With only about a hundred possible values, hundreds of nodes tie on most decisions.

Second, the vendored `selectHost` is patched to pick the lexicographically smallest node name among ties. A code comment says this is for reproducibility. Upstream kube-scheduler's reservoir sampling is commented out.

Together, these fill nodes in a fixed order: first-fit packing layered on top of FGD.

Default workload, 10 task orders (seeds 1001 to 1010):

| Policy | Unallocated at 100% | sd |
|---|---|---|
| FGD as released (integer scores, ties by node name) | 5.04% | 0.17 |
| FGD as released, node names shuffled | 4.77% | 0.08 |
| FGD real-valued, exact ties kept, ties by node name | 5.91% | 0.13 |
| FGD as released, random tie-breaking | 10.28% | 0.34 |
| GPU packing | 5.81% | 0.16 |
| Random | 13.04% | 0.41 |

Tie-breaking is the dominant factor. Any fixed node order works — shuffling the names changes little — while random selection among ties roughly doubles the waste. Integer rounding is worth about another 0.9 points. Real-valued FGD with exact ties kept is on par with GPU packing (5.91% versus 5.81%).

A team that implements FGD from the paper inside stock Kubernetes should not expect the paper's numbers unless it also reproduces these two details. A short literature and repository search found no earlier report of this dependence; that is not proof that none exists. The observation needs confirmation from the authors or an independent reproduction.

## Most of the remaining waste is not a placement problem

At the end of one run of released FGD (seed 1001), 297 GPUs' worth of capacity is unallocated. About 81% of it sits in slivers smaller than 200 milli-GPU: 37.8 GPUs in slivers under 100 milli, and 203.1 GPUs in slivers of 100 to 200 milli. Only 20 GPUs are fully idle.

The source is mostly one request size. 13.2% of tasks request exactly 810 milli-GPU. Each leaves 190 milli, and only 1.3% of tasks are small enough to use such a remainder. No placement policy can recover that capacity, because the scheduler may not refuse a task that fits. The headroom left for placement policies on this benchmark is therefore small.

## A 48-weight rule found by search

An evolutionary search (CMA-ES) over a linear rule with 4 task classes × 12 features — 48 weights — starts from real-valued FGD and is evaluated with random tie-breaking. Trained on 5 workloads and 6 task orders, the selected weights leave **4.19%** unallocated on the default workload over 10 held-out task orders, against 5.04% for released FGD (difference 0.85 points, standard error 0.06, better in 10 of 10). Across all 24 published workloads it is better by more than 0.05 points on 22 and worse on 2. The result does not depend on tie-breaking.

That is not a drop-in replacement. The rule was optimised only for 100% load and above. When arrivals reach 90% of capacity it has already failed to place 1.23% of capacity, against 0.01% for released FGD, and it rejects more multi-GPU tasks. It is worse than FGD below about 95% load.

## A check of one later comparison

A public project, [Backfill](https://github.com/techstar9797/GPUoptimizer) (published 11 October 2026), reports an LLM-written policy beating FGD by 31% on the same trace: 311 idle GPUs against 452 for FGD on the production mix.

The project's own files show that its FGD baseline is the real-valued score, without integer rounding or name-order tie-breaking, and that the 452 figure in the README is close to the project's random baseline (450 and 456). Independently re-implementing the published Backfill rule — the project's code was not executed — yields 325 idle GPUs on the production mix, against 307 for FGD as released. On that mix the Backfill rule is worse than released FGD in 10 of 10 task orders, by 18 GPUs on average. It is better on two heavy-sharing workloads and far worse where tasks are pinned to GPU models.

This is a check of one headline comparison, not a judgement of the project.

## Limitations

- Simulation only. Nothing was run on a real cluster.
- One trace from one operator. There is no evidence that the numbers transfer.
- Tasks only arrive and never finish, as in the original evaluation.
- The simulator matches the authors' published sample output to within 0.3 to 0.6 points, not exactly. It has not been reviewed by a second person.
- The baselines other than FGD (packing, best-fit, random) are re-implementations and use random tie-breaking.

## Please confirm or correct

The experiments, the simulator, and the repository write-up were produced by Claude, an AI model made by Anthropic, in a single working session at the direction of the repository owner. Confirmation from the original FGD authors is pending.

Code, results, and a short reproduction script are at [github.com/kobima6813/fgd-tiebreak-audit](https://github.com/kobima6813/fgd-tiebreak-audit). If the finding is wrong, incomplete, or already known, a correction would be gratefully received — an issue on that repository, or a note from the authors, would settle it.

## References

- Q. Weng et al., ["Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent"](https://www.usenix.org/conference/atc23/presentation/weng), USENIX ATC 2023.
- [Authors' simulator](https://github.com/hkust-adsl/kubernetes-scheduler-simulator)
- [Alibaba cluster-trace-gpu-v2023](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023)

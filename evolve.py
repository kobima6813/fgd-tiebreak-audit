"""Search for a scoring rule with CMA-ES (finding 3).

    export FGD_TRACE_DIR=/path/to/clusterdata/cluster-trace-gpu-v2023/csv
    python evolve.py 900        # wall-clock budget in seconds; writes weights/evolved_new.json

The rule is linear: score(node, task) = sum_k w[class(task), k] * feature_k(node after placing the task),
with 4 task classes (CPU-only, shared GPU, one whole GPU, several GPUs) x 12 features = 48 weights (see _evo in fgdsim.py).
Search starts from real-valued FGD (weight 1 on the FGD delta, 0 elsewhere) and is scored with random tie-breaking.
Objective: mean allocated GPU share when arrived demand is 100%, 110%, 120% and 130% of capacity.
Weights are selected on validation streams that the search never optimises on.
"""
import json, os, sys, time
import numpy as np, cma, multiprocessing as mp
import fgdsim as G

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = [('default', 1), ('default', 2), ('gpushare40', 3), ('multigpu30', 4), ('gpuspec20', 5), ('cpu100', 6)]
VAL = [('default', 101), ('default', 102), ('gpushare40', 103), ('multigpu30', 104), ('gpuspec20', 105), ('cpu100', 106)]
XS = [1.0, 1.1, 1.2, 1.3]
ENV, ST = {}, {}


def init():
    nodes = G.load_nodes()
    for tr, sd in TRAIN + VAL:
        if tr not in ENV:
            ENV[tr] = G.Env(tr, nodes)
        ST[(tr, sd)] = ENV[tr].stream(1.3, sd)


def ev(args):
    w, tasks = args; out = []
    for tr, sd in tasks:
        e = ENV[tr]; r = e.run(G.P_EVO, ST[(tr, sd)], w, seed=sd, tie=1)
        out.append(float(np.mean(e.curve(r, XS))))
    return out


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 900
    init(); pool = mp.Pool(os.cpu_count(), initializer=init)
    x0 = np.zeros(G.K * G.NCLS); x0[[0, 12, 24, 36]] = 1.0
    es = cma.CMAEvolutionStrategy(x0, 0.4, {'popsize': 16, 'seed': 7, 'verbose': -9})
    t0 = time.time(); gen = 0; best = -1
    while time.time() - t0 < budget:
        X = es.ask(); F = pool.map(ev, [(x, TRAIN) for x in X], chunksize=1)
        f = [-np.mean(v) for v in F]; es.tell(X, f); gen += 1
        if gen % 5 == 0:
            m = es.mean.copy(); xb = X[int(np.argmin(f))]
            v = pool.map(ev, [(m, VAL), (xb, VAL)]); vm, vb = np.mean(v[0]), np.mean(v[1])
            cand, vv = (m, vm) if vm >= vb else (xb, vb)
            if vv > best:
                best = vv
                json.dump({'w': list(map(float, cand)), 'val': float(vv), 'gen': gen},
                          open(os.path.join(HERE, 'weights', 'evolved_new.json'), 'w'))
            print(f'gen {gen}  t={time.time() - t0:.0f}s  train best {-min(f):.4f}  validation {vv:.4f}  best validation {best:.4f}', flush=True)

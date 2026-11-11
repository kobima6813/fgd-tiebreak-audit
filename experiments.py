"""Reproduce every number in the README.

    export FGD_TRACE_DIR=/path/to/clusterdata/cluster-trace-gpu-v2023/csv
    python experiments.py validate         # compare against the sample output published by the FGD authors
    python experiments.py mechanism        # finding 1: tie-breaking and score rounding
                                           # (lost@90% = arrived demand that could not be placed when arrivals reach 90% of capacity)
    python experiments.py leftover         # finding 2: where the unallocated GPU capacity sits
    python experiments.py generalization   # finding 3: evolved rule on all 24 workloads
    python experiments.py audit            # finding 4: independent check of a published "beats FGD" claim

Each command prints a table and writes results/<name>.json. Runs on a laptop CPU in seconds to minutes.
"""
import glob, json, os, sys
import numpy as np
import fgdsim as G

HERE = os.path.dirname(os.path.abspath(__file__))
W_EVO = np.array(json.load(open(os.path.join(HERE, 'weights', 'evolved.json')))['w'])


def save(name, obj):
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(obj, open(os.path.join(HERE, 'results', name + '.json'), 'w'), indent=1)


def permuted(nodes, seed=5):
    p = np.random.default_rng(seed).permutation(len(nodes['ng']))
    return {k: v[p].copy() for k, v in nodes.items()}


def unalloc(env, res, x):
    """Unallocated GPU capacity (% of cluster) when arrived GPU demand reaches x * capacity."""
    return float(100 * (1 - env.curve(res, [x])[0]))


def validate():
    e = G.Env('default'); nd = e.nodes; st = e.stream(1.3, 42)
    r = e.run(3, st, tie=0)
    print('FGD as released, workload inflated to 130% of GPU capacity, seed 42')
    print(f"  this simulator : GPU {100 * r['final']:.1f}%  CPU {100 * (1 - r['cpu'].sum() / nd['cap_cpu'].sum()):.1f}%  "
          f"memory {100 * (1 - r['mem'].sum() / nd['cap_mem'].sum()):.1f}%")
    print('  authors\' sample: GPU 95.3%  CPU 84.0%  memory 62.3%   (experiments/README.md of their simulator; different random stream)')


def mechanism():
    nodes = G.load_nodes(); e = G.Env('default', nodes); es = G.Env('default', permuted(nodes))
    variants = [
        ('FGD as released (integer scores, ties by node name)', e, 3, 0, None),
        ('FGD as released, node names shuffled', es, 3, 0, None),
        ('FGD real-valued, exact ties kept, ties by node name', e, 6, 0, None),
        ('FGD real-valued (plain floats), ties by node name', e, 4, 0, None),
        ('FGD as released, random tie-breaking (upstream Kubernetes)', e, 3, 1, None),
        ('FGD real-valued, random tie-breaking', e, 6, 1, None),
        ('GPU packing', e, 2, 1, None), ('Best-fit', e, 1, 1, None), ('Random', e, 0, 1, None),
        ('Evolved rule (random tie-breaking)', e, 5, 1, W_EVO),
    ]
    seeds = range(1001, 1011); out = {}
    print(f"{'policy':62s} {'@100%':>8s} {'sd':>6s} {'@130%':>8s} {'lost@90%':>9s}")
    for name, env, pol, tie, w in variants:
        a, b, c = [], [], []
        for s in seeds:
            st = e.stream(1.3, s); r = env.run(pol, st, w, seed=s, tie=tie)
            a.append(unalloc(env, r, 1.0)); b.append(100 * (1 - r['final'])); c.append(unalloc(env, r, 0.9) - 10.0)
        out[name] = dict(at100=a, at130=b, lost_at90=c)
        print(f"{name:62s} {np.mean(a):7.2f}% {np.std(a, ddof=1):6.2f} {np.mean(b):7.2f}% {np.mean(c):8.2f}%")
    save('mechanism', out)


def leftover():
    e = G.Env('default'); st = e.stream(1.3, 1001); out = {}
    for name, pol, tie, w in [('FGD as released', 3, 0, None), ('Evolved rule', 5, 1, W_EVO)]:
        gl = e.run(pol, st, w, seed=1001, tie=tie)['gl']; part = gl[(gl > 0) & (gl < 1000)]
        bins = [(1, 100), (100, 200), (200, 400), (400, 1000)]
        row = dict(total_gpus=float(gl.sum() / 1000), fully_idle_gpus=int((gl == 1000).sum()),
                   slivers={f'{a}-{b} milli': float(part[(part >= a) & (part < b)].sum() / 1000) for a, b in bins})
        out[name] = row; print(name, json.dumps(row))
    sp = e.tr['specs'][e.tr['pod_spec']]
    print('share of tasks requesting exactly 810 milli-GPU: %.1f%%' % (100 * ((sp[:, 2] == 1) & (sp[:, 3] == 810)).mean()))
    save('leftover', out)


def generalization():
    nodes = G.load_nodes(); out = {}
    traces = sorted(os.path.basename(f)[len('openb_pod_list_'):-4] for f in glob.glob(G.D + 'openb_pod_list_*.csv'))
    train = {'default', 'gpushare40', 'multigpu30', 'gpuspec20', 'cpu100'}
    print(f"{'workload':14s} {'FGD':>7s} {'packing':>8s} {'evolved':>8s}   (unallocated GPU %, at 100% arrived demand, 3 seeds)")
    for t in traces:
        e = G.Env(t, nodes); rows = []
        for s in (2001, 2002, 2003):
            st = e.stream(1.3, s)
            rows.append([unalloc(e, e.run(3, st, seed=s, tie=0), 1.0), unalloc(e, e.run(2, st, seed=s, tie=1), 1.0),
                         unalloc(e, e.run(5, st, W_EVO, seed=s, tie=1), 1.0)])
        m = np.mean(rows, 0); out[t] = dict(fgd=m[0], packing=m[1], evolved=m[2], seen_in_training=t in train)
        print(f"{t:14s} {m[0]:6.2f}% {m[1]:7.2f}% {m[2]:7.2f}% {'  (training workload)' if t in train else ''}")
    d = np.array([v['evolved'] - v['fgd'] for v in out.values()])
    print(f"evolved better by >0.05pp on {(d < -0.05).sum()}/{len(d)}, worse by >0.05pp on {(d > 0.05).sum()}/{len(d)}; "
          f"mean FGD {np.mean([v['fgd'] for v in out.values()]):.2f}%, mean evolved {np.mean([v['evolved'] for v in out.values()]):.2f}%")
    save('generalization', out)


def audit():
    """Arrivals sampled i.i.d. until arrived GPU demand reaches 100% of capacity; metric = idle GPUs at that point."""
    nodes = G.load_nodes(); out = {}
    wls = ['default', 'gpushare40', 'multigpu20', 'gpuspec33', 'gpushare80', 'multigpu50', 'gpuspec20', 'gpushare60']
    print(f"{'workload':12s} {'FGD(authors)':>13s} {'Backfill rule':>14s} {'Backfill FGD':>13s} {'evolved':>8s}   (idle GPUs)")
    for wl in wls:
        e = G.Env(wl, nodes); cap = e.cap / 1000; rows = []
        for s in range(10 if wl == 'default' else 4):
            st = e.stream_iid(100 + s)
            rows.append([(1 - e.run(3, st, seed=s, tie=0)['final']) * cap,
                         (1 - e.run(7, st, seed=s, tie=1, top60=True)['final']) * cap,
                         (1 - e.run(8, st, seed=s, tie=0, top60=True)['final']) * cap,
                         (1 - e.run(5, st, W_EVO, seed=s, tie=1)['final']) * cap])
        r = np.array(rows); out[wl] = r.tolist(); m = r.mean(0)
        print(f"{wl:12s} {m[0]:13.1f} {m[1]:14.1f} {m[2]:13.1f} {m[3]:8.1f}   Backfill rule better than FGD(authors) in {(r[:, 1] < r[:, 0]).sum()}/{len(r)} runs")
    m = np.mean([np.mean(v, 0) for v in out.values()], 0)
    print(f"{'mean of 8':12s} {m[0]:13.1f} {m[1]:14.1f} {m[2]:13.1f} {m[3]:8.1f}")
    save('audit', out)


if __name__ == '__main__':
    cmds = dict(validate=validate, mechanism=mechanism, leftover=leftover, generalization=generalization, audit=audit)
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()

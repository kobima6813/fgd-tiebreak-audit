"""fgdsim: a fast re-implementation of the GPU-sharing scheduling simulation used in
"Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent" (USENIX ATC'23).

Semantics follow the authors' simulator (hkust-adsl/kubernetes-scheduler-simulator: pkg/utils/frag.go,
pkg/simulator/plugin/fgd_score.go, and the patched selectHost in the vendored kube-scheduler).

Policies (first argument of Env.run):
  0 random            1 best-fit            2 GPU packing
  3 FGD as released   (integer score floor(100*sigmoid(delta/1000)))
  4 FGD real-valued   (score = delta, as described in the paper)
  6 FGD real-valued, rounded to 1e-3 milli-GPU (removes floating-point noise, keeps exact ties)
  5 evolved linear scoring rule (weights in weights/evolved.json)
  7 scoring formula published by the "Backfill" project (re-implemented from its description, not its code)
  8 real-valued delta with best-fit GPU inside the node (Backfill-style FGD baseline)
tie=0 breaks score ties by lowest node name (as the released simulator does); tie=1 breaks them uniformly at random
(as upstream kube-scheduler does).

Set FGD_TRACE_DIR to the csv folder of alibaba/clusterdata cluster-trace-gpu-v2023.
"""
import os
import numpy as np, pandas as pd, numba as nb

D = os.environ.get('FGD_TRACE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clusterdata', 'cluster-trace-gpu-v2023', 'csv')).rstrip('/') + '/'
MODELS = ['A10', 'G2', 'G3', 'P100', 'T4', 'V100M16', 'V100M32']
NEG = -1e300
K = 12          # features per pod class
NCLS = 4        # cpu-only, shared gpu, one whole gpu, multi gpu

P_RANDOM, P_BESTFIT, P_PACKING, P_FGDQ, P_FGDX, P_EVO = 0, 1, 2, 3, 4, 5


def load_nodes():
    n = pd.read_csv(D + 'openb_node_list_gpu_node.csv').sort_values('sn').reset_index(drop=True)
    return dict(cap_cpu=n.cpu_milli.values.astype(np.int64), cap_mem=n.memory_mib.values.astype(np.int64),
                ng=n.gpu.values.astype(np.int64), ntype=n.model.map({m: i for i, m in enumerate(MODELS)}).values.astype(np.int64),
                model=n.model.values)


def spec_mask(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0
    m = 0
    for x in s.split('|'):
        if x:
            m |= (1 << MODELS.index(x)) if x in MODELS else (1 << 20)
    return m


def load_trace(name):
    p = pd.read_csv(D + f'openb_pod_list_{name}.csv')
    p = p.sort_values('name').reset_index(drop=True)
    milli = np.where(p.num_gpu.values == 0, 0, np.clip(p.gpu_milli.values, 0, 1000)).astype(np.int64)
    mask = np.array([spec_mask(s) for s in (p.gpu_spec.values if 'gpu_spec' in p else [None] * len(p))], dtype=np.int64)
    tab = np.stack([p.cpu_milli.values.astype(np.int64), p.memory_mib.values.astype(np.int64),
                    p.num_gpu.values.astype(np.int64), milli, mask], 1)
    specs, inv = np.unique(tab, axis=0, return_inverse=True)
    return dict(name=name, specs=specs, pod_spec=inv.astype(np.int64), df=p)


def typical_pods(tr, popularity=95, gpu_res_weight=0.0, involve_cpu=True):
    """utils.GetTypicalPods: key = (cpu, milli, num_gpu, type); top specs covering `popularity`% of pods."""
    sp = tr['specs'][tr['pod_spec']]
    cnt = {}
    for cpu, mem, ng, milli, mask in sp:
        if not involve_cpu and ng == 0:
            continue
        w = 1.0
        if gpu_res_weight > 0 and milli == 1000:
            w = 1 + ng * gpu_res_weight
        k = (int(cpu), int(ng), int(milli), int(mask))
        cnt[k] = cnt.get(k, 0.0) + w
    total = sum(cnt.values())
    items = sorted(cnt.items(), key=lambda kv: -kv[1])
    need = popularity * total / 100.0
    out, cum = [], 0.0
    for k, v in items:
        if cum >= need:
            break
        out.append((k, v)); cum += v
    arr = np.array([k for k, _ in out], dtype=np.int64)
    freq = np.array([v for _, v in out]) / cum
    return dict(cpu=arr[:, 0].copy(), ng=arr[:, 1].copy(), milli=arr[:, 2].copy(), mask=arr[:, 3].copy(), freq=freq)


def make_stream(tr, cap_gpu_milli, ratio=1.3, seed=42):
    """Workload tuning as in TunePodsByNodeTotalResource (resample to ratio x cluster GPU), then shuffle."""
    rng = np.random.default_rng(seed)
    ps = tr['pod_spec']
    g = tr['specs'][:, 2] * tr['specs'][:, 3]
    idx = list(range(len(ps)))
    tot = int(g[ps].sum()); tgt = ratio * cap_gpu_milli
    if tot > tgt:
        while tot > tgt:
            j = int(rng.integers(len(idx))); tot -= int(g[ps[idx[j]]]); idx.pop(j)
    else:
        while True:
            j = int(rng.integers(len(ps)))
            if tot + int(g[ps[j]]) > tgt:
                break
            tot += int(g[ps[j]]); idx.append(j)
    idx = np.array(idx); rng.shuffle(idx)
    return ps[idx].astype(np.int64)


@nb.njit(cache=True)
def node_frag(cpu, mem, gl, ng, ntype, t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq):
    tot = 0
    for i in range(ng):
        tot += gl[i]
    if tot == 0:
        return 0.0
    f = 0.0
    for t in range(t_cpu.shape[0]):
        m = t_milli[t]
        if m == 0 or (t_mask[t] != 0 and ((t_mask[t] >> ntype) & 1) == 0) or cpu < t_cpu[t] or mem < t_mem[t]:
            f += t_freq[t] * tot
            continue
        c = 0; small = 0
        for i in range(ng):
            if gl[i] >= m:
                c += 1
            else:
                small += gl[i]
        if c >= t_ng[t]:
            f += t_freq[t] * small
        else:
            f += t_freq[t] * tot
    return f


@nb.njit(cache=True)
def score_node(pol, w, n, cpu, mem, gl, cap_cpu, cap_mem, ngs, ntype, fragb,
               r_cpu, r_mem, r_ng, r_milli, r_mask,
               t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq, rho, tmp, use_fgd_feat):
    """Score placing a pod (r_*) on node n. Returns (score, gpu_index) ; NEG if infeasible."""
    ng = ngs[n]
    if cpu[n] < r_cpu or mem[n] < r_mem:
        return NEG, -1
    if r_mask != 0 and ((r_mask >> ntype[n]) & 1) == 0:
        return NEG, -1
    share = r_ng == 1 and r_milli < 1000
    idle = 0; tot = 0
    for i in range(ng):
        tot += gl[n, i]
        if gl[n, i] == 1000:
            idle += 1
    if r_ng > 0:
        if share:
            ok = False
            for i in range(ng):
                if gl[n, i] >= r_milli:
                    ok = True; break
            if not ok:
                return NEG, -1
        elif idle < r_ng:
            return NEG, -1
    cls = 0
    if r_ng > 0:
        cls = 1 if share else (2 if r_ng == 1 else 3)
    cpu_after = cpu[n] - r_cpu
    tot_after = tot - r_ng * r_milli

    if pol == 0:
        g = -1
        tmp[0] = 0
        if share:
            for i in range(ng):
                if gl[n, i] >= r_milli and (g < 0 or gl[n, i] < gl[n, g]):
                    g = i
        return 0.0, g
    if pol == 1:   # best fit (cpu, gpu) normalised by max spec; best-fit GPU inside node
        g = -1
        if share:
            for i in range(ng):
                if gl[n, i] >= r_milli and (g < 0 or gl[n, i] < gl[n, g]):
                    g = i
        s = 0.5 * cpu_after / 128000.0 + 0.5 * tot_after / 8000.0
        return float(int((1.0 - s) * 100)), g
    if pol == 2:   # gpu packing
        g = -1
        if share:
            for i in range(ng):
                if gl[n, i] >= r_milli and (g < 0 or gl[n, i] < gl[n, g]):
                    g = i
        if idle == ng:
            return float(max(100 // 3 - idle, idle)), g
        if r_ng == 0:
            return 100.0, g
        if share:
            if gl[n, g] == 1000:
                return float(max(100 // 2 - 1, 100 // 3)), g
            return float(max(100 - (gl[n, g] * 100 // 1000) // 10, 50)), g
        return float(max(100 // 2 - r_ng, 100 // 3)), g

    if pol == 7 or pol == 8:   # 7: published agent formula (state part), 8: real-valued delta with best-fit GPU
        g = -1
        for i in range(ng):
            tmp[i] = gl[n, i]
        if share:
            for i in range(ng):
                if gl[n, i] >= r_milli and (g < 0 or gl[n, i] < gl[n, g]):
                    g = i
            tmp[g] -= r_milli
        else:
            k = r_ng
            for i in range(ng):
                if k == 0:
                    break
                if tmp[i] == 1000:
                    tmp[i] = 0; k -= 1
        fa = node_frag(cpu_after, mem[n] - r_mem, tmp, ng, ntype[n], t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq)
        if pol == 8:
            return fragb[n] - fa, g
        used = 0.0; slots = 0
        for i in range(ng):
            used += 1000 - tmp[i]
            if tmp[i] >= 1000:
                slots += 1
        a = -2.5 * fa + (1.0 - cpu_after / cap_cpu[n]) + 0.7 * (1.0 - (mem[n] - r_mem) / cap_mem[n]) + 5.0 * used / (1000.0 * ng + 1e-6) - 2.0 * slots
        return a, g
    # ---- policies that need the after-state fragmentation --------------------------------
    need_frag = (pol == 3 or pol == 4 or pol == 6 or use_fgd_feat)
    for i in range(ng):
        tmp[i] = gl[n, i]
    if not share:
        k = r_ng
        for i in range(ng):
            if k == 0:
                break
            if tmp[i] == 1000:
                tmp[i] = 0; k -= 1
        delta = 0.0
        if need_frag:
            delta = fragb[n] - node_frag(cpu_after, mem[n] - r_mem, tmp, ng, ntype[n], t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq)
        if pol == 3:
            return float(int(100.0 / (1.0 + np.exp(-delta / 1000.0)))), -1
        if pol == 4:
            return delta, -1
        if pol == 6:
            return np.floor(delta * 1000.0 + 0.5) / 1000.0, -1
        sc = _evo(w, cls, delta, 0.0, 0.0, idle - r_ng, idle == ng, cpu_after, tot_after, mem[n] - r_mem,
                  cap_cpu[n], cap_mem[n], ng, rho)
        return sc, -1
    best = NEG; bg = -1
    for i in range(ng):
        if gl[n, i] < r_milli:
            continue
        dup = False
        for j in range(i):
            if gl[n, j] == gl[n, i]:
                dup = True; break
        if dup:
            continue
        tmp[i] -= r_milli
        delta = 0.0
        if need_frag:
            delta = fragb[n] - node_frag(cpu_after, mem[n] - r_mem, tmp, ng, ntype[n], t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq)
        if pol == 3:
            sc = float(int(100.0 / (1.0 + np.exp(-delta / 1000.0))))
        elif pol == 4:
            sc = delta
        elif pol == 6:
            sc = np.floor(delta * 1000.0 + 0.5) / 1000.0
        else:
            was_idle = gl[n, i] == 1000
            sc = _evo(w, cls, delta, tmp[i] / 1000.0, 1.0 if tmp[i] == 0 else 0.0, idle - (1 if was_idle else 0),
                      idle == ng, cpu_after, tot_after, mem[n] - r_mem, cap_cpu[n], cap_mem[n], ng, rho)
        tmp[i] += r_milli
        if sc > best:
            best = sc; bg = i
    return best, bg


@nb.njit(cache=True)
def _evo(w, cls, delta, resid, perfect, idle_after, node_was_idle, cpu_after, tot_after, mem_after,
         cap_cpu, cap_mem, ng, rho):
    o = cls * K
    slack = (cpu_after - rho * tot_after) / cap_cpu      # cpu left relative to what the leftover GPUs will need
    s = w[o + 0] * (delta / 1000.0)
    s += w[o + 1] * resid
    s += w[o + 2] * perfect
    s += w[o + 3] * (idle_after / 8.0)
    s += w[o + 4] * (1.0 if node_was_idle else 0.0)
    s += w[o + 5] * (cpu_after / cap_cpu)
    s += w[o + 6] * (-slack if slack < 0 else 0.0)
    s += w[o + 7] * (slack if slack > 0 else 0.0)
    s += w[o + 8] * (tot_after / (1000.0 * ng))
    s += w[o + 9] * (ng / 8.0)
    s += w[o + 10] * (cap_cpu / (ng * 64000.0))
    s += w[o + 11] * (mem_after / cap_mem)
    return s


@nb.njit(cache=True)
def simulate(pol, w, cap_cpu, cap_mem, ngs, ntype, specs, stream,
             t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq, seed, rho, use_fgd_feat, tie, out_node, out_gpu):
    N = cap_cpu.shape[0]; S = specs.shape[0]
    cpu = cap_cpu.copy(); mem = cap_mem.copy()
    gl = np.zeros((N, 8), np.int64)
    fragb = np.zeros(N)
    tmp = np.zeros(8, np.int64)
    for n in range(N):
        for i in range(ngs[n]):
            gl[n, i] = 1000
        fragb[n] = node_frag(cpu[n], mem[n], gl[n], ngs[n], ntype[n], t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq)
    ver = np.zeros(N, np.int64)
    cver = np.full((S, N), -1, np.int64)
    csc = np.zeros((S, N)); cg = np.zeros((S, N), np.int8)
    state = np.uint64(seed * 2654435761 + 88172645463325252)
    capg = 0
    for n in range(N):
        capg += ngs[n] * 1000
    arrived = 0
    hsh = np.zeros(N, np.int64); mcnt = np.zeros(32); srt = np.zeros(8, np.int64)
    for k in range(stream.shape[0]):
        s = stream[k]
        r_cpu = specs[s, 0]; r_mem = specs[s, 1]; r_ng = specs[s, 2]; r_milli = specs[s, 3]; r_mask = specs[s, 4]
        best = NEG; bn = -1; cnt = 0
        arrived += r_ng * r_milli
        T = 1.0 + 3.0 * arrived / capg
        if pol == 7:
            # refresh cache, then count distinct candidate states per GPU model (the formula's rarity term)
            nf = 0
            for n in range(N):
                if cver[s, n] != ver[n]:
                    sc, g = score_node(pol, w, n, cpu, mem, gl, cap_cpu, cap_mem, ngs, ntype, fragb,
                                       r_cpu, r_mem, r_ng, r_milli, r_mask,
                                       t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq, rho, tmp, use_fgd_feat)
                    csc[s, n] = sc; cg[s, n] = g; cver[s, n] = ver[n]
                if csc[s, n] > NEG:
                    for i in range(8):
                        srt[i] = gl[n, i]
                    srt.sort()
                    h = cpu[n] * 1000003 + mem[n]
                    for i in range(8):
                        h = h * 1009 + srt[i]
                    hsh[nf] = h * 8 + ntype[n]; nf += 1
            for i in range(32):
                mcnt[i] = 0
            if nf > 0:
                hs = np.sort(hsh[:nf])
                for i in range(nf):
                    if i == 0 or hs[i] != hs[i - 1]:
                        mcnt[hs[i] % 8] += 1
        for n in range(N):
            if cver[s, n] != ver[n]:
                sc, g = score_node(pol, w, n, cpu, mem, gl, cap_cpu, cap_mem, ngs, ntype, fragb,
                                   r_cpu, r_mem, r_ng, r_milli, r_mask,
                                   t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq, rho, tmp, use_fgd_feat)
                csc[s, n] = sc; cg[s, n] = g; cver[s, n] = ver[n]
            sc = csc[s, n]
            if pol == 7 and sc > NEG:
                sc = sc * T + 3.5 / mcnt[ntype[n]]
            if sc > best:
                best = sc; bn = n; cnt = 1
            elif tie == 1 and sc == best and sc > NEG:
                cnt += 1
                state ^= state << np.uint64(13); state ^= state >> np.uint64(7); state ^= state << np.uint64(17)
                if state % np.uint64(cnt) == 0:
                    bn = n
        out_node[k] = bn
        if bn < 0:
            out_gpu[k] = -1
            continue
        g = cg[s, bn]; out_gpu[k] = g
        cpu[bn] -= r_cpu; mem[bn] -= r_mem
        if r_ng > 0:
            if g >= 0:
                gl[bn, g] -= r_milli
            else:
                kk = r_ng
                for i in range(ngs[bn]):
                    if kk == 0:
                        break
                    if gl[bn, i] == 1000:
                        gl[bn, i] = 0; kk -= 1
        ver[bn] += 1
        fragb[bn] = node_frag(cpu[bn], mem[bn], gl[bn], ngs[bn], ntype[bn], t_cpu, t_mem, t_ng, t_milli, t_mask, t_freq)
    return cpu, mem, gl


class Env:
    def __init__(self, trace, nodes=None, popularity=95, gpu_res_weight=0.0):
        self.typ_top60 = None
        self.nodes = nodes or load_nodes()
        self.tr = load_trace(trace) if isinstance(trace, str) else trace
        self.cap = int(self.nodes['ng'].sum()) * 1000
        self.typ = typical_pods(self.tr, popularity, gpu_res_weight)
        sp = self.tr['specs'][self.tr['pod_spec']]
        gp = sp[sp[:, 2] > 0]
        self.rho = float(gp[:, 0].sum() / (gp[:, 2] * gp[:, 3]).sum())   # cpu-milli demanded per gpu-milli
        self.gdem = (self.tr['specs'][:, 2] * self.tr['specs'][:, 3]).astype(np.int64)

    def stream(self, ratio=1.3, seed=42):
        return make_stream(self.tr, self.cap, ratio, seed)

    def run(self, pol, stream, w=None, seed=1, use_fgd_feat=True, rho=None, tie=0, top60=False):
        nd, t = self.nodes, (self.top60() if top60 else self.typ)
        if w is None:
            w = np.zeros(K * NCLS)
        on = np.empty(len(stream), np.int64); og = np.empty(len(stream), np.int64)
        cpu, mem, gl = simulate(pol, np.asarray(w, dtype=np.float64), nd['cap_cpu'], nd['cap_mem'], nd['ng'], nd['ntype'],
                                self.tr['specs'], stream, t['cpu'], t.get('mem', np.zeros(len(t['cpu']), np.int64)), t['ng'], t['milli'], t['mask'], t['freq'],
                                seed, self.rho if rho is None else rho, use_fgd_feat, tie, on, og)
        d = self.gdem[stream]
        arrived = np.cumsum(d); alloc = np.cumsum(np.where(on >= 0, d, 0))
        return dict(node=on, gpu=og, arrived=arrived, alloc=alloc, cpu=cpu, mem=mem, gl=gl,
                    final=alloc[-1] / self.cap, failed=int((on < 0).sum()))

    def curve(self, res, xs):
        """allocated GPU fraction of capacity when arrived workload reaches x * capacity"""
        i = np.searchsorted(res['arrived'], np.asarray(xs) * self.cap, side='right') - 1
        return res['alloc'][np.clip(i, 0, len(res['alloc']) - 1)] / self.cap

    def top60(self):
        """Task-shape distribution as in the audited project: top 60 distinct (cpu, mem, gpus, milli, model) shapes."""
        if self.typ_top60 is None:
            sp = self.tr['specs']; cnt = np.bincount(self.tr['pod_spec'], minlength=len(sp))
            o = np.argsort(-cnt, kind='stable')[:60]
            self.typ_top60 = dict(cpu=sp[o, 0].copy(), mem=sp[o, 1].copy(), ng=sp[o, 2].copy(), milli=sp[o, 3].copy(),
                                  mask=sp[o, 4].copy(), freq=cnt[o] / cnt[o].sum())
        return self.typ_top60

    def stream_iid(self, seed):
        """Arrivals sampled i.i.d. with replacement until arrived GPU demand reaches 100% of capacity."""
        rng = np.random.default_rng(seed); ps = self.tr['pod_spec']; out = []; a = 0
        while a < self.cap:
            i = int(rng.integers(len(ps))); out.append(ps[i]); a += int(self.gdem[ps[i]])
        return np.array(out, dtype=np.int64)

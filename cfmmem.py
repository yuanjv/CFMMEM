"""cfmmem.py -- CFMMEM v0.1 + patch 1: model (§8-§12), training (§21), first audit (§16, §23-24).

  python cfmmem.py --cache gut_minilm.pt --no_memory --sigma 0.05    # cue-only baseline = denoising gate
  python cfmmem.py --cache gut_minilm.pt --sigma 0.05                # full model
  python cfmmem.py --cache gut_minilm.pt --label unif                # M_unif   (w = 1)
  python cfmmem.py --cache gut_minilm.pt --label scram               # M_scram  (placebo: weights permuted)

Patch 1 (own additions, not in the spec):
  1. sigma=0 and single-sample reads in eval; gate line on absent-probe cos(u,q)
  2. teacher-slot mixing: reader sometimes reads the stream's Z rows; 'teacher' eval row = perfect-storage bound
  3. delay-0 scaffold probes (annealed to 0) + short->long stream curriculum
  4. smaller lr for the writer + structural soft norm cap inside write()
Needs cfmmem_data.py with the Zc field. Untested at scale: read the first eval table before trusting anything.
"""
import argparse, copy, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Beta
from cfmmem_data import Episodes


# ---------------------------------------------------------------- primitives
class TimeEmb(nn.Module):
    def __init__(self, d_t=128):
        super().__init__()
        f = torch.exp(torch.linspace(0, math.log(32), d_t // 2)) * 2 * math.pi
        self.register_buffer("f", f)
        self.mlp = nn.Sequential(nn.Linear(d_t, d_t), nn.SiLU(), nn.Linear(d_t, d_t))

    def forward(self, t):                                   # t: (B,1)
        a = t * self.f
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class AdaLN(nn.Module):
    def __init__(self, d, d_t):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(d_t, 2 * d)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, e):
        g, b = self.proj(e).chunk(2, -1)
        if x.dim() == 3:
            g, b = g.unsqueeze(1), b.unsqueeze(1)
        return self.ln(x) * (1 + g) + b


class WriterBlock(nn.Module):                               # F8: self-attn(W) + cross-attn(Z); no PE on W
    def __init__(self, d, d_t, heads=4):
        super().__init__()
        self.n1, self.n2, self.n3 = AdaLN(d, d_t), AdaLN(d, d_t), AdaLN(d, d_t)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, W, Z, e):
        h = self.n1(W, e); W = W + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(W, e); W = W + self.ca(h, Z, Z, need_weights=False)[0]
        return W + self.mlp(self.n3(W, e))


class WriterVelocity(nn.Module):                            # v_phi: conditional neural ODE field (F1)
    def __init__(self, d, d_t=128, layers=2):
        super().__init__()
        self.t = TimeEmb(d_t)
        self.zin = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.blocks = nn.ModuleList(WriterBlock(d, d_t) for _ in range(layers))
        self.nf = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # identity start

    def forward(self, W, t, Z):                             # W:(B,K,d) t:(B,1) Z:(B,m,d)
        e, Z = self.t(t), self.zin(Z)
        for blk in self.blocks:
            W = blk(W, Z, e)
        return self.out(self.nf(W))


class ReaderVelocity(nn.Module):                            # v_theta: CFM field; state conditioning differentiable (F3)
    def __init__(self, d, d_t=128, hidden=512, heads=4):
        super().__init__()
        self.t = TimeEmb(d_t)
        self.nq, self.nx, self.nw = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qp = nn.Linear(2 * d, d)                       # address from cue AND current x_t
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ada = AdaLN(3 * d, d_t)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))
        self.s = d ** 0.5

    def forward(self, x, t, q, W):                          # x,q:(N,d) t:(N,1) W:(N,K,d); rows are LayerNormed
        Wn = self.nw(W)
        qry = self.qp(torch.cat([self.nq(q), self.nx(x)], -1)).unsqueeze(1)
        ctx = self.ca(qry, Wn, Wn, need_weights=False)[0].squeeze(1)
        h = torch.cat([x * self.s, q * self.s, ctx], -1)
        return self.mlp(self.ada(h, self.t(t)))


def softcap(W, c):                                          # structural soft norm cap: row norm n -> c*tanh(n/c)
    n = W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return W * (c * torch.tanh(n / c) / n)


# ---------------------------------------------------------------- model
class CFMMEM(nn.Module):
    def __init__(self, K=32, d=384, S_w=3, S_r=6, kappa=2.0, eps=0.1, sigma=0.05,
                 p_uniform=0.2, tau_r=10.0, w_max=5.0, cap_c=1.5):
        super().__init__()
        self.K, self.d, self.S_w, self.S_r = K, d, S_w, S_r
        self.kappa, self.eps, self.sigma, self.cap_c = kappa, eps, sigma, cap_c   # sigma fixed (learned sigma -> 0)
        self.p_uniform, self.tau_r, self.w_max = p_uniform, tau_r, w_max
        self.W_init = nn.Parameter(torch.randn(K, d) * 0.5 / d ** 0.5)
        self.writer, self.reader = WriterVelocity(d), ReaderVelocity(d)
        self.register_buffer("s_ema", torch.ones(()))

    def write(self, W, Z):                                  # S_w Euler steps; source is the state, not noise
        for s in range(self.S_w):
            t = torch.full((W.shape[0], 1), s / self.S_w, device=W.device)
            W = softcap(W + self.writer(W, t, Z) / self.S_w, self.cap_c)
        return W

    def err(self, W, t, q, z, x0):                          # per-sample ||v - (z - x0)||^2 at times t:(N,1)
        xt = (1 - t) * x0 + t * z
        return ((self.reader(xt, t, q, W) - (z - x0)) ** 2).sum(-1)

    @torch.no_grad()
    def read(self, W, q, sigma=None):                       # inference; never writes. sigma=0 -> deterministic
        s0 = self.sigma if sigma is None else sigma
        x = q + s0 * torch.randn_like(q)
        for s in range(self.S_r):
            t = torch.full((q.shape[0], 1), s / self.S_r, device=q.device)
            x = x + self.reader(x, t, q, W) / self.S_r
        return F.normalize(x, dim=-1)

    def sample_t(self, w, absent):                          # F4: Beta tilt + uniform floor; TRAINING ONLY
        tilt = Beta(1 + self.kappa * w, torch.ones_like(w)).sample()
        mix = (torch.rand_like(w) < self.p_uniform) | absent
        return torch.where(mix, torch.rand_like(w), tilt)

    @torch.no_grad()
    def surprise(self, W_pre, zc, z):
        """F5: once, pre-write, detached. Uses the item's OWN PARTIAL CUE zc against its full z.
        (A self-cue q=z makes the target -sigma*xi, solvable without memory, so it measures denoising, not novelty.)"""
        x0 = zc + self.sigma * torch.randn_like(zc)
        e = 0
        for t in (0.125, 0.375, 0.625, 0.875):              # fixed 4-point quadrature
            e = e + self.err(W_pre, torch.full((z.shape[0], 1), t, device=z.device), zc, z, x0)
        return e / 4

    def norm_surprise(self, raw):                           # running normalisation; frozen outside training
        if self.training:
            self.s_ema.mul_(0.95).add_(0.05 * raw.mean())
        return raw / self.s_ema

    def item_weights(self, s, label):                       # log w = log b + log rho   (F12)
        B, T = s.shape
        b = self.eps + s
        delta = torch.arange(T - 1, -1, -1, device=s.device).float()   # writes since encoding
        rho = self.eps + torch.exp(-delta / self.tau_r)
        w = b * rho
        if label == "unif":
            w = torch.ones_like(w)
        elif label == "scram":                              # placebo: permutation within episode
            w = w.gather(1, torch.rand(B, T, device=s.device).argsort(1))
        return w.clamp(max=self.w_max), b


def teacher_state(model, Z):
    """Perfect storage: the stream's Z rows, padded with W_init rows to K rows. Reader rows are LayerNormed,
    so scale is irrelevant; row order is irrelevant (no positional encodings)."""
    B, T, d = Z.shape
    assert T <= model.K, "teacher state needs T <= K"
    pad = model.W_init[: model.K - T].detach().unsqueeze(0).expand(B, -1, -1)
    return torch.cat([Z, pad], 1)


# ---------------------------------------------------------------- one episode
def run_episode(model, bt, L, label, grad, memory=True):
    """No-grad prefix + grad window (F9). Returns final state W, labels, last-write pair for reflow."""
    Z, Zc = bt["Z"], bt["Zc"]
    B, T, d = Z.shape
    W = model.W_init.unsqueeze(0).expand(B, -1, -1)
    raw, disp, last = [], [], None
    for t in range(T):
        with torch.no_grad():
            raw.append(model.surprise(W.detach(), Zc[:, t], Z[:, t]))   # pre-write, detached
        if not memory:
            continue
        win = grad and t >= T - L
        with torch.set_grad_enabled(win):
            W1 = model.write(W, Z[:, t:t + 1])
        if win:
            disp.append((W1 - W).norm(dim=-1).mean().detach())
        if t == T - 1:
            last = (W, W1, Z[:, t:t + 1])
        W = W1
    w, b = model.item_weights(model.norm_surprise(torch.stack(raw, 1)), label)
    return W, dict(w=w, b=b, disp=torch.stack(disp).mean() if disp else torch.zeros(()), last=last)


def probe_terms(model, W, bt, w_item, p_teach=0.0):
    B, P, d = bt["q"].shape
    K = W.shape[1]
    if p_teach > 0:                                         # per-episode: read teacher slots instead of written W
        use = torch.rand(B, 1, 1, device=W.device) < p_teach
        W = torch.where(use, teacher_state(model, bt["Z"]), W)      # grad reaches the writer only where ~use
    q, z, ab = bt["q"].reshape(-1, d), bt["z"].reshape(-1, d), bt["absent"].reshape(-1)
    w = w_item.gather(1, bt["t_idx"].clamp_min(0)).masked_fill(bt["absent"], 1.0).reshape(-1)
    Wp = W.unsqueeze(1).expand(B, P, K, d).reshape(B * P, K, d)
    x0 = q + model.sigma * torch.randn_like(q)
    t = model.sample_t(w, ab).unsqueeze(-1)
    L_read = (w * model.err(Wp, t, q, z, x0)).mean()        # loss -> W -> writer unroll -> phi
    Wo = z.unsqueeze(1).expand(-1, K, -1)                   # oracle state: z as K identical tokens (§16b)
    L_or = model.err(Wo, torch.rand_like(t), q, z, x0).mean()
    return L_read, L_or


def scaffold_loss(model, W, bt, n):
    """Spec §21 scaffolding: probe the last n written items (delay 0..n-1) with their own partial cues, written W only."""
    B, T, d = bt["Z"].shape
    n = min(n, T)
    K = W.shape[1]
    q, z = bt["Zc"][:, -n:].reshape(-1, d), bt["Z"][:, -n:].reshape(-1, d)
    Wp = W.unsqueeze(1).expand(B, n, K, d).reshape(B * n, K, d)
    x0 = q + model.sigma * torch.randn_like(q)
    t = torch.rand(B * n, 1, device=W.device)
    return model.err(Wp, t, q, z, x0).mean()


def reflow_loss(model, last):                               # F2: endpoints detached
    W0, W1, Zl = last
    t = torch.rand(W0.shape[0], 1, 1, device=W0.device)
    Wbar = (1 - t) * W0.detach() + t * W1.detach()
    return ((model.writer(Wbar, t.squeeze(-1), Zl) - (W1.detach() - W0.detach())) ** 2).mean()


def to_dev(bt, dev):
    return {k: v.to(dev) for k, v in bt.items()}


# ---------------------------------------------------------------- evaluation (§16: measure once, label many ways)
@torch.no_grad()
def evaluate(model, ep, args, dev, n_batches=8, n_xi=4, m_min=0.1):
    model.eval()
    grid = torch.tensor([.06, .19, .31, .44, .56, .69, .81, .94], device=dev)   # common, weight-independent
    lo, hi = grid <= .25, grid >= .75
    names = ("mem", "abl", "shuf", "teach")
    R = {k: [] for k in ("cos_q", "seg", "margin", "b", "absent_cos", "absent_cos0")}
    for c in names:
        for k in ("cos_", "cos0_", "C_", "F_"):
            R[k + c] = []
    for _ in range(n_batches):
        bt = to_dev(ep.batch(args.B), dev)
        W, info = run_episode(model, bt, args.L, args.label, grad=False, memory=not args.no_memory)
        B, P, d = bt["q"].shape
        K = W.shape[1]
        rep = lambda X: X.unsqueeze(1).expand(B, P, K, d).reshape(B * P, K, d)
        q, z, ab, mar = bt["q"].reshape(-1, d), bt["z"].reshape(-1, d), bt["absent"].reshape(-1), bt["margin"].reshape(-1)
        keep = mar >= m_min                                 # NaN (absent) compares False
        conds = {"mem": rep(W), "abl": rep(model.W_init.unsqueeze(0).expand(B, -1, -1)),
                 "shuf": rep(W.roll(1, 0)), "teach": rep(teacher_state(model, bt["Z"]))}
        x0s = [q + model.sigma * torch.randn_like(q) for _ in range(n_xi)]      # common random numbers

        def profile(Wc):
            e = torch.zeros(q.shape[0], len(grid), device=dev)
            for x0 in x0s:
                for g, t in enumerate(grid):
                    e[:, g] += model.err(Wc, t.expand(q.shape[0], 1), q, z, x0)
            return e / n_xi

        e_or = profile(z.unsqueeze(1).expand(-1, K, -1))
        for c in names:
            tl = profile(conds[c]) - e_or                   # memory-attributable excess error
            u, u0 = model.read(conds[c], q), model.read(conds[c], q, sigma=0.0)
            R["C_" + c].append(tl[:, lo].mean(1)[keep]); R["F_" + c].append(tl[:, hi].mean(1)[keep])
            R["cos_" + c].append((u * z).sum(-1)[keep]); R["cos0_" + c].append((u0 * z).sum(-1)[keep])
            if c == "mem":
                R["absent_cos"].append((u * q).sum(-1)[ab]); R["absent_cos0"].append((u0 * q).sum(-1)[ab])
        R["cos_q"].append((q * z).sum(-1)[keep])
        R["seg"].append(torch.arange(P, device=dev).expand(B, P).reshape(-1)[keep])
        R["margin"].append(mar[keep])
        R["b"].append(info["b"].gather(1, bt["t_idx"].clamp_min(0)).reshape(-1)[keep])
    model.train()
    return {k: torch.cat(v) for k, v in R.items()}


def did(Fe, Ce, lab, seg, mar, min_cell=8):
    """Difference-in-differences within strata (delay segment x margin tercile), label split at cell median."""
    qm = torch.quantile(mar, torch.tensor([1 / 3, 2 / 3], device=mar.device))
    mb = torch.bucketize(mar, qm)
    sF = sC = n = 0.0
    for k in seg.unique():
        for j in range(3):
            c = (seg == k) & (mb == j)
            if c.sum() < min_cell:
                continue
            high = lab[c] > lab[c].median()
            if high.all() or (~high).all():
                continue
            sF += ((Fe[c][~high].mean() - Fe[c][high].mean()) * c.sum()).item()
            sC += ((Ce[c][~high].mean() - Ce[c][high].mean()) * c.sum()).item()
            n += c.sum().item()
    return (sF / n, sC / n) if n else (float("nan"), float("nan"))


def report(R, P, tag):
    cq = R["cos_q"]
    print(f"\n[{tag}] {cq.numel()} probes (margin >= 0.1) | identity u=q: cos(q,z) = {cq.mean():.3f}")
    print(f"  {'state used at read':<26}{'cos(u,z)':>9}{'cos s=0':>9}{'gain':>8}{'C':>9}{'F':>9}")
    rows = (("mem", "W (written)"), ("abl", "W_init (no writes)"),
            ("shuf", "W of other episode"), ("teach", "teacher (perfect slots)"))
    for c, nm in rows:
        cz = R["cos_" + c]
        print(f"  {nm:<26}{cz.mean():9.3f}{R['cos0_' + c].mean():9.3f}{(cz - cq).mean():8.3f}"
              f"{R['C_' + c].mean():9.3f}{R['F_' + c].mean():9.3f}")
    for c in ("mem", "abl", "teach"):
        g = [(R["cos_" + c] - cq)[R["seg"] == k].mean().item() for k in range(P)]
        print(f"  gain by delay (long->short), {c}: " + " ".join(f"{x:6.3f}" for x in g))
    if R["absent_cos"].numel():
        a, a0 = R["absent_cos"].mean().item(), R["absent_cos0"].mean().item()
        print(f"  absent probes: cos(u,q) sample {a:.3f} | sigma=0 {a0:.3f}   gate (>= 0.95): {'PASS' if a >= 0.95 else 'not yet'}")
    dF, dC = did(R["F_mem"], R["C_mem"], R["b"], R["seg"], R["margin"])
    print(f"  audit (crude strata): dF_b = {dF:.4f}  dC_b = {dC:.4f}  S_b = {dF - dC:.4f}   [compare across full / unif / scram runs]")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="gut_minilm.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--S_w", type=int, default=3)
    ap.add_argument("--S_r", type=int, default=6)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--sigma_cue", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_w", type=float, default=1e-4, help="learning rate for the writer (phi)")
    ap.add_argument("--cap_c", type=float, default=1.5, help="structural soft norm cap on state rows")
    ap.add_argument("--lam_cap", type=float, default=1e-3)
    ap.add_argument("--lam_or", type=float, default=0.25)
    ap.add_argument("--lam_str", type=float, default=0.0)
    ap.add_argument("--p_teach", type=float, default=0.5, help="prob. an episode's probes read the teacher state")
    ap.add_argument("--lam_scaf", type=float, default=1.0)
    ap.add_argument("--scaf_steps", type=int, default=1000, help="linear anneal of scaffold weight to 0")
    ap.add_argument("--n_scaf", type=int, default=3, help="scaffold probes the last n written items")
    ap.add_argument("--T0", type=int, default=4, help="starting stream length of the curriculum")
    ap.add_argument("--curr_steps", type=int, default=1000, help="steps to grow T from T0 to T (0 = off)")
    ap.add_argument("--label", choices=["full", "unif", "scram"], default="full")
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.T <= args.K and args.P <= args.T0 <= args.T, "need P <= T0 <= T <= K"
    if args.no_memory:                                      # keep the cue-only baseline pure
        args.p_teach, args.lam_scaf = 0.0, 0.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tr = Episodes(args.cache, T=args.T, n_probe=args.P, sigma_cue=args.sigma_cue, seed=args.seed)
    ev = copy.copy(tr)                                      # shares tensors; disjoint books for eval (always full T)
    ev.rng, ev.g = random.Random(999), torch.Generator().manual_seed(999)
    held = {b for b in tr.books if b % 10 == 0}
    ev.books, ev.big = [b for b in tr.books if b in held], [b for b in tr.big if b in held]
    tr.books, tr.big = [b for b in tr.books if b not in held], [b for b in tr.big if b not in held]

    model = CFMMEM(K=args.K, S_w=args.S_w, S_r=args.S_r, kappa=args.kappa, sigma=args.sigma, cap_c=args.cap_c).to(dev)
    wids = {id(p) for p in model.writer.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in wids], "lr": args.lr},
        {"params": list(model.writer.parameters()), "lr": args.lr_w}], weight_decay=0.0)
    mode = "cue-only (no_memory)" if args.no_memory else f"memory, label={args.label}"
    print(f"{mode} | sigma {args.sigma} | device {dev} | train books {len(tr.books)}, eval books {len(ev.books)}")

    for step in range(1, args.steps + 1):
        T_cur = args.T if args.curr_steps <= 0 else min(args.T, args.T0 + int((args.T - args.T0) * step / args.curr_steps))
        tr.T, tr.n_home = T_cur, int(T_cur * tr.p_home)    # curriculum: short -> long streams
        bt = to_dev(tr.batch(args.B), dev)
        W, info = run_episode(model, bt, min(args.L, T_cur), args.label, grad=True, memory=not args.no_memory)
        L_read, L_or = probe_terms(model, W, bt, info["w"], args.p_teach)
        L_cap = (W.norm(dim=-1) - 1).clamp_min(0).pow(2).sum(-1).mean()
        lam_s = args.lam_scaf * max(0.0, 1.0 - step / args.scaf_steps)
        L_sc = scaffold_loss(model, W, bt, args.n_scaf) if lam_s > 0 else torch.zeros((), device=dev)
        loss = L_read + args.lam_or * L_or + args.lam_cap * L_cap + lam_s * L_sc
        if args.lam_str > 0 and info["last"] is not None:
            loss = loss + args.lam_str * reflow_loss(model, info["last"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_w = sum(p.grad.norm() ** 2 for p in model.writer.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d} T {T_cur:2d} | read {L_read.item():7.3f} oracle {L_or.item():7.3f} scaf {L_sc.item():7.3f} "
                  f"| disp {float(info['disp']):.4f} |grad phi| {float(g_w):.4f} |W| {W.norm(dim=-1).mean().item():.3f} "
                  f"| s_ema {model.s_ema.item():.2f}")
        if step % args.eval_every == 0 and step != args.steps:
            report(evaluate(model, ev, args, dev), args.P, f"eval @ {step}")
    report(evaluate(model, ev, args, dev), args.P, "final")


if __name__ == "__main__":
    main()
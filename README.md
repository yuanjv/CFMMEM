# CFMMEM v0.1

**Conditional Flow Matching Memory.** Two conditional flows trained jointly: a writer that transports the state, a reader that transports a cue to content. An autoencoder whose code is a persistent fixed-size state and whose reconstruction loss is importance-weighted, delayed, and **causally audited**.

Self-contained; supersedes HopMem v4.0.0. Three parts: **I. Ideas and goals · II. Mathematics · III. Implementation** (with PyTorch pseudocode). All empirical claims tagged **TO MEASURE**.

---

## Findings incorporated

Carried from v4 (F1–F10) and extended by the evaluation-protocol analysis (F11–F14):

| # | Finding | Resolution |
|---|---|---|
| F1 | Writer has no velocity-matching target → it is a **conditional neural ODE trained by task loss**, not vanilla CFM. CFM-proper only via reflow (§13) or inner-loop teacher (§26). | Naming corrected; project name reflects the reader's CFM core. |
| F2 | Reflow without stop-grad is minimized by $W_1=W_0,\ v\equiv0$ (no-op writer). | Endpoints detached, mandatory. |
| F3 | Detaching reader state-conditioning severs the only gradient path to the writer. | `detach_cond=False`, mandatory. |
| F4 | Scalar weight $w_i$ scales absolute pressure but not the coarse:fine ratio. | $t\sim\mathrm{Beta}(1+\kappa w_i,1)$ + uniform floor (§10). |
| F5 | Recomputing surprise at probe time creates a store→forget→surprise loop. | Measured once pre-write, detached, frozen. |
| F6 | Fixed shape ≠ fixed scale; residual integration can drift in norm. | Soft norm cap + monitors (§14). |
| F7 | Detail is bounded by encoder invertibility, not just memory. | Oracle-floor normalization (§16); token-level extension (§26). |
| F8 | Per-token-independent writers cannot compete, deduplicate, or reallocate. | Self-attn over $W$ + cross-attn to $Z$ (§17). |
| F9 | 20 writes × 3 steps = 60 sequential blocks; credit assignment needs a window. | No-grad prefix + grad window, multi-delay probes (§21). |
| F10 | Importance tilts recall under ambiguity toward important items. | Feature; tested separately on a margin ladder (§16, §24). |
| **F11** | **Eval-time Beta tilt is a measurement confound**: probing high-$w$ items mostly at high $t$ makes buckets incomparable. | Evaluation uses a **fixed, weight-independent $t$-grid**; tilt lives only in training (§16). |
| **F12** | $w=b\cdot\rho$ mixes surprise and recency; separation by $w$ cannot attribute cause. | Log-factorization $\log w=\log b+\log\rho$; matched stratification by factor (§16). |
| **F13** | Cue ambiguity and reader/encoder floor contaminate error; "detail separation" can be decoder difficulty. | Cue-margin matching + oracle-floor excess error; ambiguity tested separately (§16). |
| **F14** | Stratification is correlational: intrinsic item difficulty can correlate with $b$ and fake the effect. | Interventional training controls: uniform-weight and label-scramble (§16, §24). |

---

# PART I — IDEAS AND GOALS

## 1. The picture

One fixed-size latent state $W$. Two learned transports:

- **Writer**: given the state and new information, flow the state to a new state.
- **Reader**: given the state and a related cue, flow the cue to the stored content.

Remembering is writing what you recalled. Nothing is indexed, evicted, or merged by a rule.

## 2. The autoencoder analogy, precisely

| AE | CFMMEM |
|---|---|
| encoder | writer (state transition) |
| code | $W$, **persistent across the stream** |
| decoder | reader (cue-conditioned flow) |
| reconstruction | delayed, partial-cue retrieval |
| loss weight | importance = surprise × recency of success |
| fidelity | allocated along the flow's time axis |
| **audit** | **factorized labels, matched strata, interventional controls** |

The code is shared by all items and updated sequentially, so the writer must compress. The weighting decides *what deserves fidelity*. That is the entire retention policy — and v0.1 adds the requirement that this policy be **demonstrably causal**, not merely correlated with item difficulty or stream position.

## 3. Commitments

1. **Fixed size.** $W\in\mathbb R^{K\times d}$, always.
2. **Two flows, one primitive.** `reinforce(u) ≡ write(u)`.
3. **No hand-written memory policy.** No decay, dose, gate, merge rule, or ceiling.
4. **Importance lives in the loss**, never in the update.
5. **Detail is precision on the time axis**, demanded only where importance is high.
6. **Forgetting is caused by writing.** No clock in the core.
7. **Measurement is weight-agnostic.** Importance tilts *training*; evaluation probes on a common grid and stratifies post-hoc (F11).

## 4. Non-goals

Exact recall; guaranteed capacity; an auditable forgetting price; a descent theorem; collapse-freedom; orthogonal equivariance in $d$ (only permutation equivariance over state rows, §15); wall-clock recency.

## 5. What success looks like

Important items are recovered precisely from partial cues after long delays; unimportant items as gist. Repetition and recall do not erase neighbours. Overload degrades smoothly. Ablating the state collapses retrieval.

**The direct test of the goal is now a causal statement (TO MEASURE):**

> At matched delay, cue margin, and recency factor, fine-region excess error decreases with surprise-importance $b$ while coarse-region error stays flat — **and this separation vanishes under uniform-weight training and follows scrambled labels under label-scramble training.**

## 6. Naming honesty (F1)

The reader is CFM-proper: explicit velocity targets, conditional-mean minimizer. The writer is a conditional neural ODE trained end-to-end by the reader's loss; it acquires CFM structure only through the reflow objective (§13) or the inner-loop teacher (§26). The name **CFMMEM** claims the reader's mathematics and the writer's optional straightening — nothing more.

---

# PART II — MATHEMATICS

## 7. Objects

$$W\in\mathbb R^{K\times d}\ (K=16\text{–}32,\ d=384),\qquad z,q,u\in S^{d-1},\qquad \phi,\theta\ \text{shared, constant.}$$

Footprint $Kd+|\phi|+|\theta|$, independent of stream length. Rows are latent coordinates, **not** memories.

## 8. Writer: conditional transport on state space

$$\dot W_t=v_\phi(W_t,\,t,\,Z),\qquad W_0=W_{\text{prev}},\qquad W_{\text{new}}=W_1,$$

integrated with $S_w$ Euler steps ($2$–$4$). $Z$ is the information token set derived from $z$.

- Source is **the current state**, not noise: a transition, not generation.
- Optional source noise $W_0=W_{\text{prev}}+\sigma_w\Xi$ for exploration.
- $S_w=1$ with residual form is exactly a residual transformer. The flow framing buys adaptive depth, a straightening objective, and a place for stochasticity — not generative semantics.

## 9. Reader: CFM with explicit targets

Conditioning $c=(q,\,\mathrm{CrossAttn}(x_t,W))$. Source $x_0=q+\sigma\xi$, $\xi\sim\mathcal N(0,I)$.

$$x_t=(1-t)x_0+tz,\qquad \mathcal L_{\text{read}}=\mathbb E\big[w_i\|v_\theta(x_t,t,c)-(z_i-x_0)\|^2\big],$$

minimized by $\mathbb E[z_i-x_0\mid x_t,c]$ (Lipman et al. 2023; Tong et al. 2023). Inference: $S_r$ Euler steps, $u=\mathrm{normalize}(x_1)$.

**Absent trials:** $z=q$ gives target $-\sigma\xi$ — denoise back to the cue. Identity is learned behaviour, not a threshold.

**Training needs no ODE solve** — one velocity evaluation per probe. This makes joint training affordable.

## 10. Importance and the precision axis (F4)

Along the straight path, the conditional mean at $t\approx0$ averages over plausible targets (**coarse basin selection**); at $t\approx1$ the posterior concentrates on the specific item (**fine detail**). The time axis is a precision axis.

To make importance change the coarse:fine *ratio*, tilt the **training** sampling:

$$t\sim\mathrm{Beta}(1+\kappa w_i,\,1)\ \big(\text{density}\propto t^{\kappa w_i}\big),$$

with a floor mixture $t\sim U(0,1)$ (probability $0.2$) so basin selection stays trained for important items — neglecting low $t$ lands you on the wrong memory entirely. Absent trials use uniform $t$.

High $w$: mass near $t=1$ → exact resolution demanded. Low $w$: mass spread → coarse alignment demanded → gist. Because the writer is trained by this same loss, state capacity follows precision demand. **This tilt is the allocation mechanism and exists only in training; evaluation never uses it (F11, §16).**

## 11. Surprise, recency, and the label factorization (F5, F12)

$$s_i=\frac1{|\mathcal T|}\sum_{t\in\mathcal T}\big\|v_\theta(x_t,t,z_i,W_{\text{pre}})-(z_i-x_0)\big\|^2\quad\text{(detached, running-normalized, frozen)}$$

$\mathcal T$ = fixed 4-point quadrature, self-cue $z_i$, measured **once before writing**, frozen for the episode. Recency counted in write events (commitment 6). Both are metadata; neither enters the update.

Define the **factorized labels**:

$$b_i=\epsilon+s_i\quad\text{(base importance / surprise)},\qquad \rho_i=\epsilon+r_i,\quad r_i=\exp\!\big(-(n_{\text{probe}}-n_{\text{last-success}_i})/\tau_r\big),$$

$$w_i=b_i\,\rho_i\qquad\Longleftrightarrow\qquad \log w_i=\log b_i+\log\rho_i.$$

The log-factorization is the audit handle: any importance effect decomposes additively into a surprise channel and a recency channel, which §16 separates by matched stratification and §24 confirms by intervention. Note $b$ and $\rho$ are *not* independent in natural streams (early items meet an empty state → high surprise); matching is therefore mandatory, not optional.

## 12. Joint objective and gradient path (F3)

$$\mathcal L=\mathcal L_{\text{read}}+\lambda_{\text{cap}}\mathcal L_{\text{cap}}+\lambda_{\text{str}}\mathcal L_{\text{str}}$$

Gradient chain: $\mathcal L_{\text{read}}\to c\to W\to$ writer unroll $\to\phi$. **The reader's state conditioning must be differentiable.** Cross-attention gives per-token gradients; mean-pooling diffuses them — use cross-attention. No target $W$ exists or is needed: the writer amortizes "produce a state that makes future weighted retrieval succeed."

## 13. Optional reflow (F2)

From the same forward pass, with **detached** endpoints:

$$\bar W_t=(1-t)\,\mathrm{sg}[W_0]+t\,\mathrm{sg}[W_1],\qquad \mathcal L_{\text{str}}=\big\|v_\phi(\bar W_t,t,Z)-\big(\mathrm{sg}[W_1]-\mathrm{sg}[W_0]\big)\big\|^2.$$

Straightens the trajectory → 1-step inference. Without stop-grad this term rewards a no-op writer. Keep $\lambda_{\text{str}}$ small or alternate with task loss.

## 14. Invariants and scale (F6)

**Guaranteed:** shape $K\times d$ for all stream lengths; $O(1)$ per-event cost in stream length; permutation equivariance (§15); reader CFM minimizer is the conditional mean.

**Not guaranteed:** bounded scale — enforce softly,

$$\mathcal L_{\text{cap}}=\sum_k\max(0,\|W_k\|-1)^2;$$

descent; capacity; collapse-freedom; identifiability of importance weights.

## 15. Equivariance

For any permutation $P$ of the $K$ state rows, with **no positional encodings on state tokens**:

$$v_\phi(PW,t,Z)=P\,v_\phi(W,t,Z),\qquad \mathrm{Read}(PW,q)=\mathrm{Read}(W,q).$$

Row order carries no semantics. Info tokens *may* carry positional encodings (ordered text). Orthogonal equivariance in $d$ is **not** claimed.

## 16. Evaluation mathematics (F11–F14) — the causal audit

**Principle: measure once, label many ways.** Evaluation produces weight-agnostic error profiles; importance enters only as a post-hoc stratification label. The v4 "Condition A/B" dual-run collapses into a single measurement pass with three stratifications ($w$, $b$, $\rho$).

**(a) Common-grid error profile.** For item $i$ with cue $q_i$, target $z_i$, delay $\delta_i$ (writes since encoding), measured on a **fixed grid** $\mathcal T_{\text{eval}}$ (8 points, identical for every item, independent of any weight):

$$e_i(t)=\mathbb E_\xi\big\|v_\theta(x_t,t,c_i)-(z_i-x_0)\big\|^2,\qquad t\in\mathcal T_{\text{eval}}.$$

Eval reads are **side-effect-free**: no reinforce, no state update, labels frozen from the stream log. Measurement must not perturb the measured.

**(b) Oracle floor (F7, F13).** Condition the same reader on the target itself ($W\to z_i$ as a single token):

$$e_\star(t)=\mathbb E_\xi\big\|v_\theta(x_t,t,q_i,z_i)-(z_i-x_0)\big\|^2,\qquad \tilde e_i(t)=e_i(t)-e_\star(t).$$

$e_\star$ absorbs reader-approximation error; decoder inversion is scored separately. $\tilde e$ is the **memory-attributable excess error**.

**(c) Coarse/fine functionals.** Fixed windows $t_c=0.25,\ t_f=0.75$:

$$C_i=\underset{t\le t_c}{\mathrm{mean}}\,\tilde e_i(t),\qquad F_i=\underset{t\ge t_f}{\mathrm{mean}}\,\tilde e_i(t).$$

**(d) Matching.** Cue margin $m_i=\cos(q_i,z_i)-\max_{j\ne i}\cos(q_i,z_j)$. The core test uses only unambiguous cues ($m_i\ge m_{\min}$) and compares $b$-buckets **within strata of $(\delta_i, m_i, \rho_i)$**; the recency test compares $\rho$-buckets within strata of $(\delta_i, m_i, b_i)$. Ambiguity is studied separately, on its own ladder (below).

**(e) The contrast (difference-in-differences).** With $\Delta F_b=\mathbb E[F\mid b\text{ low}]-\mathbb E[F\mid b\text{ high}]$ and $\Delta C_b$ likewise:

$$S_b=\Delta F_b-\Delta C_b.$$

The goal predicts $S_b>0$ **and** $\Delta C_b\approx0$: importance buys *fine precision*, not uniformly better storage (gist survives for low-$b$ items). Define $S_\rho, S_w$ analogously. Parametric complement: regress $F_i$ on $(\log b_i,\log\rho_i,\delta_i,m_i)$ with episode effects; the hypothesis is a negative partial slope on $\log b$ that is absent in the $C$ regression.

**(f) Interventional controls (F14).** Train variants, identical except the label:

| Variant | Training weight | Prediction (TO MEASURE) |
|---|---|---|
| $M_{\text{full}}$ | $w=b\rho$ | $S_b>0$, $\Delta C_b\approx0$ |
| $M_{\text{unif}}$ | $w\equiv1$ | $S_b\approx0$ — separation must **vanish** |
| $M_{\text{scram}}$ | $w=b_{\pi(i)}\rho_{\pi(i)}$, random per-episode permutation | separation follows **assigned** labels, not intrinsic $b$ |
| $M_b$ / $M_\rho$ (optional) | $b$ only / $\rho$ only | factorial completeness: $S_b>0,S_\rho\approx0$ / vice versa |

If $M_{\text{unif}}$ still shows separation, the effect is an artifact of item difficulty leaking into $b$ — tighten matching and the oracle floor before claiming anything.

**(g) Ambiguity ladder (F10, separate test).** $q(\alpha)=\mathrm{normalize}(\alpha z_i+(1-\alpha)z_j+\eta)$, $\alpha\in\{1.0,0.9,0.7,0.5\}$. Prediction: the decision boundary $\alpha^\star$ where recall flips from $z_j$ to $z_i$ shifts with $w_i-w_j$. This is the *feature* channel; it must not be mixed into the detail-separation test.

---

# PART III — IMPLEMENTATION

## 17. Why the writer must attend (F8)

1. **Token–token interaction is the mechanism of memory management.** Independent per-row updates make $K$ copies of one function — no competition, deduplication, or reallocation. Self-attention over $W$ is what lets the state negotiate *who stores what*.
2. **Cross-attention to $Z$ is learned addressing**, replacing any hand-written content gate.
3. **Flatten-and-MLP is not viable**: $Kd\approx12$k inputs/outputs, no sharing, no permutation structure.

So: **self-attn over state rows + cross-attn to info tokens + AdaLN on $t$**, no positional encodings on the state. Cost $O(K^2d+Kmd)$ — trivial at $K=32$.

## 18. Components

| Component | Shape / role | Status |
|---|---|---|
| Encoder | text → unit $z$, $d=384$ | frozen (MiniLM) |
| $W$ | $K\times d$ latent state | opaque, fixed size |
| $v_\phi$ | 2-layer transformer: self-attn($W$) + cross-attn($Z$) + AdaLN($t$) → velocity | learned |
| $v_\theta$ | MLP + cross-attn($x_t\!\to\!W$), AdaLN($t$) → velocity | learned |
| Decoder | $u$ → text (embedding inversion) | frozen, scored separately |

$K=32$, $S_w=3$, $S_r=6$, hidden 512, 4 heads. $Z$: pooled $z$ as 1 token (core); token embeddings as $m$ tokens (extension, F7).

## 19. Parameters and defaults

Learned: $\phi,\theta$; optionally $\sigma$ (softplus, init 0.1), $\sigma_w$ (init 0).
Fixed: $K,S_w,S_r,\kappa,\epsilon,\tau_r,\lambda_{\text{cap}},\lambda_{\text{str}}$, uniform-$t$ mixture 0.2, $p_{\text{absent}}=0.2$, eval grid $|\mathcal T_{\text{eval}}|=8$, $t_c=0.25$, $t_f=0.75$, $m_{\min}$.
**Mandatory defaults:** `detach_cond=False` (F3), `reflow_stopgrad=True` (F2), `surprise_frozen=True` (F5), `eval_grid_fixed=True` (F11), `eval_reads_side_effect_free=True`.
**Removed from v3:** $\omega$, dose, dilution, $\rho$-ceiling, $a_r$, $\Gamma$, information clock inside the update.

## 20. API

```
W = init_state()                  # K x d, learned init tokens
W = writer(W, z)                  # S_w-step transport
u = reader(W, q)                  # S_r-step transport
W = writer(W, u)                  # reinforce == write
```

No `diffuse`, no gate, no surprise argument in the update. Surprise/recency are loss-side only.

## 21. Training recipe

- **No-grad prefix + grad window** (F9): first $T-L$ writes under `no_grad`, then a window of $L\approx6$ writes with gradients. Realistic prefix, bounded cost.
- **Multi-delay probes**: 4 probes per episode at short/medium/long delays (including prefix items); 20% absent.
- **Surprise**: 4-point quadrature, pre-write, detached, running-normalized, frozen (F5).
- **Curriculum**: short→long streams; clean→partial cues; few→same-passage competitors.
- Init $v_\phi$ output near zero → identity start, learns to edit.
- Optional scaffolding: current-item reconstruction, annealed to zero. Final test is delayed weighted retrieval alone.
- **Audit variants** $M_{\text{unif}}, M_{\text{scram}}$ (§16f) are trained with the identical recipe, changing only the label source.

## 22. Numerics and monitors

LayerNorm inside the nets (state itself unnormalized); soft norm cap; clip $\|x\|$ before normalize; stable softplus.

Monitor: $\|W\|_F$ and per-token norms; writer displacement $\|W_1-W_0\|$ (→0 = no-op collapse); **gradient norm reaching $\phi$** (→0 = F3 regressed); state-token diversity; surprise distribution; fine-vs-coarse excess error by bucket; empirical $\mathrm{corr}(\log b,\log\rho)$ per suite (must be ≈0 in ST1).

## 23. Evaluation protocol and synthetic stress suites

**Protocol (one pass):** run episodes; log frozen labels $(b_i,\rho_i,\delta_i,m_i)$; measure common-grid profiles $e_i(t)$ and oracle floor $e_\star(t)$ side-effect-free; compute $\tilde e, C_i, F_i$; stratify three ways ($w,b,\rho$) within matched strata; report $S_b,S_\rho,S_w$ and $\Delta C_b$; compare against $M_{\text{unif}},M_{\text{scram}}$.

**Synthetic suites (the "both" — theory needs controlled data):**

| Suite | Construction | Isolates |
|---|---|---|
| **ST1: factorial novelty × success** | 2×2 episodes: content ∈ {near-duplicate of stored, orthogonal novel} × success schedule ∈ {reinforced recently, reinforced long ago}. Verify $\mathrm{corr}(b,\rho)\approx0$. | F12: surprise vs recency, orthogonally by construction |
| **ST2: margin ladder** | $q(\alpha)$ mixtures of two stored items, $\alpha$ sweep | F10/F13: ambiguity tilt, separated from detail |
| **ST3: collapse probes** | displacement, grad-at-$\phi$, state ablation, ignores-$z$ | F2/F3/F8 failure modes |
| **ST4: overload** | stream length $\gg K$, smooth-degradation curve | capacity behaviour |
| **ST5: interference** | repeat/reinforce item $i$; measure $\Delta F_j,\Delta C_j$ for $j\ne i$ | "forgetting is caused by writing" — neighbours intact |

## 24. Baselines, predictions, interpretation

**Baselines:** cue-only; $W$ zeroed / shuffled / cross-episode; fixed-rule v3 writer with same reader; $K=1$; **plus** $M_{\text{unif}}$, $M_{\text{scram}}$, oracle floor.

**Falsifiable predictions (TO MEASURE):**
1. $S_b>0$ with $\Delta C_b\approx0$ at matched $(\delta,m,\rho)$ — the direct goal test.
2. $S_b(M_{\text{unif}})\approx0$; scramble separation follows assigned labels — causal guard.
3. Delayed retrieval beats fixed-rule writer at matched state size.
4. ST5: unrelated profiles intact after repetition/recall.
5. Overload degrades smoothly, no cliff.
6. State ablation collapses retrieval.
7. Permuting state rows changes nothing.
8. $K=1$ collapses to latest-item behaviour.
9. ST2: ambiguity boundary shifts with $w_i-w_j$.

**Interpretation matrix (corrected logic):**

| $S_b$(full) | $S_\rho$(full) | $S_b$(unif) | Verdict |
|---|---|---|---|
| $>0$ | $\approx0$ | $\approx0$ | Surprise-driven allocation **confirmed** |
| $>0$ | $>0$ | $\approx0$ | Both channels allocate (expected: $w=b\rho$); report partial effects |
| $\approx0$ | $>0$ | $\approx0$ | Recency-only; surprise channel broken — check freeze/normalization |
| $\approx0$ | $\approx0$ | $\approx0$, errors high | Transport/capacity failure, not importance — check no-op, grad path |
| $>0$ | any | $>0$ | **Measurement confound**: difficulty leaks into $b$ — tighten matching/floor |

## 25. Self-checks (20)

**Structure (1–5).** 1: shape $K\times d$ always. 2: permutation equivariance of writer, invariance of reader. 3: gradient norm at $\phi$ nonzero. 4: reflow endpoints detached (assert no grad path). 5: $S_w=1$ equals residual form.
**Writer (6–9).** 6: displacement nonzero after training. 7: per-token norms bounded by cap. 8: ignores-$z$ ablation degrades delayed retrieval. 9: prefix/window split matches full unroll forward.
**Reader (10–13).** 10: absent trials return $\approx q$. 11: CFM loss decreases. 12: $\sigma=0$ deterministic; $\sigma>0$ varied but on-topic. 13: output unit-norm, finite.
**Goal & audit (14–20).** 14: surprise frozen across probes. 15: training Beta-tilt mean increases with $w$. 16: **eval grid identical across buckets** (F11). 17: **eval reads leave $W$ bit-identical**. 18: matched strata balanced on $(\delta,m,\rho)$ across $b$-buckets. 19: $S_b(M_{\text{unif}})\approx0$. 20: scramble separation follows assigned labels.

## 26. Extensions and computational trade-offs (the "both", part 2)

**Inner-loop teacher (makes the writer CFM-proper).** Compute a target endpoint by a few first-order steps on the retention objective over the current probe set:

$$W^{(p+1)}=W^{(p)}-\eta\,\nabla_W\!\sum_j w_j\big\|v_\theta(x_t,t,c_j)-(z_j-x_0)\big\|^2,\qquad W^\star=W^{(P)},$$

then velocity-match $v_\phi$ to $\mathrm{sg}[W^\star-W_0]$ on the straight path. $\theta$ frozen inside the loop; no second-order gradients. The writer distills an optimizer.

| Extension | Buys | Extra train cost / event | Inference | Risk |
|---|---|---|---|---|
| Core | — | $S_w$ writer fwd + 1 reader eval/probe | $S_w+S_r$ steps | unroll stability |
| Reflow | straight trajectory | +1 writer eval | enables $S_w=1$ | no-op if sg missing (F2) |
| **Teacher** | writer CFM-proper | +$P$ reader fwd/bwd w.r.t. $W$ ($P=3$–$5$) ≈ **2–4× step cost** | unchanged | teacher objective mismatch |
| Token-level $Z$/reader | real detail fidelity (F7) | ×$m$ cross-attn; sequence reader ≈ ×$m$ | heavier | lifts encoder ceiling |
| Capacity sweep $K$ | capacity curve | $O(K^2d)$ | $O(K^2d)$ | — |
| Stochastic writer $\sigma_w>0$ | exploration | +noise | sampling variance | drift |
| 1-step distillation | fast inference | reflow training | $S_w=1$ | fidelity loss |

All variants remain $O(1)$ per event in stream length.

## 27. PyTorch pseudocode

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Beta

# ---------- primitives ----------
class AdaLN(nn.Module):                      # time-conditioned LayerNorm
    def __init__(self, d, d_t):
        super().__init__(); self.ln = nn.LayerNorm(d); self.proj = nn.Linear(d_t, 2*d)
    def forward(self, x, t_emb):
        g, b = self.proj(t_emb).chunk(2, -1)
        return self.ln(x) * (1 + g) + b

class WriterBlock(nn.Module):                # F8: self-attn(W) + cross-attn(Z), no PE on W
    def __init__(self, d, d_t, heads=4):
        super().__init__()
        self.a1, self.a2, self.a3 = AdaLN(d, d_t), AdaLN(d, d_t), AdaLN(d, d_t)
        self.self_attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, W, Z, t_emb):
        h = self.a1(W, t_emb); W = W + self.self_attn(h, h, h)[0]
        h = self.a2(W, t_emb); W = W + self.cross_attn(h, Z, Z)[0]
        return W + self.mlp(self.a3(W, t_emb))

class WriterVelocity(nn.Module):             # v_phi: conditional neural ODE field (F1)
    def __init__(self, d=384, d_t=128, layers=2):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, d_t), nn.SiLU(), nn.Linear(d_t, d_t))
        self.blocks = nn.ModuleList([WriterBlock(d, d_t) for _ in range(layers)])
        self.out = nn.Linear(d, d); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, W, t, Z):              # W:(B,K,d)  Z:(B,m,d)  t:(B,1)
        e = self.time(t)
        for blk in self.blocks: W = blk(W, Z, e)
        return self.out(W)                   # near-zero init -> identity start

class ReaderVelocity(nn.Module):             # v_theta: CFM field; cond MUST be differentiable in W (F3)
    def __init__(self, d=384, d_t=128, heads=4):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, d_t), nn.SiLU(), nn.Linear(d_t, d_t))
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)   # x_t -> W
        self.adaln = AdaLN(3*d, d_t)
        self.mlp = nn.Sequential(nn.Linear(3*d, 512), nn.SiLU(), nn.Linear(512, d))
    def forward(self, x, t, q, W):           # x,q:(B,d)  W:(B,K,d)
        ctx = self.cross(x.unsqueeze(1), W, W)[0].squeeze(1)              # detach_cond=False
        h = torch.cat([x, q, ctx], -1)
        return self.mlp(self.adaln(h, self.time(t)))

# ---------- model ----------
class CFMMEM(nn.Module):
    def __init__(self, K=32, d=384, S_w=3, S_r=6, kappa=2.0, eps=0.1,
                 sigma=0.1, sigma_w=0.0, p_uniform=0.2, p_absent=0.2):
        super().__init__()
        self.W_init = nn.Parameter(0.1*torch.randn(K, d))
        self.writer, self.reader = WriterVelocity(d), ReaderVelocity(d)
        self.K, self.d, self.S_w, self.S_r = K, d, S_w, S_r
        self.kappa, self.eps, self.sigma, self.sigma_w = kappa, eps, sigma, sigma_w
        self.p_uniform, self.p_absent = p_uniform, p_absent

    def write(self, W, Z):                   # S_w Euler steps; source = state, not noise
        if self.sigma_w > 0: W = W + self.sigma_w*torch.randn_like(W)
        for s in range(self.S_w):
            t = torch.full((W.shape[0], 1), s/self.S_w, device=W.device)
            W = W + (1.0/self.S_w)*self.writer(W, t, Z)
        return W

    @torch.no_grad()
    def read(self, W, q):                    # inference only; eval reads never write
        x = q + self.sigma*torch.randn_like(q)
        for s in range(self.S_r):
            t = torch.full((q.shape[0], 1), s/self.S_r, device=q.device)
            x = x + (1.0/self.S_r)*self.reader(x, t, q, W)
        return F.normalize(x.clamp(-1e4, 1e4), dim=-1)

    def sample_t(self, w, absent):           # F4: Beta tilt + uniform floor; TRAIN ONLY
        b = Beta(1 + self.kappa*w, torch.ones_like(w)).sample()
        u = torch.rand_like(w)
        mix = (torch.rand_like(w) < self.p_uniform) | absent
        return torch.where(mix, u, b)

    def probe_loss(self, W, q, z, w, absent):
        x0 = q + self.sigma*torch.randn_like(q)
        t  = self.sample_t(w, absent).unsqueeze(-1)
        xt = (1-t)*x0 + t*z
        pred = self.reader(xt, t, q, W)      # gradient: loss -> W -> writer unroll -> phi
        return (w*((pred - (z - x0))**2).sum(-1)).mean()

    @torch.no_grad()
    def surprise(self, W_pre, z):            # F5: once, pre-write, detached, frozen
        x0 = z + self.sigma*torch.randn_like(z)
        errs = []
        for t in (0.125, 0.375, 0.625, 0.875):                # 4-point quadrature
            tt = torch.full((z.shape[0], 1), t, device=z.device)
            xt = (1-tt)*x0 + tt*z
            errs.append(((self.reader(xt, tt, z, W_pre) - (z - x0))**2).sum(-1))
        return running_normalize(torch.stack(errs).mean(0))   # detached label

# ---------- training step (F9: no-grad prefix + grad window) ----------
def train_episode(model, stream, labels, L=6, lam_cap=1e-3, lam_str=1e-2):
    W = model.W_init.unsqueeze(0).expand(B, -1, -1)
    with torch.no_grad():                                     # realistic prefix, bounded cost
        for item in stream[:-L]:
            labels[item.id] = (model.surprise(W, item.z), recency(item))
            W = model.write(W, item.Z)
    for item in stream[-L:]:                                  # grad window
        b = model.surprise(W.detach(), item.z)                # pre-write, frozen
        labels[item.id] = (b, recency(item))
        W0 = W; W = model.write(W, item.Z)
        L_read = sum(model.probe_loss(W, q_j, z_j, w_j, abs_j)  # multi-delay probe set
                     for (q_j, z_j, w_j, abs_j) in probe_set(labels, p_absent=0.2))
        t = torch.rand(B, 1)                                  # F2: reflow, endpoints detached
        Wbar = (1-t)*W0.detach() + t*W.detach()
        L_str = ((model.writer(Wbar, t, item.Z) - (W.detach()-W0.detach()))**2).mean()
        L_cap = (W.norm(dim=-1) - 1).clamp_min(0).pow(2).sum()  # F6
        (L_read + lam_str*L_str + lam_cap*L_cap).backward()
        monitor(grad_norm_at=model.writer, displacement=(W-W0).norm(), W_norm=W.norm())

# ---------- evaluation (F11-F14: measure once, label many ways) ----------
@torch.no_grad()
def error_profile(model, W, q, z, grid=(.06,.19,.31,.44,.56,.69,.81,.94), n_xi=8):
    e = torch.zeros(len(grid))
    for _ in range(n_xi):
        x0 = q + model.sigma*torch.randn_like(q)
        for k, t in enumerate(grid):                          # COMMON grid, weight-independent
            tt = torch.full((q.shape[0],1), t)
            xt = (1-tt)*x0 + tt*z
            e[k] += ((model.reader(xt, tt, q, W) - (z-x0))**2).sum(-1).mean()
    return e / n_xi                                           # W untouched: side-effect-free

def audit(model, episodes, variants):                         # variants: full/unif/scram
    profiles, floor = {}, error_profile_oracle(model)         # oracle: W -> z as 1 token
    for i in episodes:
        tilde = profiles[i] - floor
        C_i, F_i = tilde[grid <= .25].mean(), tilde[grid >= .75].mean()
    # stratify by (delay, margin, rho) -> S_b ; by (delay, margin, b) -> S_rho
    S_b = did(F, C, label='b', match=('delay','margin','rho'))
    assert abs(S_b_unif) < tol                                # self-check 19
```

## 28. Status and risks

The writer's unrolled transport is the cost centre and stability risk. Degenerate solutions — no-op writer, state-ignoring reader, $K$-token collapse — are the real failure modes; §22–25 are their detectors. The **audit risk** is subtler: a confounded evaluation can *validate a broken mechanism* (F14) or *condemn a working one* (F12/F13). The common grid, oracle floor, matched strata, and interventional controls exist precisely to make that impossible. No guarantees are claimed. Everything empirical is **TO MEASURE**.

---

**Summary of the rename and the fix.** CFMMEM v0.1 keeps HopMem v4's architecture and all ten corrected findings, and adds the causal audit as a first-class component: importance factorizes as $\log w=\log b+\log\rho$; evaluation measures weight-agnostic excess-error profiles on a common grid, side-effect-free; the goal is tested by a difference-in-differences $S_b>0,\ \Delta C_b\approx0$ within matched strata, and confirmed causally by uniform-weight and label-scramble training controls. The mechanism tilts training; the measurement never tilts. That separation is what makes the claim falsifiable.

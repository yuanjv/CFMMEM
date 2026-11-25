"""cfmmem_data.py -- Phase 0 data layer for CFMMEM.

Gutenberg passages -> frozen all-MiniLM-L6-v2 embeddings (full / prefix / suffix) -> episodes.
Untested sketch: run `check` after `prep` and read the numbers before trusting anything.

  pip install datasets sentence-transformers torch
  python cfmmem_data.py prep  --books 300 --cache gut_minilm.pt
  python cfmmem_data.py check --cache gut_minilm.pt
"""
import argparse, json, random, re
import torch
import torch.nn.functional as F

DATASET = "sedthh/gutenberg_english"                      # columns: TEXT, SOURCE, METADATA (json string)
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"        # d = 384, 256 word-piece limit

# ---------------------------------------------------------------- text -> passages
_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\n", re.I)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


def strip_boilerplate(text):
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[:m.start()]
    return text


def passages(text, lo=70, hi=150, min_sents=4):
    """Pack consecutive sentences into passages of lo..hi words; yield lists of sentences."""
    flat = re.sub(r"\s+", " ", strip_boilerplate(text))
    cur, n = [], 0
    for s in _SENT.split(flat):
        if not s:
            continue
        cur.append(s)
        n += len(s.split())
        if n >= lo:
            body = " ".join(cur)
            alpha = sum(c.isalpha() for c in body) / max(1, len(body))
            if len(cur) >= min_sents and n <= hi and alpha > 0.75 and "[Illustration" not in body:
                yield cur
            cur, n = [], 0


# ---------------------------------------------------------------- prep: one-time embedding cache
def prep(n_books, cache, window=200, seed=0, device=None):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rng = random.Random(seed)
    ds = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=64)
    full, pre, suf, book, pos, titles = [], [], [], [], [], []
    for row in ds:
        ps = list(passages(row["TEXT"]))
        ps = ps[len(ps) // 20:]                            # skip front matter
        if len(ps) < window:
            continue
        s0 = rng.randrange(len(ps) - window + 1)
        ps = ps[s0:s0 + window]                            # contiguous run keeps local coherence
        try:
            title = json.loads(row["METADATA"]).get("title", "")
        except Exception:
            title = ""
        b = len(titles)
        titles.append(title)
        for i, sents in enumerate(ps):
            h = len(sents) // 2
            full.append(" ".join(sents))
            pre.append(" ".join(sents[:h]))                # cue = first half
            suf.append(" ".join(sents[h:]))                # cue = second half
            book.append(b)
            pos.append(i)
        if len(titles) >= n_books:
            break

    enc = SentenceTransformer(ENCODER, device=device)

    def emb(xs):
        return enc.encode(xs, batch_size=256, convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=True).cpu().half()

    torch.save(dict(z=emb(full), pre=emb(pre), suf=emb(suf), book=torch.tensor(book),
                    pos=torch.tensor(pos), titles=titles, text=full), cache)
    print(f"{len(titles)} books, {len(full)} passages -> {cache}")


# ---------------------------------------------------------------- episodes
class Episodes:
    """Stream of T passages + P probes (one per delay segment).

    Half the stream (p_home) comes from one 'home' book, so same-book passages compete as
    near-neighbours; the rest come from other books. Absent probes use a home-book passage that
    was never written (hard negative) with target z = q, as in spec §9.
    Labels b (surprise) and rho (recency) are NOT produced here: they come from the model
    (spec §11). `novel` is only a natural novelty handle: first occurrence of a book in the stream.
    """

    def __init__(self, cache, T=20, n_probe=4, p_absent=0.2, p_home=0.5, sigma_cue=0.0, seed=0):
        d = torch.load(cache)
        self.z = d["z"].float()
        self.cues = {"pre": d["pre"].float(), "suf": d["suf"].float()}
        self.book = d["book"].tolist()
        self.by_book = {}
        for i, b in enumerate(self.book):
            self.by_book.setdefault(b, []).append(i)
        self.books = list(self.by_book)
        self.T, self.P, self.p_absent, self.p_home, self.sigma_cue = T, n_probe, p_absent, p_home, sigma_cue
        self.n_home = int(T * p_home)
        self.big = [b for b in self.books if len(self.by_book[b]) >= self.n_home + n_probe]
        assert self.big and len(self.books) > 1, "need more/longer books in the cache"
        self.rng = random.Random(seed)
        self.g = torch.Generator().manual_seed(seed)

    def cue(self, i):
        v = self.cues[self.rng.choice(("pre", "suf"))][i]
        if self.sigma_cue > 0:
            v = v + self.sigma_cue * torch.randn(v.shape, generator=self.g)
        return F.normalize(v, dim=-1)

    def one(self):
        r, T, P = self.rng, self.T, self.P
        home = r.choice(self.big)
        pool = r.sample(self.by_book[home], self.n_home + P)
        held, home_items = pool[:P], pool[P:]
        other = []
        while len(other) < T - self.n_home:
            b = r.choice(self.books)
            i = r.choice(self.by_book[b])
            if b != home and i not in other:
                other.append(i)
        stream = home_items + other
        r.shuffle(stream)
        Zc = torch.stack([self.cue(i) for i in stream])    # (T, d): each item's own partial cue (write-time surprise)
        Z = self.z[stream]                                 # (T, d): info tokens (core: 1 token each)

        seen, novel = set(), []
        for i in stream:
            novel.append(self.book[i] not in seen)
            seen.add(self.book[i])

        q, zt, absent, t_idx, delay, margin = [], [], [], [], [], []
        for k in range(P):
            lo, hi = k * T // P, (k + 1) * T // P          # short / medium / long delays
            if r.random() < self.p_absent:
                qk = self.cue(held[k]); zk = qk; t = -1; dl = -1; m = float("nan")
            else:
                t = r.randrange(lo, hi)
                qk = self.cue(stream[t]); zk = Z[t]; dl = T - 1 - t     # writes since encoding
                s = Z @ qk
                s[t] = -2.0
                m = (Z[t] @ qk - s.max()).item()           # cue margin vs other stored items
            q.append(qk); zt.append(zk); absent.append(t < 0)
            t_idx.append(t); delay.append(dl); margin.append(m)
        return dict(Z=Z, Zc=Zc, novel=torch.tensor(novel), q=torch.stack(q), z=torch.stack(zt),
                    absent=torch.tensor(absent), t_idx=torch.tensor(t_idx),
                    delay=torch.tensor(delay), margin=torch.tensor(margin))

    def batch(self, B):
        eps = [self.one() for _ in range(B)]
        return {k: torch.stack([e[k] for e in eps]) for k in eps[0]}


# ---------------------------------------------------------------- sanity check
def check(cache, n=500):
    ep = Episodes(cache)
    b = ep.batch(n)
    keep = ~b["absent"]
    cos_qz = (b["q"] * b["z"]).sum(-1)[keep]
    m = b["margin"][keep]
    qs = torch.tensor([.1, .5, .9])
    r3 = lambda t: [round(x, 3) for x in t.tolist()]
    print("cue->target cos  p10/50/90:", r3(torch.quantile(cos_qz, qs)))
    print("cue margin       p10/50/90:", r3(torch.quantile(m, qs)), f" frac(m>0)={(m > 0).float().mean():.2f}")
    print(f"novel writes per episode: {b['novel'].float().sum(1).mean():.1f} of {ep.T}")
    rng, within, cross = random.Random(0), [], []
    for _ in range(2000):
        bk = rng.choice(ep.big)
        i, j = rng.sample(ep.by_book[bk], 2)
        o = rng.choice(ep.by_book[rng.choice(ep.books)])
        within.append((ep.z[i] @ ep.z[j]).item())
        cross.append((ep.z[i] @ ep.z[o]).item())
    print(f"mean cos same-book {sum(within) / len(within):.3f}  vs cross-book {sum(cross) / len(cross):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "check"])
    ap.add_argument("--cache", default="gut_minilm.pt")
    ap.add_argument("--books", type=int, default=300)
    ap.add_argument("--window", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.books, a.cache, a.window)
    else:
        check(a.cache)
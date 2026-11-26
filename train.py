# -*- coding: utf-8 -*-
"""Мини-GPT с нуля на numpy: токенизатор, трансформер, ручной backprop, Adam.
Экспорт весов в model.json для инференса на JavaScript."""
import re, json, math, time, os
import numpy as np

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- tokenizer
TOK_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*|[а-яёa-z0-9]|[.,!?;:]")

def tokenize(text):
    return TOK_RE.findall(text.lower().strip())

PAD, BOS, EOS, NL, UNK = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "<nl>", "<unk>"]

print("чтение корпуса...")
stream = [BOS]
target_mask = [0]  # параллельно: 1 если на этой позиции модель должна предсказывать токен
lines = open("data/corpus.txt", encoding="utf-8").read().strip().split("\n")
word2id = {}

def wid(w):
    if w not in word2id:
        word2id[w] = len(word2id) + len(SPECIALS)
    return word2id[w]

for i in range(0, len(lines) - 1, 2):
    qline, aline = lines[i], lines[i + 1]
    assert qline.startswith("Вопрос:") and aline.startswith("Ответ:"), (qline, aline)
    toks = [BOS, wid("вопрос"), wid(":")] + [wid(t) for t in tokenize(qline[7:])]
    toks += [NL, wid("ответ"), wid(":")]
    tm = [0] * len(toks)
    atoks = [wid(t) for t in tokenize(aline[6:])] + [EOS]
    toks += atoks
    tm = [0] * (len(toks) - len(atoks)) + [1] * len(atoks)
    tm = tm[1:] + [0]  # сдвиг влево: позиция p обучается, если p+1 это ответный токен
    stream += toks
    target_mask += tm

vocab_size = len(word2id) + len(SPECIALS)
itos = {v: k for k, v in word2id.items()}
itos.update({i: s for i, s in enumerate(SPECIALS)})
print("токенов в потоке:", len(stream), "| словарь:", vocab_size)

# ------------------------------------------------------- примеры обучения
MAXLEN = 44
examples = []  # (ids, mask) выровненные по длине
i = 0
n = len(stream)
while i < n:
    j = i
    # диалог заканчивается на EOS
    while j < n and stream[j] != EOS:
        j += 1
    seg = stream[i:j + 1]; sm = target_mask[i:j + 1]
    i = j + 1
    if len(seg) < 6 or len(seg) > MAXLEN:
        if len(seg) > MAXLEN:
            seg = seg[:MAXLEN]; sm = sm[:MAXLEN]
        else:
            continue
    pad = MAXLEN - len(seg)
    examples.append((seg + [PAD] * pad, sm + [0] * pad))
print("примеров обучения:", len(examples))

X_all = np.array([e[0] for e in examples], dtype=np.int32)
M_all = np.array([e[1] for e in examples], dtype=np.float32)

# ------------------------------------------------------------- гиперпараметры
D = 96          # размер модели
HEADS = 3
DH = D // HEADS
FF = 384
N_LAYER = 2
MAXPOS = MAXLEN
LR = 2.5e-3
BATCH = 64
STEPS = 8000
WARMUP = 150

# ------------------------------------------------------------- инициализация
P = {}
P["tok_emb"] = rng.normal(0, 0.02, (vocab_size, D))
P["pos_emb"] = rng.normal(0, 0.02, (MAXPOS, D))
for l in range(N_LAYER):
    P[f"Wqkv{l}"] = rng.normal(0, 0.02, (D, 3 * D))
    P[f"bqkv{l}"] = np.zeros(3 * D)
    P[f"Wo{l}"] = rng.normal(0, 0.02 / math.sqrt(2 * N_LAYER), (D, D))
    P[f"bo{l}"] = np.zeros(D)
    P[f"W1_{l}"] = rng.normal(0, 0.02, (D, FF))
    P[f"b1_{l}"] = np.zeros(FF)
    P[f"W2_{l}"] = rng.normal(0, 0.02 / math.sqrt(2 * N_LAYER), (FF, D))
    P[f"b2_{l}"] = np.zeros(D)
    P[f"ln1g{l}"] = np.ones(D); P[f"ln1b{l}"] = np.zeros(D)
    P[f"ln2g{l}"] = np.ones(D); P[f"ln2b{l}"] = np.zeros(D)
P["lnfg"] = np.ones(D); P["lnfb"] = np.zeros(D)

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(v + eps)
    return xhat * g + b, xhat, np.sqrt(v + eps)

def ln_backward(dy, xhat, sigma, g):
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxh = dy * g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xhat * (dxh * xhat).mean(-1, keepdims=True)) / sigma
    return dx, dg, db

def forward(ids, mask):
    """ids (B,T) int, mask (B,T); возвращает loss и кэш для backward."""
    B, T = ids.shape
    cache = {"ids": ids, "mask": mask, "B": B, "T": T}
    x = P["tok_emb"][ids] + P["pos_emb"][:T]  # (B,T,D)
    causal = np.tril(np.ones((T, T), dtype=bool))
    lcache = []
    for l in range(N_LAYER):
        xn, xhat1, sig1 = layernorm(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
        qkv = xn @ P[f"Wqkv{l}"] + P[f"bqkv{l}"]          # (B,T,3D)
        qkv = qkv.reshape(B, T, 3, HEADS, DH).transpose(2, 0, 3, 1, 4)  # (3,B,H,T,DH)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pre = q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH)  # (B,H,T,T)
        pre = np.where(causal, pre, -1e9)
        pre_max = pre.max(-1, keepdims=True)
        e = np.exp(pre - pre_max); att = e / e.sum(-1, keepdims=True)
        z = att @ v                                         # (B,H,T,DH)
        zo = z.transpose(0, 2, 1, 3).reshape(B, T, D)       # (B,T,D)
        out = zo @ P[f"Wo{l}"] + P[f"bo{l}"]
        x2 = x + out
        hn, xhat2, sig2 = layernorm(x2, P[f"ln2g{l}"], P[f"ln2b{l}"])
        f1 = hn @ P[f"W1_{l}"] + P[f"b1_{l}"]
        act = np.maximum(f1, 0)
        f2 = act @ P[f"W2_{l}"] + P[f"b2_{l}"]
        x3 = x2 + f2
        lcache.append(dict(x=x, xhat1=xhat1, sig1=sig1, xn=xn, qkv3=qkv, q=q, k=k, v=v,
                           att=att, zo=zo, x2=x2, xhat2=xhat2, sig2=sig2, hn=hn,
                           f1=f1, act=act))
        x = x3
    xf, xhatf, sigf = layernorm(x, P["lnfg"], P["lnfb"])
    logits = xf @ P["tok_emb"].T                            # (B,T,V) tied
    # softmax + CE
    lmax = logits.max(-1, keepdims=True)
    lse = lmax + np.log(np.exp(logits - lmax).sum(-1, keepdims=True))
    probs = np.exp(logits - lse)
    tgt = np.zeros_like(ids, dtype=np.int32)
    tgt[:, :-1] = ids[:, 1:]                                # предсказать следующий токен
    mt = mask[:, :-1]                                       # маска на позициях предсказаний
    if mt.sum() == 0:
        mt = mt + 1
    logp_tgt = np.take_along_axis(logits, tgt[..., None], -1)[..., 0] - lse[..., 0]
    loss = -(logp_tgt[:, :-1] * mt).sum() / mt.sum()
    cache.update(lcache=lcache, xf=xf, xhatf=xhatf, sigf=sigf, probs=probs, lmax=lmax,
                 lse=lse, tgt=tgt, mt=mt, causal=causal, x_final=x)
    return loss, cache

def backward(cache):
    B, T = cache["B"], cache["T"]
    probs, tgt, mt, lse, lmax = cache["probs"], cache["tgt"], cache["mt"], cache["lse"], cache["lmax"]
    dlogits = probs.copy()
    oh = np.zeros_like(dlogits)
    np.put_along_axis(oh, tgt[..., None], 1.0, -1)
    mt_full = np.concatenate([mt, np.zeros((B, 1), dtype=mt.dtype)], axis=1)
    dlogits = (dlogits - oh) * mt_full[..., None]
    dlogits /= mt.sum()
    grads = {}
    grads["tok_emb"] = dlogits.reshape(-1, vocab_size).T @ cache["xf"].reshape(-1, D)  # tied head
    dxf = dlogits @ P["tok_emb"]                                                        # (B,T,D)
    dx, dg, db = ln_backward(dxf, cache["xhatf"], cache["sigf"], P["lnfg"])
    grads["lnfg"] = dg; grads["lnfb"] = db
    dx = dx + np.zeros_like(dx)  # от внимания выше residual-ветки нет
    for l in reversed(range(N_LAYER)):
        lc = cache["lcache"][l]
        # residual FF
        df2 = dx
        grads[f"W2_{l}"] = (lc["act"].reshape(-1, FF).T @ df2.reshape(-1, D))
        grads[f"b2_{l}"] = df2.sum((0, 1))
        dact = df2 @ P[f"W2_{l}"].T
        df1 = dact * (lc["f1"] > 0)
        grads[f"W1_{l}"] = lc["hn"].reshape(-1, D).T @ df1.reshape(-1, FF)
        grads[f"b1_{l}"] = df1.sum((0, 1))
        dhn = df1 @ P[f"W1_{l}"].T
        dx2_from_ln, dg2, db2 = ln_backward(dhn, lc["xhat2"], lc["sig2"], P[f"ln2g{l}"])
        grads[f"ln2g{l}"] = dg2; grads[f"ln2b{l}"] = db2
        dx2 = dx + dx2_from_ln
        # attention block: x2 = x + out(o)
        dout = dx2
        grads[f"Wo{l}"] = lc["zo"].reshape(-1, D).T @ dout.reshape(-1, D)
        grads[f"bo{l}"] = dout.sum((0, 1))
        dzo = dout @ P[f"Wo{l}"].T                          # (B,T,D)
        dz = dzo.reshape(B, T, HEADS, DH).transpose(0, 2, 1, 3)  # (B,H,T,DH)
        datt = dz @ lc["v"].transpose(0, 1, 3, 2)
        dv = lc["att"].transpose(0, 1, 3, 2) @ dz
        dpre = lc["att"] * (datt - (datt * lc["att"]).sum(-1, keepdims=True))
        dpre = np.where(cache["causal"], dpre, 0.0)
        dq = dpre @ lc["k"] / math.sqrt(DH)
        dk = dpre.transpose(0, 1, 3, 2) @ lc["q"] / math.sqrt(DH)
        dqkv = np.stack([dq, dk, dv], axis=0)               # (3,B,H,T,DH)
        dqkv = dqkv.transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * D)  # back to (B,T,3D)
        xn2 = lc["xn"].reshape(-1, D)
        grads[f"Wqkv{l}"] = xn2.T @ dqkv.reshape(-1, 3 * D)
        grads[f"bqkv{l}"] = dqkv.sum((0, 1))
        dxn = dqkv @ P[f"Wqkv{l}"].T
        dx_from_ln1, dg1, db1 = ln_backward(dxn, lc["xhat1"], lc["sig1"], P[f"ln1g{l}"])
        grads[f"ln1g{l}"] = dg1; grads[f"ln1b{l}"] = db1
        dx = dx2 + dx_from_ln1
    # токен-эмбеддинги из входа
    oh_in = np.zeros((B, T, vocab_size))
    np.put_along_axis(oh_in, cache["ids"][..., None], 1.0, -1)
    grads["tok_emb"] += np.einsum("btv,btd->vd", oh_in, dx)
    grads["pos_emb"] = dx.sum(0)
    return grads

# ------------------------------------------------------------- Adam
m = {k: np.zeros_like(v) for k, v in P.items()}
v2 = {k: np.zeros_like(v) for k, v in P.items()}
BETA1, BETA2, EPSA = 0.9, 0.98, 1e-8

def clip_grads(g, lim=1.0):
    total = 0.0
    for k in g:
        total += np.linalg.norm(g[k]) ** 2
    total = math.sqrt(total)
    if total > lim:
        for k in g:
            g[k] *= lim / total

def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * (0.5 * (1 + math.cos(math.pi * t)))

def encode_question(text):
    t = [BOS, word2id["вопрос"], word2id[":"]]
    for w in tokenize(text):
        t.append(word2id.get(w, UNK))
    t += [NL, word2id["ответ"], word2id[":"]]
    return t

def decode_tokens(toks):
    words = []
    for t in toks:
        if t in ("eos", "<eos>", "<nl>", "<pad>", "<unk>", "<bos>"):
            continue
        if t in (".", "!", "?", ";", ":", ","):
            if words:
                words[-1] += t
        else:
            words.append(t)
    s = " ".join(words)
    return s[0].upper() + s[1:] if s else s

def generate(text, temperature=0.8, top_k=5, max_new=24, seed=None):
    r = np.random.default_rng(seed)
    ids = encode_question(text)[-MAXPOS:]
    out = []
    for _ in range(max_new):
        ids_c = ids[-MAXPOS:]
        T = len(ids_c)
        x = P["tok_emb"][np.asarray(ids_c)] + P["pos_emb"][:T]
        causal = np.tril(np.ones((T, T), dtype=bool))
        for l in range(N_LAYER):
            xn, _, _ = layernorm(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
            qkv = (xn @ P[f"Wqkv{l}"] + P[f"bqkv{l}"]).reshape(T, 3, HEADS, DH).transpose(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            pre = np.where(causal, q @ k.transpose(0, 2, 1) / math.sqrt(DH), -1e9)
            e = np.exp(pre - pre.max(-1, keepdims=True)); att = e / e.sum(-1, keepdims=True)
            zo = (att @ v).transpose(1, 0, 2).reshape(T, D)
            x = x + zo @ P[f"Wo{l}"] + P[f"bo{l}"]
            hn, _, _ = layernorm(x, P[f"ln2g{l}"], P[f"ln2b{l}"])
            f1 = hn @ P[f"W1_{l}"] + P[f"b1_{l}"]
            x = x + np.maximum(f1, 0) @ P[f"W2_{l}"] + P[f"b2_{l}"]
        xf, _, _ = layernorm(x, P["lnfg"], P["lnfb"])
        logits = xf[-1] @ P["tok_emb"].T
        logits[PAD] = -1e9; logits[UNK] = -1e9; logits[BOS] = -1e9; logits[NL] = -1e9
        logits[EOS] -= 0.5
        logits /= max(temperature, 1e-3)
        top = np.argsort(logits)[-top_k:]
        pt = np.exp(logits[top] - logits[top].max())
        pt /= pt.sum()
        nxt = int(r.choice(top, p=pt))
        if nxt == EOS:
            break
        ids.append(nxt)
        out.append(itos[nxt])
        if len(ids) >= MAXPOS:
            break
    return decode_tokens(out)

# ------------------------------------------------------------- экспорт
def export_model():
    os.makedirs("site", exist_ok=True)
    weights = {k: np.round(v, 5).tolist() for k, v in P.items()}
    meta = {
        "vocab_size": int(vocab_size),
        "d_model": D, "heads": HEADS, "dh": DH, "ff": FF,
        "n_layer": N_LAYER, "max_len": int(MAXPOS),
        "specials": {"pad": PAD, "bos": BOS, "eos": EOS, "nl": NL, "unk": UNK},
        "vocab": {itos[i]: int(i) for i in range(vocab_size)},
        "params": int(sum(v.size for v in P.values())),
        "loss_curve": hist,
    }
    with open("site/model.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "weights": weights}, f, ensure_ascii=False)
    print("экспорт: весов", meta["params"], "| model.json",
          round(os.path.getsize("site/model.json") / 1e6, 1), "МБ")

# ------------------------------------------------------------- обучение
t0 = time.time()
hist = []
for step in range(STEPS):
    idx = rng.choice(len(X_all), BATCH, replace=False)
    loss, c = forward(X_all[idx], M_all[idx])
    grads = backward(c)
    clip_grads(grads)
    lr = lr_at(step)
    for k in P:
        m[k] = BETA1 * m[k] + (1 - BETA1) * grads[k]
        v2[k] = BETA2 * v2[k] + (1 - BETA2) * (grads[k] ** 2)
        P[k] -= lr * (m[k] / (1 - BETA1 ** (step + 1))) / (np.sqrt(v2[k] / (1 - BETA2 ** (step + 1))) + EPSA)
    if step % 100 == 0 or step == STEPS - 1:
        hist.append(float(loss))
        print(f"шаг {step:5d} | loss {float(loss):.4f} | lr {lr:.5f} | {time.time()-t0:.0f}с", flush=True)
print("обучение заняло", round(time.time() - t0, 1), "сек")
export_model()

# ------------------------------------------------------------- тесты
tests = [
    "привет", "как тебя зовут", "кто ты", "как дела", "что ты умеешь",
    "столица франции", "столица россии", "сколько будет 15 плюс 8",
    "сколько будет 7 умножить на 8", "что больше 33 или 17",
    "кто написал евгения онегина", "почему летом жарко",
    "сколько планет в солнечной системе", "как погода", "как испечь пирог",
]
print("\n--- тесты ---")
for q in tests:
    a = generate(q, temperature=0.5, top_k=3, seed=abs(hash(q)) % (2**32))
    print(f"Q: {q}\nA: {a}")

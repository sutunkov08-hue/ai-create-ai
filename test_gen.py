# -*- coding: utf-8 -*-
import re, json, math
import numpy as np

TOK_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*|[а-яёa-z0-9]|[.,!?;:]")
def tokenize(text):
    return TOK_RE.findall(text.lower().strip())
PAD, BOS, EOS, NL, UNK = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "<nl>", "<unk>"]
lines = open("data/corpus.txt", encoding="utf-8").read().strip().split("\n")
word2id = {}
def wid(w):
    if w not in word2id: word2id[w] = len(word2id) + len(SPECIALS)
    return word2id[w]
stream = [BOS]; target_mask = [0]
for i in range(0, len(lines) - 1, 2):
    qline, aline = lines[i], lines[i + 1]
    toks = [BOS, wid("вопрос"), wid(":")] + [wid(t) for t in tokenize(qline[7:])]
    toks += [NL, wid("ответ"), wid(":")]
    atoks = [wid(t) for t in tokenize(aline[6:])] + [EOS]
    toks += atoks
    tm = [0] * (len(toks) - len(atoks)) + [1] * len(atoks)
    tm = tm[1:] + [0]
    stream += toks; target_mask += tm
vocab_size = len(word2id) + len(SPECIALS)
itos = {v: k for k, v in word2id.items()}; itos.update({i: s for i, s in enumerate(SPECIALS)})

blob = json.load(open("site/model.json", encoding="utf-8"))
meta = blob["meta"]
assert meta["vocab"] == {**{s: i for i, s in enumerate(SPECIALS)}, **word2id}, "словарь не совпал!"
P = {k: np.array(v) for k, v in blob["weights"].items()}
D, HEADS, DH, FF, N_LAYER = meta["d_model"], meta["heads"], meta["dh"], meta["ff"], meta["n_layer"]
MAXPOS = meta["max_len"]
print("веса загружены, словарь совпал, vocab_size =", vocab_size)

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * g + b

# --- 1) точность на ПРАВИЛЬНОЙ маске, все 4120 примеров
examples = []
si = 0; n = len(stream)
while si < n:
    j = si
    while j < n and stream[j] != EOS: j += 1
    seg = stream[si:j+1]; sm = target_mask[si:j+1]; si = j + 1
    if len(seg) < 6 or len(seg) > MAXPOS: continue
    examples.append((seg + [PAD] * (MAXPOS - len(seg)), sm + [0] * (MAXPOS - len(seg))))
X = np.array([e[0] for e in examples], dtype=np.int32)
M = np.array([e[1] for e in examples], dtype=np.float32)
correct = 0; total = 0
for s in range(0, len(X), 512):
    Xb = X[s:s+512]; Mb = M[s:s+512]
    B, T = Xb.shape
    causal = np.tril(np.ones((T, T), dtype=bool))
    x = P["tok_emb"][Xb] + P["pos_emb"][:T]
    for l in range(N_LAYER):
        xn = layernorm(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
        qkv = (xn @ P[f"Wqkv{l}"] + P[f"bqkv{l}"]).reshape(B, T, 3, HEADS, DH).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pre = np.where(causal, q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH), -1e9)
        e = np.exp(pre - pre.max(-1, keepdims=True)); att = e / e.sum(-1, keepdims=True)
        zo = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x2 = x + zo @ P[f"Wo{l}"] + P[f"bo{l}"]
        hn = layernorm(x2, P[f"ln2g{l}"], P[f"ln2b{l}"])
        x = x2 + np.maximum(hn @ P[f"W1_{l}"] + P[f"b1_{l}"], 0) @ P[f"W2_{l}"] + P[f"b2_{l}"]
    xf = layernorm(x, P["lnfg"], P["lnfb"])
    lg = xf @ P["tok_emb"].T
    pred = lg.argmax(-1)
    tgt = np.zeros_like(Xb); tgt[:, :-1] = Xb[:, 1:]
    mt = Mb[:, :-1] > 0
    correct += int((pred[:, :-1][mt] == tgt[:, :-1][mt]).sum()); total += int(mt.sum())
print(f"точность argmax на ответных позициях (правильная маска): {correct}/{total} = {correct/total*100:.2f}%")

# --- 2) генерация
def decode_tokens(toks):
    words = []
    for t in toks:
        if t in ("eos", "<eos>", "<nl>", "<pad>", "<unk>", "<bos>"): continue
        if t in (".", "!", "?", ";", ":", ","):
            if words: words[-1] += t
        else: words.append(t)
    s = " ".join(words)
    return s[0].upper() + s[1:] if s else s

def generate(text, temperature=0.8, top_k=5, max_new=24, seed=None):
    r = np.random.default_rng(seed)
    ids = [BOS, word2id["вопрос"], word2id[":"]] + [word2id.get(w, UNK) for w in tokenize(text)]
    ids += [NL, word2id["ответ"], word2id[":"]]
    out = []
    for _ in range(max_new):
        ids_c = ids[-MAXPOS:]; T = len(ids_c)
        x = P["tok_emb"][np.asarray(ids_c)] + P["pos_emb"][:T]
        causal = np.tril(np.ones((T, T), dtype=bool))
        for l in range(N_LAYER):
            xn = layernorm(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
            qkv = (xn @ P[f"Wqkv{l}"] + P[f"bqkv{l}"]).reshape(T, 3, HEADS, DH).transpose(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            pre = np.where(causal, q @ k.transpose(0, 2, 1) / math.sqrt(DH), -1e9)
            e = np.exp(pre - pre.max(-1, keepdims=True)); att = e / e.sum(-1, keepdims=True)
            zo = (att @ v).transpose(1, 0, 2).reshape(T, D)
            x = x + zo @ P[f"Wo{l}"] + P[f"bo{l}"]
            hn = layernorm(x, P[f"ln2g{l}"], P[f"ln2b{l}"])
            f1 = hn @ P[f"W1_{l}"] + P[f"b1_{l}"]
            x = x + np.maximum(f1, 0) @ P[f"W2_{l}"] + P[f"b2_{l}"]
        xf = layernorm(x, P["lnfg"], P["lnfb"])
        logits = xf[-1] @ P["tok_emb"].T
        assert logits.shape == (vocab_size,), logits.shape
        logits[PAD] = -1e9; logits[UNK] = -1e9; logits[BOS] = -1e9; logits[NL] = -1e9
        logits[EOS] -= 0.5
        logits /= max(temperature, 1e-3)
        top = np.argsort(logits)[-top_k:]
        pt = np.exp(logits[top] - logits[top].max()); pt /= pt.sum()
        nxt = int(r.choice(top, p=pt))
        if nxt == EOS: break
        ids.append(nxt); out.append(itos[nxt])
        if len(ids) >= MAXPOS: break
    return decode_tokens(out)

tests = [
    "привет", "как тебя зовут", "кто ты", "как дела", "что ты умеешь",
    "столица франции", "столица россии", "сколько будет 15 плюс 8",
    "сколько будет 7 умножить на 8", "что больше 33 или 17",
    "кто написал евгения онегина", "почему летом жарко",
    "сколько планет в солнечной системе", "как погода", "как испечь пирог",
    "столица японии", "сколько будет 20 минус 13", "пока",
]
print("\n--- генерация (temp=0.7, top_k=3) ---")
for q in tests:
    a = generate(q, temperature=0.7, top_k=3, seed=42)
    print("Q:", q)
    print("A:", a)

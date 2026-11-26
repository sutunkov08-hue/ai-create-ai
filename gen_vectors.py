# -*- coding: utf-8 -*-
import json, math, re
import numpy as np

blob = json.load(open("site/model.json", encoding="utf-8"))
meta, W0 = blob["meta"], blob["weights"]
W = {k: np.array(v) for k, v in W0.items()}
V = meta["vocab"]; sp = meta["specials"]
D, HEADS, DH, FF, NLAY = meta["d_model"], meta["heads"], meta["dh"], meta["ff"], meta["n_layer"]
itov = {v: k for k, v in V.items()}

TOK_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*|[а-яёa-z0-9]|[.,!?;:]")
def encode_question(text):
    ids = [sp["bos"], V["вопрос"], V[":"]]
    for w in TOK_RE.findall(text.lower().strip()):
        ids.append(V.get(w, sp["unk"]))
    ids += [sp["nl"], V["ответ"], V[":"]]
    return ids

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * g + b

def forward_logits(ids):
    ids = np.array([ids], dtype=np.int32)
    B, T = ids.shape
    causal = np.tril(np.ones((T, T), dtype=bool))
    x = W["tok_emb"][ids] + W["pos_emb"][:T]
    for l in range(NLAY):
        xn = layernorm(x, W[f"ln1g{l}"], W[f"ln1b{l}"])
        qkv = (xn @ W[f"Wqkv{l}"] + W[f"bqkv{l}"]).reshape(B, T, 3, HEADS, DH).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pre = np.where(causal, q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH), -1e9)
        e = np.exp(pre - pre.max(-1, keepdims=True)); att = e / e.sum(-1, keepdims=True)
        zo = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x2 = x + zo @ W[f"Wo{l}"] + W[f"bo{l}"]
        hn = layernorm(x2, W[f"ln2g{l}"], W[f"ln2b{l}"])
        x = x2 + np.maximum(hn @ W[f"W1_{l}"] + W[f"b1_{l}"], 0) @ W[f"W2_{l}"] + W[f"b2_{l}"]
    xf = layernorm(x, W["lnfg"], W["lnfb"])
    return (xf @ W["tok_emb"].T)[0]

prompts = ["привет", "столица франции", "сколько будет 12 плюс 7", "кто ты"]
cases = []
for p in prompts:
    ids = encode_question(p)
    logits = forward_logits(ids)      # (T, V); JS forwardLogits возвращает только последнюю строку
    last = logits[-1]
    top = sorted(range(len(last)), key=lambda i: -last[i])[:5]
    cases.append({"prompt": p, "ids": ids,
                  "logits": [round(float(x), 7) for x in last],
                  "top": [itov[i] for i in top]})
json.dump({"cases": cases}, open("site/test_vectors.json", "w", encoding="utf-8"), ensure_ascii=False)
print("test_vectors.json записан, случаев:", len(cases))
for c in cases:
    print(" ", c["prompt"], "->", c["top"])

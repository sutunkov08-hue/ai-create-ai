/* Нейро — мини-GPT на чистом JavaScript.
   Точно повторяет forward-pass трансформера, обученного в train.py (numpy).
   Никаких API: все веса лежат в model.json, всё считается в браузере. */

const state = { weights: null, meta: null, ready: false, temperature: 0.6, topK: 4, busy: false };

function softLog(msg) { console.log("[нейро]", msg); }

function applyModelData(data) {
  state.meta = data.meta;
  state.weights = data.weights;
  const inv = new Array(state.meta.vocab_size);
  for (const w in state.meta.vocab) inv[state.meta.vocab[w]] = w;
  state.meta.vocab_key = inv;
  state.ready = true;
  softLog("модель загружена: " + state.meta.params + " параметров, словарь " + state.meta.vocab_size);
  document.dispatchEvent(new CustomEvent("modelready", { detail: state.meta }));
}

function failModel(msg) {
  state.loadError = msg;
  softLog(msg);
  document.dispatchEvent(new CustomEvent("modelerror", { detail: { message: msg } }));
}

function loadEmbeddedModel() {
  if (window.NEURO_WEIGHTS) { applyModelData(window.NEURO_WEIGHTS); return; }
  const s = document.createElement("script");
  s.src = "model.js";
  s.onload = () => {
    if (window.NEURO_WEIGHTS) applyModelData(window.NEURO_WEIGHTS);
    else failModel("веса не найдены: нет ни model.json, ни model.js");
  };
  s.onerror = () => failModel("веса не загрузились: модель лежит в model.js рядом с index.html");
  document.head.appendChild(s);
}

async function loadModel() {
  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    applyModelData(await res.json());
  } catch (e) {
    softLog("fetch model.json недоступен (" + e + "), пробую model.js…");
    loadEmbeddedModel();
  }
}

/* ---------- числовые примитивы ---------- */
function matmul(A, Bm, rows, inner, cols) {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const a0 = r * inner;
    const o0 = r * cols;
    for (let i = 0; i < inner; i++) {
      const av = A[a0 + i];
      if (av === 0) continue;
      const brow = Bm[i];
      for (let j = 0; j < cols; j++) out[o0 + j] += av * brow[j];
    }
  }
  return out;
}

function layernorm(x, g, b, D, eps) {
  eps = eps || 1e-5;
  const out = new Float64Array(x.length);
  const T = x.length / D;
  for (let t = 0; t < T; t++) {
    const off = t * D;
    let mu = 0;
    for (let i = 0; i < D; i++) mu += x[off + i];
    mu /= D;
    let v = 0;
    for (let i = 0; i < D; i++) { const d = x[off + i] - mu; v += d * d; }
    v /= D;
    const sigma = Math.sqrt(v + eps);
    for (let i = 0; i < D; i++) {
      const h = (x[off + i] - mu) / sigma;
      out[off + i] = h * g[i] + b[i];
    }
  }
  return out;
}

function projectOntoVocab(vec, tok, vocabSize, D) {
  const out = new Float64Array(vocabSize);
  for (let j = 0; j < vocabSize; j++) {
    const row = tok[j];
    let s = 0;
    for (let d = 0; d < D; d++) s += vec[d] * row[d];
    out[j] = s;
  }
  return out;
}

/* ---------- токенизация (та же, что в train.py) ---------- */
const TOK_RE = /[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*|[а-яёa-z0-9]|[.,!?;:]/g;

function tokenize(text) {
  return (text.toLowerCase().trim().match(TOK_RE)) || [];
}

function encodeQuestion(text, meta) {
  const V = meta.vocab, sp = meta.specials;
  const ids = [sp.bos, V["вопрос"], V[":"]];
  for (const w of tokenize(text)) ids.push(w in V ? V[w] : sp.unk);
  ids.push(sp.nl, V["ответ"], V[":"]);
  return ids;
}

/* ---------- forward ---------- */
function forwardLogits(ids, W, meta) {
  const D = meta.d_model, H = meta.heads, DH = meta.dh, L = meta.n_layer;
  const T = ids.length;
  const tok = W.tok_emb, pos = W.pos_emb;
  let x = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    const tr = tok[ids[t]], pr = pos[t];
    for (let i = 0; i < D; i++) x[t * D + i] = tr[i] + pr[i];
  }
  for (let l = 0; l < L; l++) {
    const xn = layernorm(x, W["ln1g" + l], W["ln1b" + l], D);
    const Wqkv = W["Wqkv" + l], bqkv = W["bqkv" + l];
    const qkv = matmul(xn, Wqkv, T, D, 3 * D);
    for (let t = 0; t < T; t++) for (let j = 0; j < 3 * D; j++) qkv[t * 3 * D + j] += bqkv[j];

    const Wo = W["Wo" + l], bo = W["bo" + l];
    const outAtt = new Float64Array(T * D);
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const offQ = h * DH, offK = D + h * DH, offV = 2 * D + h * DH;
        const q0 = t * 3 * D;
        let maxp = -Infinity;
        const scores = new Float64Array(t + 1);
        for (let s = 0; s <= t; s++) {
          const k0 = s * 3 * D;
          let dot = 0;
          for (let i = 0; i < DH; i++) dot += qkv[q0 + offQ + i] * qkv[k0 + offK + i];
          scores[s] = dot / Math.sqrt(DH);
          if (scores[s] > maxp) maxp = scores[s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) { scores[s] = Math.exp(scores[s] - maxp); sum += scores[s]; }
        for (let s = 0; s <= t; s++) {
          const wgt = scores[s] / sum;
          const k0 = s * 3 * D;
          for (let i = 0; i < DH; i++) outAtt[t * D + h * DH + i] += wgt * qkv[k0 + offV + i];
        }
      }
    }
    const proj = matmul(outAtt, Wo, T, D, D);
    for (let t = 0; t < T; t++) for (let j = 0; j < D; j++) proj[t * D + j] += bo[j];
    const x2 = new Float64Array(T * D);
    for (let i = 0; i < x.length; i++) x2[i] = x[i] + proj[i];
    const hn = layernorm(x2, W["ln2g" + l], W["ln2b" + l], D);
    const W1 = W["W1_" + l], b1 = W["b1_" + l], W2 = W["W2_" + l], b2 = W["b2_" + l];
    const f1 = matmul(hn, W1, T, D, meta.ff);
    const act = new Float64Array(f1.length);
    for (let i = 0; i < f1.length; i++) { const val = f1[i] + b1[i % meta.ff]; act[i] = val > 0 ? val : 0; }
    const f2 = matmul(act, W2, T, meta.ff, D);
    const x3 = new Float64Array(T * D);
    for (let t = 0; t < T; t++)
      for (let j = 0; j < D; j++) x3[t * D + j] = x2[t * D + j] + f2[t * D + j] + b2[j];
    x = x3;
  }
  const xf = layernorm(x, W.lnfg, W.lnfb, D);
  return projectOntoVocab(xf.subarray((T - 1) * D), tok, meta.vocab_size, D);
}

/* ---------- детокенизация: пробелы между словами, пунктуация приклеивается ---------- */
const DROPPED = new Set(["eos", "<eos>", "<nl>", "<pad>", "<unk>", "<bos>"]);

function decodeTokens(tokens) {
  const words = [];
  for (const t of tokens) {
    if (DROPPED.has(t)) continue;
    if (t.length === 1 && ".!?;:,".includes(t)) {
      if (words.length) words[words.length - 1] += t;
    } else {
      words.push(t);
    }
  }
  const s = words.join(" ");
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/* ---------- генерация ответа ---------- */
function generateAnswer(text, temperature, topK) {
  if (!state.ready) return null;
  const meta = state.meta, W = state.weights, sp = meta.specials;
  const rng = Math.random;
  const ids = encodeQuestion(text, meta);
  const outTokens = [];
  const trace = [];
  for (let step = 0; step < 24; step++) {
    const ctx = ids.slice(-meta.max_len);
    const logits = forwardLogits(ctx, W, meta);
    logits[sp.pad] = -1e9; logits[sp.unk] = -1e9; logits[sp.bos] = -1e9; logits[sp.nl] = -1e9;
    logits[sp.eos] -= 0.5;
    for (let i = 0; i < logits.length; i++) logits[i] /= Math.max(temperature, 1e-3);
    const idx = new Array(meta.vocab_size);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    idx.sort((a, b) => logits[b] - logits[a]);
    const top = idx.slice(0, topK);
    let mx = -Infinity;
    for (const t of top) mx = Math.max(mx, logits[t]);
    let sum = 0;
    const pt = top.map((t) => { const e = Math.exp(logits[t] - mx); sum += e; return e; });
    for (let i = 0; i < pt.length; i++) pt[i] /= sum;
    let r = rng(), acc = 0, nxt = top[0];
    for (let i = 0; i < top.length; i++) { acc += pt[i]; if (r <= acc) { nxt = top[i]; break; } }
    trace.push({
      word: meta.vocab_key[nxt],
      candidates: top.map((t, i) => ({ w: meta.vocab_key[t], p: pt[i] }))
    });
    if (nxt === sp.eos) break;
    ids.push(nxt);
    outTokens.push(meta.vocab_key[nxt]);
    if (ids.length >= meta.max_len) break;
  }
  state.lastTrace = trace;
  return decodeTokens(outTokens);
}

window.addEventListener("load", () => { loadModel(); });

/* ---------- живые нейроны: повреждение весов и след мыслей ---------- */
function gauss() {
  return (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.732;
}

function damageWeights(intensity) {
  if (!state.ready || state.damaged) return;
  state.backup = JSON.parse(JSON.stringify(state.weights));
  const sigma = (intensity || 1) * 0.15;
  for (const k in state.weights) {
    const m = state.weights[k];
    for (let i = 0; i < m.length; i++)
      for (let j = 0; j < m[i].length; j++) m[i][j] += sigma * gauss();
  }
  state.damaged = true;
}

function restoreWeights() {
  if (state.backup) {
    state.weights = state.backup;
    state.backup = null;
    state.damaged = false;
  }
}

window.NEURO = { state, generateAnswer, decodeTokens, tokenize, encodeQuestion, forwardLogits, damageWeights, restoreWeights };

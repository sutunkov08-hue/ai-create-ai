// Тест: тот же forward на node, сравнение с эталоном из test_vectors.json
const fs = require("fs"), vm = require("vm");
const code = fs.readFileSync(__dirname + "/app.js", "utf8");
const sandbox = { window: { addEventListener() {} }, document: { dispatchEvent() {}, addEventListener() {} },
                  fetch: () => { throw new Error("no fetch in test"); }, console, CustomEvent: class {},
                  performance: { now: () => 0 }, Math };
vm.createContext(sandbox);
vm.runInContext(code.replace(/window\.addEventListener\("load".*$/, ""), sandbox);
const { meta, weights } = JSON.parse(fs.readFileSync(__dirname + "/model.json", "utf8"));
const NEURO = sandbox.window.NEURO || sandbox.NEURO;
NEURO.state.meta = meta;
NEURO.state.weights = weights;
const inv = new Array(meta.vocab_size);
for (const w in meta.vocab) inv[meta.vocab[w]] = w;
meta.vocab_key = inv;
NEURO.state.ready = true;

const tv = JSON.parse(fs.readFileSync(__dirname + "/test_vectors.json", "utf8"));
let worst = 0;
for (const t of tv.cases) {
  const logits = NEURO.forwardLogits(t.ids, weights, meta);
  for (let i = 0; i < logits.length; i++) worst = Math.max(worst, Math.abs(logits[i] - t.logits[i]));
  const idx = Array.from({ length: meta.vocab_size }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
  const topJs = idx.slice(0, 5).map((i) => meta.vocab_key[i]);
  if (topJs.join("|") !== t.top.join("|")) {
    console.log("!! расходится top-5 для", t.prompt, "\n  js:", topJs, "\n  py:", t.top);
  }
}
console.log("максимальное расхождение logits js vs numpy:", worst.toExponential(2));
console.log(worst < 1e-6 ? "PARITY OK" : "PARITY FAIL");

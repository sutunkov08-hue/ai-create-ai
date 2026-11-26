// E2E: жадная генерация в app.js против numpy-эталона
const fs = require("fs"), vm = require("vm");
const code = fs.readFileSync(__dirname + "/app.js", "utf8");
const sandbox = { window: { addEventListener() {} }, document: { dispatchEvent() {}, addEventListener() {} },
                  fetch: () => { throw new Error("no fetch"); }, console, CustomEvent: class {},
                  performance: { now: () => 0 }, Math };
vm.createContext(sandbox);
vm.runInContext(code.replace(/window\.addEventListener\("load".*$/, ""), sandbox);
const { meta, weights } = JSON.parse(fs.readFileSync(__dirname + "/model.json", "utf8"));
const NEURO = sandbox.window.NEURO;
NEURO.state.meta = meta; NEURO.state.weights = weights;
const inv = new Array(meta.vocab_size);
for (const w in meta.vocab) inv[meta.vocab[w]] = w;
meta.vocab_key = inv; NEURO.state.ready = true;
// фиксируем Math.random в единицу => жадный выбор первого из top-1
vm.runInContext("Math.random = () => 0.999999;", sandbox);

const prompts = ["привет", "как тебя зовут", "столица франции", "столица японии",
                 "сколько будет 15 плюс 8", "что больше 33 или 17", "пока"];
console.log("--- JS генерация (greedy) ---");
for (const p of prompts) {
  console.log("Q:", p);
  console.log("A:", NEURO.generateAnswer(p, 1.0, 1));
}

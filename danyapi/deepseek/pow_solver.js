const fs = require("fs");
const path = require("path");
const wasmPath = path.join(__dirname, "sha3_wasm_bg.wasm");

if (!fs.existsSync(wasmPath)) {
  process.stdout.write(
    JSON.stringify({ error: "sha3_wasm_bg.wasm not found" }),
  );
  process.exit(1);
}

const wasmBuf = fs.readFileSync(wasmPath);

let instancePromise = null;
function getInstance() {
  if (!instancePromise) instancePromise = WebAssembly.instantiate(wasmBuf, {});
  return instancePromise;
}

function solve(challenge, prefix, difficulty) {
  return getInstance().then(({ instance }) => {
    const {
      memory,
      wasm_solve,
      __wbindgen_add_to_stack_pointer,
      __wbindgen_export_0: malloc,
      __wbindgen_export_2: free,
    } = instance.exports;
    let m = new Uint8Array(memory.buffer);
    let view = new DataView(memory.buffer);
    function refresh() {
      m = new Uint8Array(memory.buffer);
      view = new DataView(memory.buffer);
    }
    const enc = new TextEncoder();
    const hexBytes = enc.encode(challenge);
    const pBytes = enc.encode(prefix);
    const hexPtr = malloc(hexBytes.length, 1);
    refresh();
    m.set(hexBytes, hexPtr);
    const pPtr = malloc(pBytes.length, 1);
    refresh();
    m.set(pBytes, pPtr);
    const retptr = __wbindgen_add_to_stack_pointer(-16);
    try {
      try {
        wasm_solve(
          retptr,
          hexPtr,
          hexBytes.length,
          pPtr,
          pBytes.length,
          difficulty,
        );
      } catch (e) {
        wasm_solve(
          retptr,
          hexPtr,
          hexBytes.length,
          pPtr,
          pBytes.length,
          BigInt(difficulty),
        );
      }
      refresh();
      const status = view.getInt32(retptr, true);
      if (status === 0) return null;
      const floatVal = view.getFloat64(retptr + 8, true);
      if (
        Number.isInteger(floatVal) &&
        floatVal >= 0 &&
        floatVal <= Number.MAX_SAFE_INTEGER
      ) {
        return floatVal;
      }
      const bigVal = view.getBigUint64(retptr + 8, true);
      return Number(bigVal);
    } finally {
      __wbindgen_add_to_stack_pointer(16);
      if (typeof free === "function") {
        free(hexPtr, hexBytes.length, 1);
        free(pPtr, pBytes.length, 1);
      }
    }
  });
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  let req;
  try {
    req = JSON.parse(input);
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: "bad json: " + e.message }));
    process.exit(1);
    return;
  }
  const prefix = `${req.salt}_${req.expire_at}_`;
  solve(req.challenge, prefix, req.difficulty)
    .then((answer) => {
      if (answer === null) {
        process.stdout.write(
          JSON.stringify({ error: "solver returned no answer" }),
        );
      } else {
        process.stdout.write(JSON.stringify({ answer }));
      }
    })
    .catch((err) => {
      process.stdout.write(
        JSON.stringify({ error: String((err && err.message) || err) }),
      );
      process.exit(1);
    });
});

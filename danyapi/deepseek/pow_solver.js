const fs = require('fs');
const path = require('path');
const wasmPath = path.join(__dirname, 'sha3_wasm_bg.wasm');
const wasmBuf = fs.readFileSync(wasmPath);

let instancePromise = null;
function getInstance() {
  if (!instancePromise) instancePromise = WebAssembly.instantiate(wasmBuf, {});
  return instancePromise;
}

function solve(challenge, prefix, difficulty) {
  return getInstance().then(({ instance }) => {
    const { memory, wasm_solve, __wbindgen_add_to_stack_pointer, __wbindgen_export_0: malloc } = instance.exports;
    let m = new Uint8Array(memory.buffer);
    let view = new DataView(memory.buffer);
    function refresh() { m = new Uint8Array(memory.buffer); view = new DataView(memory.buffer); }
    function writeStr(ptr, s) { for (let i = 0; i < s.length; i++) m[ptr + i] = s.charCodeAt(i); }
    const hexPtr = malloc(challenge.length, 1); refresh(); writeStr(hexPtr, challenge);
    const pPtr = malloc(prefix.length, 1); refresh(); writeStr(pPtr, prefix);
    const retptr = __wbindgen_add_to_stack_pointer(-16);
    try {
      wasm_solve(retptr, hexPtr, challenge.length, pPtr, prefix.length, difficulty);
      refresh();
      const status = view.getInt32(retptr, true);
      const value = view.getFloat64(retptr + 8, true);
      return status !== 0 ? Number(value) : null;
    } finally {
      __wbindgen_add_to_stack_pointer(16);
    }
  });
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  let req;
  try {
    req = JSON.parse(input);
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: 'bad json: ' + e.message }));
    process.exit(1);
    return;
  }
  const prefix = `${req.salt}_${req.expire_at}_`;
  solve(req.challenge, prefix, req.difficulty)
    .then(answer => {
      if (answer === null) {
        process.stdout.write(JSON.stringify({ error: 'solver returned no answer' }));
      } else {
        process.stdout.write(JSON.stringify({ answer }));
      }
    })
    .catch(err => {
      process.stdout.write(JSON.stringify({ error: String(err && err.message || err) }));
      process.exit(1);
    });
});

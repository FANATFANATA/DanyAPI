const fs = require("fs");
const path = require("path");
const wasmPath = path.join(__dirname, "sha3_wasm_bg.wasm");
const useWasm = fs.existsSync(wasmPath);
const RATE = 136;
const ROUNDS = 23;
const MAX_INPUT = 8190;
const MAX_DIFFICULTY = 2000000000;
const M64 = 0xffffffffffffffffn;
const RC = [
  0x0000000000000001n,
  0x0000000000008082n,
  0x800000000000808an,
  0x8000000080008000n,
  0x000000000000808bn,
  0x0000000080000001n,
  0x8000000080008081n,
  0x8000000000008009n,
  0x000000000000008an,
  0x0000000000000088n,
  0x0000000080008009n,
  0x000000008000000an,
  0x000000008000808bn,
  0x800000000000008bn,
  0x8000000000008089n,
  0x8000000000008003n,
  0x8000000000008002n,
  0x8000000000000080n,
  0x000000000000800an,
  0x800000008000000an,
  0x8000000080008081n,
  0x8000000000008080n,
  0x0000000080000001n,
  0x8000000080008008n,
];
const ROUND_RC = RC.slice(1, 1 + ROUNDS);
const ROT = [
  [0, 36, 3, 41, 18],
  [1, 44, 10, 45, 2],
  [62, 6, 43, 15, 61],
  [28, 55, 25, 21, 56],
  [27, 20, 39, 8, 14],
];
function rol64(x, n) {
  return ((x << BigInt(n)) | (x >> BigInt(64 - n))) & M64;
}
function keccakF(st) {
  const c = new Array(5);
  const d = new Array(5);
  for (let r = 0; r < ROUNDS; r++) {
    for (let x = 0; x < 5; x++)
      c[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
    for (let x = 0; x < 5; x++)
      d[x] = c[(x + 4) % 5] ^ rol64(c[(x + 1) % 5], 1);
    for (let x = 0; x < 5; x++)
      for (let y = 0; y < 5; y++) st[x + 5 * y] ^= d[x];
    const b = new Array(25);
    for (let x = 0; x < 5; x++)
      for (let y = 0; y < 5; y++)
        b[y + 5 * ((2 * x + 3 * y) % 5)] = rol64(st[x + 5 * y], ROT[x][y]);
    for (let x = 0; x < 5; x++)
      for (let y = 0; y < 5; y++)
        st[x + 5 * y] =
          b[x + 5 * y] ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
    st[0] ^= ROUND_RC[r];
  }
}
function laneAt(bytes, off) {
  let v = 0n;
  for (let b = 0; b < 8; b++) v |= BigInt(bytes[off + b]) << BigInt(8 * b);
  return v;
}
function hexNibble(code) {
  if (code >= 0x30 && code <= 0x39) return code - 0x30;
  if (code >= 0x61 && code <= 0x66) return code - 0x57;
  if (code >= 0x41 && code <= 0x46) return code - 0x37;
  return -1;
}
function hexToBytes(hex) {
  if (typeof hex !== "string") return null;
  if (hex.length === 0 || hex.length % 2 || hex.length > 64) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    const hi = hexNibble(hex.charCodeAt(i));
    const lo = hexNibble(hex.charCodeAt(i + 1));
    if (hi < 0 || lo < 0) return null;
    out[i / 2] = (hi << 4) | lo;
  }
  return out;
}
function absorbPrefix(bytes) {
  const st = new Array(25).fill(0n);
  let pos = 0;
  while (bytes.length - pos >= RATE) {
    for (let i = 0; i < RATE; i += 8) st[i >> 3] ^= laneAt(bytes, pos + i);
    keccakF(st);
    pos += RATE;
  }
  const rem = bytes.length - pos;
  for (let i = 0; i < rem; i++)
    st[i >> 3] ^= BigInt(bytes[pos + i]) << BigInt(8 * (i & 7));
  return { st, off0: rem };
}
function counterMatches(base, off0, digits, target) {
  const st = base.slice();
  let off = off0;
  for (let i = 0; i < digits.length; i++) {
    st[off >> 3] ^= BigInt(digits.charCodeAt(i)) << BigInt(8 * (off & 7));
    off++;
    if (off === RATE) {
      keccakF(st);
      off = 0;
    }
  }
  st[off >> 3] ^= 6n << BigInt(8 * (off & 7));
  off++;
  if (off === RATE) {
    keccakF(st);
    off = 0;
  }
  st[16] ^= 0x8000000000000000n;
  keccakF(st);
  for (let i = 0; i < target.length; i++) {
    if (Number((st[i >> 3] >> BigInt(8 * (i & 7))) & 0xffn) !== target[i])
      return false;
  }
  return true;
}
function nextDigits(s) {
  const a = s.split("");
  let i = a.length - 1;
  while (i >= 0 && a[i] === "9") {
    a[i] = "0";
    i--;
  }
  if (i < 0) a.unshift("1");
  else a[i] = String.fromCharCode(a[i].charCodeAt(0) + 1);
  return a.join("");
}
function solveJs(challenge, prefix, difficulty) {
  const target = hexToBytes(challenge);
  if (!target) return null;
  const { st, off0 } = absorbPrefix(Buffer.from(prefix, "utf8"));
  let limit = Number(difficulty);
  if (!Number.isFinite(limit) || limit < 0) limit = 0;
  limit = Math.min(Math.floor(limit), MAX_DIFFICULTY);
  if (limit <= 0) return null;
  let digits = "0";
  for (let c = 0; c < limit; c++) {
    if (counterMatches(st, off0, digits, target)) return c;
    digits = nextDigits(digits);
  }
  return null;
}
function searchLimit(difficulty) {
  const limit = Math.floor(Number(difficulty));
  if (!Number.isFinite(limit) || limit <= 0) return 0;
  return Math.min(limit, MAX_DIFFICULTY);
}
function wasmFitsSingleBlock(prefix, difficulty) {
  const limit = searchLimit(difficulty);
  if (limit <= 0) return false;
  const off = Buffer.from(prefix, "utf8").length % RATE;
  return off + String(limit - 1).length <= RATE - 2;
}
let instancePromise = null;
function wasmBinding(instance) {
  const exports = instance.exports;
  const binding = {
    memory: exports.memory,
    wasm_solve: exports.wasm_solve,
    malloc: exports.__wbindgen_export_0,
    stack: exports.__wbindgen_add_to_stack_pointer,
  };
  if (
    !binding.memory ||
    typeof binding.wasm_solve !== "function" ||
    typeof binding.malloc !== "function" ||
    typeof binding.stack !== "function"
  )
    throw new Error("wasm solver exports are unavailable");
  return binding;
}
function getBinding() {
  if (!instancePromise)
    instancePromise = WebAssembly.instantiate(
      fs.readFileSync(wasmPath),
      {},
    ).then((result) => wasmBinding(result.instance));
  return instancePromise;
}
function solveWasm(binding, challenge, prefix, difficulty) {
  const challengeBytes = Buffer.from(challenge, "utf8");
  const prefixBytes = Buffer.from(prefix, "utf8");
  const hexPtr = binding.malloc(challengeBytes.length, 1);
  new Uint8Array(binding.memory.buffer).set(challengeBytes, hexPtr);
  const pPtr = binding.malloc(prefixBytes.length, 1);
  new Uint8Array(binding.memory.buffer).set(prefixBytes, pPtr);
  const retptr = binding.stack(-16);
  try {
    binding.wasm_solve(
      retptr,
      hexPtr,
      challengeBytes.length,
      pPtr,
      prefixBytes.length,
      difficulty,
    );
    const view = new DataView(binding.memory.buffer);
    const status = view.getInt32(retptr, true);
    const value = view.getFloat64(retptr + 8, true);
    return status !== 0 ? Number(value) : null;
  } finally {
    binding.stack(16);
  }
}
function fail(message) {
  process.stdout.write(JSON.stringify({ error: String(message) }));
  process.exitCode = 1;
}
process.stdin.setEncoding("utf8");
let input = "";
let overflow = false;
process.stdin.on("data", (chunk) => {
  if (overflow) return;
  input += chunk;
  if (input.length > MAX_INPUT) {
    input = "";
    overflow = true;
    fail("input too large");
  }
});
process.stdin.on("end", () => {
  if (overflow) return;
  let req;
  try {
    req = JSON.parse(input);
  } catch (e) {
    fail("bad json: " + e.message);
    return;
  }
  if (req === null || typeof req !== "object" || Array.isArray(req)) {
    fail("bad request");
    return;
  }
  const { challenge, salt, expire_at: expireAt, difficulty } = req;
  if (typeof challenge !== "string" || typeof salt !== "string") {
    fail("missing challenge/salt");
    return;
  }
  if (expireAt === undefined || expireAt === null || !Number.isFinite(Number(expireAt))) {
    fail("bad expire_at");
    return;
  }
  if (difficulty === undefined || difficulty === null || !Number.isFinite(Number(difficulty))) {
    fail("bad difficulty");
    return;
  }
  const prefix = `${salt}_${expireAt}_`;
  const attempt =
    useWasm && wasmFitsSingleBlock(prefix, difficulty)
      ? getBinding().then(
          (binding) => solveWasm(binding, challenge, prefix, Number(difficulty)),
          () => solveJs(challenge, prefix, difficulty),
        )
      : Promise.resolve(solveJs(challenge, prefix, difficulty));
  attempt
    .then((answer) => {
      if (answer === null || answer === undefined) {
        process.stdout.write(
          JSON.stringify({ error: "solver returned no answer" }),
        );
        process.exitCode = 1;
      } else {
        process.stdout.write(JSON.stringify({ answer }));
      }
    })
    .catch((err) => {
      fail((err && err.message) || err);
    });
});

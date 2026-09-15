const fs = require("fs");
const path = require("path");
const wasmPath = path.join(__dirname, "sha3_wasm_bg.wasm");
const useWasm = fs.existsSync(wasmPath);
const RATE = 136;
const ROUNDS = 23;
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
function hexToBytes(hex) {
  if (typeof hex !== "string") return null;
  if (hex.length % 2 || hex.length > 64) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    const hi = parseInt(hex[i], 16);
    const lo = parseInt(hex[i + 1], 16);
    if (isNaN(hi) || isNaN(lo)) return null;
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
  const pre = Buffer.from(prefix, "utf8");
  const { st, off0 } = absorbPrefix(pre);
  let limit = Number(difficulty);
  if (!Number.isFinite(limit) || limit < 0) limit = 0;
  limit = Math.min(Math.floor(limit), 2000000000);
  let digits = "0";
  for (let c = 0; c < limit; c++) {
    if (counterMatches(st, off0, digits, target)) return c;
    digits = nextDigits(digits);
  }
  return null;
}
let instancePromise = null;
function getInstance() {
  if (!instancePromise)
    instancePromise = WebAssembly.instantiate(fs.readFileSync(wasmPath), {});
  return instancePromise;
}
function solveWasm(challenge, prefix, difficulty) {
  return getInstance().then(({ instance }) => {
    const {
      memory,
      wasm_solve,
      __wbindgen_add_to_stack_pointer,
      __wbindgen_export_0: malloc,
    } = instance.exports;
    let m = new Uint8Array(memory.buffer);
    let view = new DataView(memory.buffer);
    function refresh() {
      m = new Uint8Array(memory.buffer);
      view = new DataView(memory.buffer);
    }
    function writeStr(ptr, s) {
      for (let i = 0; i < s.length; i++) m[ptr + i] = s.charCodeAt(i);
    }
    const hexPtr = malloc(challenge.length, 1);
    refresh();
    writeStr(hexPtr, challenge);
    const pPtr = malloc(prefix.length, 1);
    refresh();
    writeStr(pPtr, prefix);
    const retptr = __wbindgen_add_to_stack_pointer(-16);
    try {
      wasm_solve(
        retptr,
        hexPtr,
        challenge.length,
        pPtr,
        prefix.length,
        difficulty,
      );
      refresh();
      const status = view.getInt32(retptr, true);
      const value = view.getFloat64(retptr + 8, true);
      return status !== 0 ? Number(value) : null;
    } finally {
      __wbindgen_add_to_stack_pointer(16);
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
  const attempt = useWasm
    ? solveWasm(req.challenge, prefix, req.difficulty)
    : Promise.resolve(solveJs(req.challenge, prefix, req.difficulty));
  attempt
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

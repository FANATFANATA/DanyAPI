#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define RATE 136
#define ROUNDS 23
#define MAX_DIGITS 20

static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline uint64_t rotl64(uint64_t x, int n) {
  return (x << n) | (x >> (64 - n));
}

static void keccak_f(uint64_t *s) {
  uint64_t bc[5], t, p[25];
  for (int r = 0; r < ROUNDS; r++) {
    bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

    t = bc[4] ^ rotl64(bc[1], 1);
    s[0] ^= t;
    s[5] ^= t;
    s[10] ^= t;
    s[15] ^= t;
    s[20] ^= t;
    t = bc[0] ^ rotl64(bc[2], 1);
    s[1] ^= t;
    s[6] ^= t;
    s[11] ^= t;
    s[16] ^= t;
    s[21] ^= t;
    t = bc[1] ^ rotl64(bc[3], 1);
    s[2] ^= t;
    s[7] ^= t;
    s[12] ^= t;
    s[17] ^= t;
    s[22] ^= t;
    t = bc[2] ^ rotl64(bc[4], 1);
    s[3] ^= t;
    s[8] ^= t;
    s[13] ^= t;
    s[18] ^= t;
    s[23] ^= t;
    t = bc[3] ^ rotl64(bc[0], 1);
    s[4] ^= t;
    s[9] ^= t;
    s[14] ^= t;
    s[19] ^= t;
    s[24] ^= t;

    p[0] = s[0];
    p[10] = rotl64(s[1], 1);
    p[20] = rotl64(s[2], 62);
    p[5] = rotl64(s[3], 28);
    p[15] = rotl64(s[4], 27);
    p[16] = rotl64(s[5], 36);
    p[1] = rotl64(s[6], 44);
    p[11] = rotl64(s[7], 6);
    p[21] = rotl64(s[8], 55);
    p[6] = rotl64(s[9], 20);
    p[7] = rotl64(s[10], 3);
    p[17] = rotl64(s[11], 10);
    p[2] = rotl64(s[12], 43);
    p[12] = rotl64(s[13], 25);
    p[22] = rotl64(s[14], 39);
    p[23] = rotl64(s[15], 41);
    p[8] = rotl64(s[16], 45);
    p[18] = rotl64(s[17], 15);
    p[3] = rotl64(s[18], 21);
    p[13] = rotl64(s[19], 8);
    p[14] = rotl64(s[20], 18);
    p[24] = rotl64(s[21], 2);
    p[9] = rotl64(s[22], 61);
    p[19] = rotl64(s[23], 56);
    p[4] = rotl64(s[24], 14);

    for (int y = 0; y < 25; y += 5) {
      uint64_t a0 = p[y], a1 = p[y + 1], a2 = p[y + 2], a3 = p[y + 3],
               a4 = p[y + 4];
      s[y] = a0 ^ ((~a1) & a2);
      s[y + 1] = a1 ^ ((~a2) & a3);
      s[y + 2] = a2 ^ ((~a3) & a4);
      s[y + 3] = a3 ^ ((~a4) & a0);
      s[y + 4] = a4 ^ ((~a0) & a1);
    }
    s[0] ^= RC[r + 1];
  }
}

static void absorb_prefix(uint64_t st[25], const uint8_t *prefix, size_t len) {
  memset(st, 0, 25 * sizeof(uint64_t));
  size_t off = 0;
  while (len - off >= RATE) {
    for (size_t i = 0; i < RATE; i += 8) {
      uint64_t w = 0;
      for (int b = 0; b < 8; b++)
        w |= (uint64_t)prefix[off + i + b] << (8 * b);
      st[i / 8] ^= w;
    }
    keccak_f(st);
    off += RATE;
  }
  for (size_t i = 0; i < len - off; i++)
    st[i / 8] ^= (uint64_t)prefix[off + i] << (8 * (i % 8));
}

static int to_digits(uint64_t v, char *buf) {
  char tmp[MAX_DIGITS];
  int n = 0;
  do {
    if (n >= MAX_DIGITS)
      break;
    tmp[n++] = (char)('0' + (int)(v % 10));
    v /= 10;
  } while (v > 0);
  for (int i = 0; i < n; i++)
    buf[i] = tmp[n - 1 - i];
  return n;
}

static void inc_digits(char *buf, int *dlen) {
  if (*dlen < 1 || *dlen > MAX_DIGITS)
    return;
  int i = *dlen - 1;
  while (i >= 0 && buf[i] == '9') {
    buf[i] = '0';
    i--;
  }
  if (i < 0) {
    buf[0] = '1';
    for (int j = 1; j <= *dlen; j++)
      buf[j] = '0';
    (*dlen)++;
  } else {
    buf[i]++;
  }
}

static int check_counter(const uint64_t base[25], size_t off0,
                         const char *digits, int dlen,
                         const uint8_t target[32]) {
  uint64_t st[25];
  memcpy(st, base, sizeof(st));
  size_t off = off0;
  for (int i = 0; i < dlen; i++) {
    st[off >> 3] ^= (uint64_t)(uint8_t)digits[i] << (8 * (off & 7));
    off++;
    if (off == RATE) {
      keccak_f(st);
      off = 0;
    }
  }
  st[off >> 3] ^= (uint64_t)0x06 << (8 * (off & 7));
  off++;
  if (off == RATE) {
    keccak_f(st);
  }
  st[16] ^= (uint64_t)0x80 << 56;
  keccak_f(st);
  return memcmp(st, target, 32) == 0;
}

typedef struct {
  const uint64_t *base;
  size_t off0;
  const uint8_t *target;
  uint64_t start;
  uint64_t end;
  uint64_t result;
} WorkerArgs;

static void run_worker(WorkerArgs *a) {
  a->result = UINT64_MAX;
  if (a->start >= a->end)
    return;
  char digits[MAX_DIGITS];
  int dlen = to_digits(a->start, digits);
  for (uint64_t c = a->start; c < a->end; c++) {
    if (check_counter(a->base, a->off0, digits, dlen, a->target)) {
      a->result = c;
      return;
    }
    inc_digits(digits, &dlen);
  }
}

#if defined(_WIN32)
static DWORD WINAPI worker(LPVOID arg) {
  run_worker((WorkerArgs *)arg);
  return 0;
}
#else
static void *worker(void *arg) {
  run_worker((WorkerArgs *)arg);
  return NULL;
}
#endif

static int detect_threads(void) {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  int n = (int)si.dwNumberOfProcessors;
  return n > 0 ? n : 1;
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
#endif
}

static int hex_to_bytes(const char *hex, uint8_t *out) {
  size_t n = strlen(hex);
  if (n % 2)
    return -1;
  for (size_t i = 0; i < n; i += 2) {
    int hi = hex[i], lo = hex[i + 1];
    int hv = (hi >= '0' && hi <= '9')   ? hi - '0'
             : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10
             : (hi >= 'A' && hi <= 'F') ? hi - 'A' + 10
                                        : -1;
    int lv = (lo >= '0' && lo <= '9')   ? lo - '0'
             : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10
             : (lo >= 'A' && lo <= 'F') ? lo - 'A' + 10
                                        : -1;
    if (hv < 0 || lv < 0)
      return -1;
    out[i / 2] = (uint8_t)((hv << 4) | lv);
  }
  return (int)(n / 2);
}

static const char *find_json_str(const char *json, const char *key, char *buf,
                                 size_t bufsz) {
  char pat[64];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char *p = strstr(json, pat);
  if (!p)
    return NULL;
  p = strchr(p + strlen(pat), ':');
  if (!p)
    return NULL;
  p++;
  while (*p == ' ' || *p == '\t')
    p++;
  if (*p != '"')
    return NULL;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i + 1 < bufsz)
    buf[i++] = *p++;
  buf[i] = '\0';
  return buf;
}

static long long find_json_ll(const char *json, const char *key) {
  char pat[64];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char *p = strstr(json, pat);
  if (!p)
    return -1;
  p = strchr(p + strlen(pat), ':');
  if (!p)
    return -1;
  return strtoll(p + 1, NULL, 10);
}

int main(void) {
  char input[8192];
  size_t n = fread(input, 1, sizeof(input) - 1, stdin);
  input[n] = '\0';

  char challenge[128] = {0}, salt[4096] = {0};
  if (!find_json_str(input, "challenge", challenge, sizeof(challenge)) ||
      !find_json_str(input, "salt", salt, sizeof(salt))) {
    puts("{\"error\":\"missing challenge/salt\"}");
    return 1;
  }
  long long expire_at = find_json_ll(input, "expire_at");
  long long difficulty = find_json_ll(input, "difficulty");
  if (expire_at < 0 || difficulty <= 0) {
    puts("{\"error\":\"bad expire_at/difficulty\"}");
    return 1;
  }

  uint8_t target[32];
  if (hex_to_bytes(challenge, target) != 32) {
    puts("{\"error\":\"bad challenge hex\"}");
    return 1;
  }

  char prefix[4120];
  int plen = snprintf(prefix, sizeof(prefix), "%s_%lld_", salt, expire_at);
  if (plen < 0 || (size_t)plen >= sizeof(prefix)) {
    puts("{\"error\":\"salt too long\"}");
    return 1;
  }

  uint64_t base[25];
  absorb_prefix(base, (const uint8_t *)prefix, (size_t)plen);
  size_t off0 = (size_t)plen % RATE;

  uint64_t limit =
      difficulty < 2000000000LL ? (uint64_t)difficulty : 2000000000ULL;
  if (limit == 0) {
    puts("{\"error\":\"answer not found in range\"}");
    return 1;
  }

  int nthreads = detect_threads();
  const char *env = getenv("POW_SOLVER_THREADS");
  if (env && env[0]) {
    int v = atoi(env);
    if (v > 0)
      nthreads = v;
  }
#if defined(_WIN32)
  if (nthreads > 64)
    nthreads = 64;
#endif
  if ((uint64_t)nthreads > limit)
    nthreads = (int)limit;

  WorkerArgs *args = (WorkerArgs *)calloc((size_t)nthreads, sizeof(WorkerArgs));
  if (!args) {
    puts("{\"error\":\"out of memory\"}");
    return 1;
  }
#if defined(_WIN32)
  HANDLE *threads = (HANDLE *)calloc((size_t)nthreads, sizeof(HANDLE));
#else
  pthread_t *threads = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
#endif
  if (!threads) {
    free(args);
    puts("{\"error\":\"out of memory\"}");
    return 1;
  }

  uint64_t chunk = (limit + (uint64_t)nthreads - 1) / (uint64_t)nthreads;
  for (int i = 0; i < nthreads; i++) {
    args[i].base = base;
    args[i].off0 = off0;
    args[i].target = target;
    args[i].start = (uint64_t)i * chunk;
    uint64_t end = args[i].start + chunk;
    args[i].end = end > limit ? limit : end;
    args[i].result = UINT64_MAX;
#if defined(_WIN32)
    threads[i] = CreateThread(NULL, 0, worker, &args[i], 0, NULL);
    if (!threads[i])
      run_worker(&args[i]);
#else
    pthread_create(&threads[i], NULL, worker, &args[i]);
#endif
  }

#if defined(_WIN32)
  for (int i = 0; i < nthreads; i++) {
    if (threads[i]) {
      WaitForSingleObject(threads[i], INFINITE);
      CloseHandle(threads[i]);
    }
  }
#else
  for (int i = 0; i < nthreads; i++)
    pthread_join(threads[i], NULL);
#endif

  uint64_t best = UINT64_MAX;
  for (int i = 0; i < nthreads; i++)
    if (args[i].result < best)
      best = args[i].result;

  free(threads);
  free(args);

  if (best != UINT64_MAX) {
    printf("{\"answer\":%llu}\n", (unsigned long long)best);
    return 0;
  }
  puts("{\"error\":\"answer not found in range\"}");
  return 1;
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3")
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <stdlib.h>
#define FORCE_INLINE __forceinline
#define ROTL64(x, n) _rotl64((x), (n))
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#else
#define FORCE_INLINE inline
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#define RATE 136
#define ROUNDS 24
#define MAX_DIGITS 32

static const uint64_t RC[24] = {
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808aULL,
    0x8000000080008000ULL,
    0x000000000000808bULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008aULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000aULL,
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL,
};

static FORCE_INLINE void keccak_f(uint64_t *s)
{
  uint64_t bc[5], t, p[25];
  for (int r = 0; r < ROUNDS; r++)
  {
    bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

    t = bc[4] ^ ROTL64(bc[1], 1);
    s[0] ^= t;
    s[5] ^= t;
    s[10] ^= t;
    s[15] ^= t;
    s[20] ^= t;
    t = bc[0] ^ ROTL64(bc[2], 1);
    s[1] ^= t;
    s[6] ^= t;
    s[11] ^= t;
    s[16] ^= t;
    s[21] ^= t;
    t = bc[1] ^ ROTL64(bc[3], 1);
    s[2] ^= t;
    s[7] ^= t;
    s[12] ^= t;
    s[17] ^= t;
    s[22] ^= t;
    t = bc[2] ^ ROTL64(bc[4], 1);
    s[3] ^= t;
    s[8] ^= t;
    s[13] ^= t;
    s[18] ^= t;
    s[23] ^= t;
    t = bc[3] ^ ROTL64(bc[0], 1);
    s[4] ^= t;
    s[9] ^= t;
    s[14] ^= t;
    s[19] ^= t;
    s[24] ^= t;

    p[0] = s[0];
    p[10] = ROTL64(s[1], 1);
    p[20] = ROTL64(s[2], 62);
    p[5] = ROTL64(s[3], 28);
    p[15] = ROTL64(s[4], 27);
    p[16] = ROTL64(s[5], 36);
    p[1] = ROTL64(s[6], 44);
    p[11] = ROTL64(s[7], 6);
    p[21] = ROTL64(s[8], 55);
    p[6] = ROTL64(s[9], 20);
    p[7] = ROTL64(s[10], 3);
    p[17] = ROTL64(s[11], 10);
    p[2] = ROTL64(s[12], 43);
    p[12] = ROTL64(s[13], 25);
    p[22] = ROTL64(s[14], 39);
    p[23] = ROTL64(s[15], 41);
    p[8] = ROTL64(s[16], 45);
    p[18] = ROTL64(s[17], 15);
    p[3] = ROTL64(s[18], 21);
    p[13] = ROTL64(s[19], 8);
    p[14] = ROTL64(s[20], 18);
    p[24] = ROTL64(s[21], 2);
    p[9] = ROTL64(s[22], 61);
    p[19] = ROTL64(s[23], 56);
    p[4] = ROTL64(s[24], 14);

    s[0] = p[0] ^ ((~p[1]) & p[2]);
    s[1] = p[1] ^ ((~p[2]) & p[3]);
    s[2] = p[2] ^ ((~p[3]) & p[4]);
    s[3] = p[3] ^ ((~p[4]) & p[0]);
    s[4] = p[4] ^ ((~p[0]) & p[1]);

    s[5] = p[5] ^ ((~p[6]) & p[7]);
    s[6] = p[6] ^ ((~p[7]) & p[8]);
    s[7] = p[7] ^ ((~p[8]) & p[9]);
    s[8] = p[8] ^ ((~p[9]) & p[5]);
    s[9] = p[9] ^ ((~p[5]) & p[6]);

    s[10] = p[10] ^ ((~p[11]) & p[12]);
    s[11] = p[11] ^ ((~p[12]) & p[13]);
    s[12] = p[12] ^ ((~p[13]) & p[14]);
    s[13] = p[13] ^ ((~p[14]) & p[10]);
    s[14] = p[14] ^ ((~p[10]) & p[11]);

    s[15] = p[15] ^ ((~p[16]) & p[17]);
    s[16] = p[16] ^ ((~p[17]) & p[18]);
    s[17] = p[17] ^ ((~p[18]) & p[19]);
    s[18] = p[18] ^ ((~p[19]) & p[15]);
    s[19] = p[19] ^ ((~p[15]) & p[16]);

    s[20] = p[20] ^ ((~p[21]) & p[22]);
    s[21] = p[21] ^ ((~p[22]) & p[23]);
    s[22] = p[22] ^ ((~p[23]) & p[24]);
    s[23] = p[23] ^ ((~p[24]) & p[20]);
    s[24] = p[24] ^ ((~p[20]) & p[21]);

    s[0] ^= RC[r];
  }
}

static void absorb_prefix(uint64_t st[25], const uint8_t *prefix, size_t len)
{
  memset(st, 0, 25 * sizeof(uint64_t));
  size_t off = 0;
  while (len - off >= RATE)
  {
    for (size_t i = 0; i < RATE; i += 8)
    {
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

static FORCE_INLINE int to_digits(uint64_t v, char *buf)
{
  char tmp[MAX_DIGITS + 1];
  int n = 0;
  do
  {
    if (n >= MAX_DIGITS)
      break;
    tmp[n++] = (char)('0' + (int)(v % 10));
    v /= 10;
  } while (v > 0);
  for (int i = 0; i < n; i++)
    buf[i] = tmp[n - 1 - i];
  buf[n] = '\0';
  return n;
}

static FORCE_INLINE void inc_digits(char *buf, int *dlen)
{
  int i = *dlen - 1;
  while (i >= 0 && buf[i] == '9')
  {
    buf[i] = '0';
    i--;
  }
  if (i < 0)
  {
    buf[0] = '1';
    for (int j = 1; j <= *dlen; j++)
      buf[j] = '0';
    (*dlen)++;
    buf[*dlen] = '\0';
  }
  else
  {
    buf[i]++;
  }
}

static FORCE_INLINE void prepare_template(uint64_t template_st[25],
                                          const uint64_t base[25], size_t off0,
                                          int dlen)
{
  memcpy(template_st, base, 25 * sizeof(uint64_t));
  size_t pad_off = off0 + (size_t)dlen;
  template_st[pad_off >> 3] ^= (uint64_t)0x06 << (8 * (pad_off & 7));
  template_st[16] ^= (uint64_t)0x80 << 56;
}

static FORCE_INLINE int check_counter_fast(const uint64_t template_st[25],
                                           size_t off0, const char *digits,
                                           int dlen,
                                           const uint64_t target64[4])
{
  uint64_t st[25];
  memcpy(st, template_st, sizeof(st));
  for (int i = 0; i < dlen; i++)
    st[(off0 + (size_t)i) >> 3] ^= (uint64_t)(uint8_t)digits[i]
                                   << (8 * ((off0 + (size_t)i) & 7));
  keccak_f(st);
  return (st[0] == target64[0]) && (st[1] == target64[1]) &&
         (st[2] == target64[2]) && (st[3] == target64[3]);
}

static int check_counter_general(const uint64_t base[25], size_t off0,
                                 const char *digits, int dlen,
                                 const uint64_t target64[4])
{
  uint64_t st[25];
  memcpy(st, base, sizeof(st));
  size_t off = off0;
  for (int i = 0; i < dlen; i++)
  {
    st[off >> 3] ^= (uint64_t)(uint8_t)digits[i] << (8 * (off & 7));
    off++;
    if (off == RATE)
    {
      keccak_f(st);
      off = 0;
    }
  }
  st[off >> 3] ^= (uint64_t)0x06 << (8 * (off & 7));
  st[16] ^= (uint64_t)0x80 << 56;
  keccak_f(st);
  return (st[0] == target64[0]) && (st[1] == target64[1]) &&
         (st[2] == target64[2]) && (st[3] == target64[3]);
}

#if defined(_WIN32)
static volatile LONG64 g_min_found = -1;

static FORCE_INLINE uint64_t get_min_found(void)
{
  LONG64 v = InterlockedCompareExchange64(&g_min_found, 0, 0);
  return v < 0 ? UINT64_MAX : (uint64_t)v;
}

static FORCE_INLINE void update_min_found(uint64_t val)
{
  while (1)
  {
    LONG64 cur = InterlockedCompareExchange64(&g_min_found, 0, 0);
    if (cur >= 0 && (uint64_t)cur <= val)
      break;
    if (InterlockedCompareExchange64(&g_min_found, (LONG64)val, cur) == cur)
      break;
  }
}
#else
static _Atomic uint64_t g_min_found = UINT64_MAX;

static FORCE_INLINE uint64_t get_min_found(void)
{
  return atomic_load_explicit(&g_min_found, memory_order_relaxed);
}

static FORCE_INLINE void update_min_found(uint64_t val)
{
  uint64_t cur = atomic_load_explicit(&g_min_found, memory_order_relaxed);
  while (val < cur &&
         !atomic_compare_exchange_weak_explicit(&g_min_found, &cur, val,
                                                memory_order_relaxed,
                                                memory_order_relaxed))
    ;
}
#endif

typedef struct
{
  const uint64_t *base;
  size_t off0;
  const uint64_t *target64;
  uint64_t start;
  uint64_t end;
  uint64_t result;
} WorkerArgs;

static void run_worker(WorkerArgs *a)
{
  a->result = UINT64_MAX;
  if (a->start >= a->end)
    return;

  char digits[MAX_DIGITS + 1];
  int dlen = to_digits(a->start, digits);
  uint64_t next_dlen_c = 1;
  for (int i = 0; i < dlen; i++)
  {
    if (next_dlen_c > UINT64_MAX / 10)
    {
      next_dlen_c = UINT64_MAX;
      break;
    }
    next_dlen_c *= 10;
  }

  uint64_t template_st[25];
  int can_fast = 0;
  if (a->off0 + (size_t)dlen + 1 <= RATE)
  {
    prepare_template(template_st, a->base, a->off0, dlen);
    can_fast = 1;
  }

  for (uint64_t c = a->start; c < a->end; c++)
  {
    if ((c & 0x3ff) == 0 && c >= get_min_found())
      break;

    if (c == next_dlen_c)
    {
      next_dlen_c = (next_dlen_c > UINT64_MAX / 10) ? UINT64_MAX
                                                    : next_dlen_c * 10;
      if (a->off0 + (size_t)dlen + 1 <= RATE)
      {
        prepare_template(template_st, a->base, a->off0, dlen);
        can_fast = 1;
      }
      else
      {
        can_fast = 0;
      }
    }

    int match = can_fast ? check_counter_fast(template_st, a->off0, digits,
                                              dlen, a->target64)
                         : check_counter_general(a->base, a->off0, digits, dlen,
                                                 a->target64);

    if (match)
    {
      a->result = c;
      update_min_found(c);
      break;
    }
    inc_digits(digits, &dlen);
  }
}

#if defined(_WIN32)
static DWORD WINAPI worker(LPVOID arg)
{
  run_worker((WorkerArgs *)arg);
  return 0;
}
#else
static void *worker(void *arg)
{
  run_worker((WorkerArgs *)arg);
  return NULL;
}
#endif

static int detect_threads(void)
{
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

static int hex_to_bytes(const char *hex, uint8_t *out, size_t max_out)
{
  size_t n = strlen(hex);
  if (n % 2 || (n / 2) > max_out)
    return -1;
  for (size_t i = 0; i < n; i += 2)
  {
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
                                 size_t bufsz)
{
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char *p = json;
  while ((p = strstr(p, pat)) != NULL)
  {
    const char *k = p + strlen(pat);
    while (*k == ' ' || *k == '\t' || *k == '\r' || *k == '\n')
      k++;
    if (*k == ':')
    {
      k++;
      while (*k == ' ' || *k == '\t' || *k == '\r' || *k == '\n')
        k++;
      if (*k == '"')
      {
        k++;
        size_t i = 0;
        while (*k && *k != '"' && i + 1 < bufsz)
        {
          if (*k == '\\' && *(k + 1))
            k++;
          buf[i++] = *k++;
        }
        buf[i] = '\0';
        return buf;
      }
    }
    p += strlen(pat);
  }
  return NULL;
}

static long long find_json_ll(const char *json, const char *key)
{
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char *p = json;
  while ((p = strstr(p, pat)) != NULL)
  {
    const char *k = p + strlen(pat);
    while (*k == ' ' || *k == '\t' || *k == '\r' || *k == '\n')
      k++;
    if (*k == ':')
    {
      k++;
      while (*k == ' ' || *k == '\t' || *k == '\r' || *k == '\n')
        k++;
      return strtoll(k, NULL, 10);
    }
    p += strlen(pat);
  }
  return -1;
}

int main(void)
{
  char input[65536];
  size_t total = 0;
  while (total < sizeof(input) - 1)
  {
    size_t n = fread(input + total, 1, sizeof(input) - 1 - total, stdin);
    if (n == 0)
      break;
    total += n;
  }
  input[total] = '\0';

  char challenge[128] = {0}, salt[16384] = {0};
  if (!find_json_str(input, "challenge", challenge, sizeof(challenge)) ||
      !find_json_str(input, "salt", salt, sizeof(salt)))
  {
    puts("{\"error\":\"missing challenge/salt\"}");
    return 1;
  }
  long long expire_at = find_json_ll(input, "expire_at");
  long long difficulty = find_json_ll(input, "difficulty");
  if (expire_at < 0 || difficulty <= 0)
  {
    puts("{\"error\":\"bad expire_at/difficulty\"}");
    return 1;
  }

  uint8_t target[32];
  if (hex_to_bytes(challenge, target, sizeof(target)) != 32)
  {
    puts("{\"error\":\"bad challenge hex\"}");
    return 1;
  }

  uint64_t target64[4];
  for (int i = 0; i < 4; i++)
  {
    uint64_t w = 0;
    for (int b = 0; b < 8; b++)
      w |= (uint64_t)target[i * 8 + b] << (8 * b);
    target64[i] = w;
  }

  char prefix[16500];
  int plen = snprintf(prefix, sizeof(prefix), "%s_%lld_", salt, expire_at);
  if (plen < 0 || (size_t)plen >= sizeof(prefix))
  {
    puts("{\"error\":\"salt too long\"}");
    return 1;
  }

  uint64_t base[25];
  absorb_prefix(base, (const uint8_t *)prefix, (size_t)plen);
  size_t off0 = (size_t)plen % RATE;

  uint64_t limit = (uint64_t)difficulty;
  if (limit == 0)
  {
    puts("{\"error\":\"answer not found in range\"}");
    return 1;
  }

  int nthreads = detect_threads();
  const char *env = getenv("POW_SOLVER_THREADS");
  if (env && env[0])
  {
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
  if (!args)
  {
    puts("{\"error\":\"out of memory\"}");
    return 1;
  }
#if defined(_WIN32)
  HANDLE *threads = (HANDLE *)calloc((size_t)nthreads, sizeof(HANDLE));
#else
  pthread_t *threads = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
#endif
  if (!threads)
  {
    free(args);
    puts("{\"error\":\"out of memory\"}");
    return 1;
  }

  uint64_t chunk = (limit + (uint64_t)nthreads - 1) / (uint64_t)nthreads;
  for (int i = 0; i < nthreads; i++)
  {
    args[i].base = base;
    args[i].off0 = off0;
    args[i].target64 = target64;
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
  for (int i = 0; i < nthreads; i++)
  {
    if (threads[i])
    {
      WaitForSingleObject(threads[i], INFINITE);
      CloseHandle(threads[i]);
    }
  }
#else
  for (int i = 0; i < nthreads; i++)
    pthread_join(threads[i], NULL);
#endif

  uint64_t best = get_min_found();

  free(threads);
  free(args);

  if (best != UINT64_MAX && best < limit)
  {
    printf("{\"answer\":%llu}\n", (unsigned long long)best);
    return 0;
  }
  puts("{\"error\":\"answer not found in range\"}");
  return 1;
}

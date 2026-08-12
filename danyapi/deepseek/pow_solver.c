#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RATE 136
#define ROUNDS 23

static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static const int ROT[5][5] = {
    { 0, 36, 3, 41, 18 },
    { 1, 44, 10, 45, 2 },
    { 62, 6, 43, 15, 61 },
    { 28, 55, 25, 21, 56 },
    { 27, 20, 39, 8, 14 },
};

static inline uint64_t rotl64(uint64_t x, int n) {
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

static void keccak_f(uint64_t *st) {
    uint64_t C[5], D[5], B[25];
    for (int round = 0; round < ROUNDS; round++) {
        for (int x = 0; x < 5; x++)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; x++)
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                st[x + 5 * y] ^= D[x];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(st[x + 5 * y], ROT[x][y]);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                st[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
        st[0] ^= RC[round + 1];
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

static int check_counter(const uint64_t base[25], size_t prefix_len, uint64_t c,
                         const uint8_t target[32]) {
    char digits[24];
    int dlen = snprintf(digits, sizeof(digits), "%llu", (unsigned long long)c);

    uint64_t st[25];
    memcpy(st, base, sizeof(st));

    size_t off = prefix_len % RATE;
    for (int i = 0; i < dlen; i++) {
        st[off / 8] ^= (uint64_t)(uint8_t)digits[i] << (8 * (off % 8));
        off++;
        if (off == RATE) {
            keccak_f(st);
            off = 0;
        }
    }

    st[off / 8] ^= (uint64_t)0x06 << (8 * (off % 8));
    off++;
    if (off == RATE) {
        keccak_f(st);
        off = 0;
    }
    st[16] ^= (uint64_t)0x80 << (8 * 7);

    keccak_f(st);

    for (int i = 0; i < 32; i++) {
        if (((st[i / 8] >> (8 * (i % 8))) & 0xff) != target[i])
            return 0;
    }
    return 1;
}

static int hex_to_bytes(const char *hex, uint8_t *out) {
    size_t n = strlen(hex);
    if (n % 2) return -1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hex[i], lo = hex[i + 1];
        int hv = (hi >= '0' && hi <= '9') ? hi - '0'
              : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10
              : (hi >= 'A' && hi <= 'F') ? hi - 'A' + 10 : -1;
        int lv = (lo >= '0' && lo <= '9') ? lo - '0'
              : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10
              : (lo >= 'A' && lo <= 'F') ? lo - 'A' + 10 : -1;
        if (hv < 0 || lv < 0) return -1;
        out[i / 2] = (uint8_t)((hv << 4) | lv);
    }
    return (int)(n / 2);
}

static const char *find_json_str(const char *json, const char *key, char *buf, size_t bufsz) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return NULL;
    p = strchr(p + strlen(pat), ':');
    if (!p) return NULL;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < bufsz) buf[i++] = *p++;
    buf[i] = '\0';
    return buf;
}

static long long find_json_ll(const char *json, const char *key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return -1;
    p = strchr(p + strlen(pat), ':');
    if (!p) return -1;
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

    uint64_t base[25];
    absorb_prefix(base, (const uint8_t *)prefix, (size_t)plen);

    long long limit = difficulty < 2000000000LL ? difficulty : 2000000000LL;
    for (long long c = 0; c < limit; c++) {
        if (check_counter(base, (size_t)plen, (uint64_t)c, target)) {
            printf("{\"answer\":%lld}\n", c);
            return 0;
        }
    }
    puts("{\"error\":\"answer not found in range\"}");
    return 1;
}

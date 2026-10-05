/* Definite CBOR subtree validation for discarded native tracking fields.
 * Never decodes numerical stage values. Bounds and nesting limits are checked
 * on every item. Return zero on malformed/unsupported/truncated input. */
#include <stddef.h>
#include <stdint.h>

static int item(const uint8_t *data, size_t len, size_t *pos, unsigned depth) {
    if (depth > 64 || *pos >= len) return 0;
    uint8_t head = data[(*pos)++];
    unsigned major = head >> 5, extra = head & 31;
    uint64_t value = extra;
    if (extra >= 24) {
        if (extra > 27) return 0;
        size_t width = (size_t)1 << (extra - 24);
        if (width > len - *pos) return 0;
        value = 0;
        for (size_t i = 0; i < width; ++i) value = (value << 8) | data[(*pos)++];
    }
    if (major < 2) return 1;
    if (major == 2 || major == 3) {
        if (value > len - *pos) return 0;
        *pos += (size_t)value;
        return 1;
    }
    if (major == 4 || major == 5) {
        if (major == 5) {
            if (value > UINT64_MAX / 2) return 0;
            value *= 2;
        }
        /* Each child needs at least its one-byte header. */
        if (value > len - *pos) return 0;
        for (uint64_t i = 0; i < value; ++i)
            if (!item(data, len, pos, depth + 1)) return 0;
        return 1;
    }
    if (major == 6) return item(data, len, pos, depth + 1);
    return extra == 20 || extra == 21 || extra == 22 ||
           extra == 25 || extra == 26 || extra == 27;
}

int skip_native_cbor(const uint8_t *data, size_t len, size_t *pos, unsigned depth) {
    if (!data || !pos || *pos > len) return 0;
    return item(data, len, pos, depth);
}

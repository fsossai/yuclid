#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint64_t next_random(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
}

static uint32_t reverse_low_bits(uint32_t value, unsigned bits) {
    uint32_t reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

static long peak_rss_kib(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return -1;
#if defined(__APPLE__)
    /* _POSIX_C_SOURCE exposes implementation-defined fields as ru_opaque. */
    return usage.ru_opaque[0] / 1024;
#else
    return usage.ru_maxrss;
#endif
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MIB PATTERN STRIDE SEED\n", argv[0]);
        return 2;
    }
    const size_t mib = (size_t)strtoull(argv[1], NULL, 10);
    const size_t stride = (size_t)strtoull(argv[3], NULL, 10);
    uint64_t state = strtoull(argv[4], NULL, 10);
    if (mib == 0 || mib > SIZE_MAX / (1024u * 1024u)) {
        fprintf(stderr, "invalid array size\n");
        return 2;
    }
    const size_t count = mib * 1024u * 1024u / sizeof(uint32_t);
    uint32_t *values = malloc(count * sizeof(*values));
    uint32_t *links = NULL;
    if (values == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    for (size_t index = 0; index < count; ++index)
        values[index] = (uint32_t)(index * 2654435761u);

    if (strcmp(argv[2], "pointer") == 0) {
        links = malloc(count * sizeof(*links));
        if (links == NULL) {
            fprintf(stderr, "link allocation failed\n");
            free(values);
            return 2;
        }
        unsigned bits = 0;
        while ((UINT64_C(1) << bits) < count)
            ++bits;
        if ((UINT64_C(1) << bits) != count || bits > 32) {
            fprintf(stderr, "pointer pattern requires a power-of-two element count\n");
            free(values); free(links);
            return 2;
        }
        for (uint32_t position = 0; position < count; ++position) {
            const uint32_t current = reverse_low_bits(position, bits);
            const uint32_t following =
                reverse_low_bits((position + 1u) % (uint32_t)count, bits);
            links[current] = following;
        }
    }

    volatile uint64_t checksum = 0;
    const double start = now_seconds();
    if (strcmp(argv[2], "linear") == 0) {
        for (unsigned pass = 0; pass < 8; ++pass)
            for (size_t index = 0; index < count; ++index)
                checksum += values[index];
    } else if (strcmp(argv[2], "strided") == 0 && stride > 0) {
        for (size_t offset = 0; offset < stride; ++offset)
            for (size_t index = offset; index < count; index += stride)
                checksum += values[index];
    } else if (strcmp(argv[2], "random") == 0) {
        for (size_t access = 0; access < count * 2; ++access)
            checksum += values[next_random(&state) % count];
    } else if (strcmp(argv[2], "pointer") == 0) {
        uint32_t index = 0;
        for (size_t access = 0; access < count; ++access) {
            index = links[index];
            checksum += values[index];
        }
    } else {
        fprintf(stderr, "unknown pattern or invalid stride\n");
        free(values); free(links);
        return 2;
    }
    const double elapsed = now_seconds() - start;
    printf("%.9f %ld %llu\n", elapsed, peak_rss_kib(),
           (unsigned long long)checksum);
    free(values);
    free(links);
    return 0;
}

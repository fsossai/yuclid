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

static void multiply_dot(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

static void multiply_rows(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            const double scale = a[i * n + k];
            for (size_t j = 0; j < n; ++j)
                c[i * n + j] += scale * b[k * n + j];
        }
}

static void multiply_columns(const double *a, const double *b, double *c, size_t n) {
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < n; ++k) {
            const double scale = b[k * n + j];
            for (size_t i = 0; i < n; ++i)
                c[i * n + j] += a[i * n + k] * scale;
        }
}

static size_t minimum(size_t left, size_t right) {
    return left < right ? left : right;
}

/* Each loop order below has a blocked counterpart: the same order, applied
   first to blocks and then within one block, so that the working set of the
   inner loops fits in cache. */

static void multiply_dot_tiled(
    const double *a, const double *b, double *c, size_t n, size_t tile
) {
    for (size_t ii = 0; ii < n; ii += tile)
        for (size_t jj = 0; jj < n; jj += tile)
            for (size_t kk = 0; kk < n; kk += tile)
                for (size_t i = ii; i < minimum(ii + tile, n); ++i)
                    for (size_t j = jj; j < minimum(jj + tile, n); ++j) {
                        double sum = 0.0;
                        for (size_t k = kk; k < minimum(kk + tile, n); ++k)
                            sum += a[i * n + k] * b[k * n + j];
                        c[i * n + j] += sum;
                    }
}

static void multiply_rows_tiled(
    const double *a, const double *b, double *c, size_t n, size_t tile
) {
    for (size_t ii = 0; ii < n; ii += tile)
        for (size_t kk = 0; kk < n; kk += tile)
            for (size_t jj = 0; jj < n; jj += tile)
                for (size_t i = ii; i < minimum(ii + tile, n); ++i)
                    for (size_t k = kk; k < minimum(kk + tile, n); ++k) {
                        const double scale = a[i * n + k];
                        for (size_t j = jj; j < minimum(jj + tile, n); ++j)
                            c[i * n + j] += scale * b[k * n + j];
                    }
}

static void multiply_columns_tiled(
    const double *a, const double *b, double *c, size_t n, size_t tile
) {
    for (size_t jj = 0; jj < n; jj += tile)
        for (size_t kk = 0; kk < n; kk += tile)
            for (size_t ii = 0; ii < n; ii += tile)
                for (size_t j = jj; j < minimum(jj + tile, n); ++j)
                    for (size_t k = kk; k < minimum(kk + tile, n); ++k) {
                        const double scale = b[k * n + j];
                        for (size_t i = ii; i < minimum(ii + tile, n); ++i)
                            c[i * n + j] += a[i * n + k] * scale;
                    }
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
    if (argc != 4) {
        fprintf(stderr, "usage: %s VARIANT INPUT TILE (TILE=0 disables blocking)\n", argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[2], "rb");
    if (input == NULL) {
        perror(argv[2]);
        return 2;
    }
    uint64_t encoded_size = 0;
    if (fread(&encoded_size, sizeof(encoded_size), 1, input) != 1) {
        fprintf(stderr, "cannot read matrix header\n");
        fclose(input);
        return 2;
    }
    const size_t n = (size_t)encoded_size;
    if (n == 0 || n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "invalid matrix size\n");
        fclose(input);
        return 2;
    }
    const size_t elements = n * n;
    double *a = malloc(elements * sizeof(*a));
    double *b = malloc(elements * sizeof(*b));
    double *c = calloc(elements, sizeof(*c));
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "allocation failed\n");
        fclose(input);
        free(a); free(b); free(c);
        return 2;
    }
    if (fread(a, sizeof(*a), elements, input) != elements ||
        fread(b, sizeof(*b), elements, input) != elements) {
        fprintf(stderr, "short matrix input\n");
        fclose(input);
        free(a); free(b); free(c);
        return 2;
    }
    fclose(input);

    /* A tile of 0 means no blocking at all. */
    const size_t tile = (size_t)strtoull(argv[3], NULL, 10);

    const double start = now_seconds();
    if (strcmp(argv[1], "dot") == 0)
        tile ? multiply_dot_tiled(a, b, c, n, tile) : multiply_dot(a, b, c, n);
    else if (strcmp(argv[1], "rows") == 0)
        tile ? multiply_rows_tiled(a, b, c, n, tile) : multiply_rows(a, b, c, n);
    else if (strcmp(argv[1], "columns") == 0)
        tile ? multiply_columns_tiled(a, b, c, n, tile)
             : multiply_columns(a, b, c, n);
    else {
        fprintf(stderr, "unknown variant: %s\n", argv[1]);
        free(a); free(b); free(c);
        return 2;
    }
    const double elapsed = now_seconds() - start;
    double checksum = 0.0;
    for (size_t index = 0; index < elements; ++index)
        checksum += c[index];
    printf("%.9f %.17g %ld\n", elapsed, checksum, peak_rss_kib());
    free(a); free(b); free(c);
    return 0;
}

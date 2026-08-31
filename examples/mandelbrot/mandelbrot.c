#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* One point of the plane: how many steps z = z^2 + c takes to leave the circle
   of radius two, up to a limit. A point inside the set never leaves and always
   costs the full limit; a point well outside costs a handful of steps. That
   difference between neighbours is the whole reason this program has a
   schedule to choose. */
static uint32_t escape(double cx, double cy, uint32_t limit) {
    double x = 0.0;
    double y = 0.0;
    uint32_t step = 0;
    while (step < limit && x * x + y * y <= 4.0) {
        const double next = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = next;
        ++step;
    }
    return step;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <width> <height> <max-iterations>\n", argv[0]);
        return 2;
    }
    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);
    const uint32_t limit = (uint32_t)strtoul(argv[3], NULL, 10);
    if (width <= 0 || height <= 0 || limit == 0) {
        fprintf(stderr, "width, height and max-iterations must be positive\n");
        return 2;
    }

    const int room = omp_get_max_threads();
    double *busy = calloc((size_t)room, sizeof(*busy));
    if (busy == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    /* The classic view. The set lies across the middle of it, so the rows
       through the middle cost the limit at nearly every pixel while the rows
       at the top and the bottom escape almost at once. Handing out equal
       numbers of rows therefore does not hand out equal amounts of work. */
    const double left = -2.0;
    const double right = 0.5;
    const double bottom = -1.25;
    const double top = 1.25;
    const double dx = (right - left) / width;
    const double dy = (top - bottom) / height;

    uint64_t total = 0;
    int used = 1;
    const double start = omp_get_wtime();
#pragma omp parallel
    {
        const int me = omp_get_thread_num();
        const double mine = omp_get_wtime();
        uint64_t local = 0;
        /* single rather than master, which OpenMP 5.1 deprecated, and nowait
           because there is nothing here for the others to wait on. */
#pragma omp single nowait
        used = omp_get_num_threads();
        /* schedule(runtime) defers the choice to OMP_SCHEDULE, so one binary
           covers every schedule in the space and the compilation does not
           have to be repeated for each of them. */
#pragma omp for schedule(runtime) nowait
        for (int row = 0; row < height; ++row) {
            const double cy = bottom + row * dy;
            for (int column = 0; column < width; ++column)
                local += escape(left + column * dx, cy, limit);
        }
        /* nowait above, so this is when the thread ran out of work rather
           than when the last of them did. The spread between the threads is
           what the schedule is being judged on. */
        if (me < room) busy[me] = omp_get_wtime() - mine;
#pragma omp atomic
        total += local;
    }
    const double elapsed = omp_get_wtime() - start;

    /* The busiest thread against the average one: 1.0 is a perfect division of
       the work, and 2.0 means the run waited about twice as long as it would
       have if every thread had finished together. */
    double most = 0.0;
    double sum = 0.0;
    const int counted = used < room ? used : room;
    for (int i = 0; i < counted; ++i) {
        if (busy[i] > most) most = busy[i];
        sum += busy[i];
    }
    const double mean = counted > 0 ? sum / counted : 0.0;
    const double imbalance = mean > 0.0 ? most / mean : 1.0;

    printf("%.9f %llu %d %.6f\n", elapsed, (unsigned long long)total, used,
           imbalance);
    free(busy);
    return 0;
}

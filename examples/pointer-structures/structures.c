/*
 * Three pointer-based containers over the same keys, and the operations one
 * plausibly performs on them.
 *
 * The structure is built once and the chosen operation is then repeated until
 * a time budget elapses. That is what makes the three comparable: a lookup in
 * a sorted list and a lookup in a hash table are orders of magnitude apart, so
 * a fixed number of probes would either take a second or be too short to
 * measure. Repeating to a budget also keeps every point about the same length.
 *
 * Everything measurable about the run is printed as `key value` lines, which
 * the yuclid metrics pick apart alongside whatever perf or strace printed.
 */

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

/* the same generator the other examples use, so a seed means one thing here */
static uint64_t next_random(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
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

/* One node type for all three containers: the same bytes moved through
 * different pointers, so what differs between them is the shape of the
 * traversal and not the size of the thing traversed. */
typedef struct node {
    uint64_t key;
    struct node *next;  /* list: successor. tree: left child. hash: chain. */
    struct node *right; /* tree only */
} node;

/* Keys are reached through pointers whatever the container, so the number of
 * nodes an operation touches is the thing to count: it is the algorithmic
 * cost, and it is what a cache-miss count has to be divided by to mean
 * anything. */
static uint64_t visits;

enum kind { LIST, TREE, HASH };

typedef struct {
    enum kind kind;
    node *head;          /* list and tree */
    node **buckets;      /* hash */
    size_t bucket_count;
    node *arena;         /* every node, so the container frees in one go */
    size_t used;
    size_t capacity;
} container;

static node *take(container *c) {
    if (c->used == c->capacity) {
        fprintf(stderr, "arena exhausted\n");
        exit(1);
    }
    return &c->arena[c->used++];
}

static size_t bucket_of(const container *c, uint64_t key) {
    /* a multiplicative hash: cheap, and it spreads sequential keys */
    return (size_t)((key * UINT64_C(11400714819323198485)) % c->bucket_count);
}

/* -- inserting ---------------------------------------------------------- */

static void insert(container *c, uint64_t key) {
    node **link;
    if (c->kind == HASH) {
        link = &c->buckets[bucket_of(c, key)];
    } else if (c->kind == LIST) {
        /* sorted, so a search may stop early: the position has to be found */
        link = &c->head;
        while (*link != NULL && (*link)->key < key) {
            ++visits;
            link = &(*link)->next;
        }
    } else {
        link = &c->head;
        while (*link != NULL) {
            ++visits;
            link = key < (*link)->key ? &(*link)->next : &(*link)->right;
        }
    }
    node *fresh = take(c);
    fresh->key = key;
    fresh->next = c->kind == TREE ? NULL : *link;
    fresh->right = NULL;
    *link = fresh;
}

/* Unlink a key that was only just inserted, which is why this is short: in a
 * tree such a node is still a leaf, so there is no successor to promote. It is
 * the delete half of `churn`, where a key is added and taken away again. */
static void unlink_key(container *c, uint64_t key) {
    node **link;
    if (c->kind == HASH)
        link = &c->buckets[bucket_of(c, key)];
    else
        link = &c->head;

    while (*link != NULL && (*link)->key != key) {
        ++visits;
        link = c->kind == TREE
                   ? (key < (*link)->key ? &(*link)->next : &(*link)->right)
                   : &(*link)->next;
    }
    if (*link == NULL)
        return;
    *link = c->kind == TREE ? NULL : (*link)->next;
    c->used -= 1; /* the node was the last one taken from the arena */
}

/* Even keys go in, so every odd key is known to be absent.
 *
 * A sorted list is built by appending, because inserting each key in turn
 * would be quadratic and this program has to build a hundred thousand of them.
 * The tree and the table are built from shuffled keys instead: in key order a
 * binary search tree degenerates into a linked list, which would be measuring
 * the wrong thing. */
static void build(container *c, size_t count, uint64_t seed) {
    c->used = 0;
    c->head = NULL;
    if (c->buckets != NULL)
        memset(c->buckets, 0, c->bucket_count * sizeof *c->buckets);

    if (c->kind == LIST) {
        node **tail = &c->head;
        for (size_t i = 0; i < count; ++i) {
            node *fresh = take(c);
            fresh->key = (uint64_t)i * 2;
            fresh->next = NULL;
            fresh->right = NULL;
            *tail = fresh;
            tail = &fresh->next;
        }
        return;
    }

    uint64_t state = seed;
    uint64_t *keys = malloc(count * sizeof *keys);
    if (keys == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (size_t i = 0; i < count; ++i)
        keys[i] = (uint64_t)i * 2;
    for (size_t i = count; i > 1; --i) {
        size_t j = (size_t)(next_random(&state) % i);
        uint64_t swap = keys[i - 1];
        keys[i - 1] = keys[j];
        keys[j] = swap;
    }
    for (size_t i = 0; i < count; ++i)
        insert(c, keys[i]);
    free(keys);
}

/* -- searching and walking ---------------------------------------------- */

static int find(const container *c, uint64_t key) {
    if (c->kind == LIST) {
        for (const node *at = c->head; at != NULL && at->key <= key; at = at->next) {
            ++visits;
            if (at->key == key)
                return 1;
        }
        return 0;
    }
    if (c->kind == TREE) {
        for (const node *at = c->head; at != NULL;) {
            ++visits;
            if (at->key == key)
                return 1;
            at = key < at->key ? at->next : at->right;
        }
        return 0;
    }
    for (const node *at = c->buckets[bucket_of(c, key)]; at != NULL; at = at->next) {
        ++visits;
        if (at->key == key)
            return 1;
    }
    return 0;
}

/* Every element once. A tree is walked with an explicit stack rather than
 * recursion, so that a million nodes does not need a million stack frames. */
static uint64_t walk(const container *c, const node **stack) {
    uint64_t seen = 0;
    if (c->kind == HASH) {
        for (size_t i = 0; i < c->bucket_count; ++i)
            for (const node *at = c->buckets[i]; at != NULL; at = at->next) {
                ++visits;
                seen += at->key;
            }
        return seen;
    }
    if (c->kind == LIST) {
        for (const node *at = c->head; at != NULL; at = at->next) {
            ++visits;
            seen += at->key;
        }
        return seen;
    }
    size_t depth = 0;
    const node *at = c->head;
    while (at != NULL || depth > 0) {
        while (at != NULL) {
            stack[depth++] = at;
            at = at->next;
        }
        at = stack[--depth];
        ++visits;
        seen += at->key;
        at = at->right;
    }
    return seen;
}

/* -- the operations ----------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <list|tree|hash> <operation> <nodes> <seed>\n",
                argv[0]);
        return 2;
    }
    const char *operation = argv[2];
    size_t count = (size_t)strtoull(argv[3], NULL, 10);
    uint64_t seed = strtoull(argv[4], NULL, 10);

    container c;
    memset(&c, 0, sizeof c);
    if (strcmp(argv[1], "list") == 0)
        c.kind = LIST;
    else if (strcmp(argv[1], "tree") == 0)
        c.kind = TREE;
    else if (strcmp(argv[1], "hash") == 0)
        c.kind = HASH;
    else {
        fprintf(stderr, "unknown structure: %s\n", argv[1]);
        return 2;
    }
    if (count == 0) {
        fprintf(stderr, "nodes must be positive\n");
        return 2;
    }

    const char *budget_text = getenv("BUDGET_MS");
    double budget = (budget_text != NULL ? atof(budget_text) : 300.0) / 1000.0;

    c.capacity = count + 1; /* churn holds one spare */
    c.arena = malloc(c.capacity * sizeof *c.arena);
    if (c.kind == HASH) {
        c.bucket_count = count / 4 + 1; /* chains of about four */
        c.buckets = malloc(c.bucket_count * sizeof *c.buckets);
    }
    /* Only a tree walk needs a stack, and only the tree allocates one: this
     * program's own footprint is a metric, so a buffer nothing uses would
     * show up as memory the container costs. */
    const node **stack = c.kind == TREE ? malloc((count + 1) * sizeof *stack) : NULL;
    if (c.arena == NULL || (c.kind == TREE && stack == NULL) ||
        (c.kind == HASH && c.buckets == NULL)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    double started = now_seconds();
    build(&c, count, seed);
    double build_seconds = now_seconds() - started;

    uint64_t state = seed ^ UINT64_C(0x9e3779b97f4a7c15);
    uint64_t sink = 0;
    uint64_t ops = 0;
    visits = 0;

    started = now_seconds();
    do {
        if (strcmp(operation, "build") == 0) {
            build(&c, count, seed);
        } else if (strcmp(operation, "lookup") == 0) {
            sink += (uint64_t)find(&c, (next_random(&state) % count) * 2);
        } else if (strcmp(operation, "absent") == 0) {
            sink += (uint64_t)find(&c, (next_random(&state) % count) * 2 + 1);
        } else if (strcmp(operation, "traverse") == 0) {
            sink += walk(&c, stack);
        } else if (strcmp(operation, "churn") == 0) {
            /* a key that is not there goes in and comes straight back out, so
             * the container is the same size at the end of every operation */
            uint64_t key = (next_random(&state) % count) * 2 + 1;
            insert(&c, key);
            unlink_key(&c, key);
            sink += 1;
        } else {
            fprintf(stderr, "unknown operation: %s\n", operation);
            return 2;
        }
        ++ops;
    } while (now_seconds() - started < budget);
    double op_seconds = now_seconds() - started;

    printf("build_seconds %.6f\n", build_seconds);
    printf("op_seconds %.6f\n", op_seconds);
    printf("ops %llu\n", (unsigned long long)ops);
    printf("visits %llu\n", (unsigned long long)visits);
    printf("nodes %llu\n", (unsigned long long)count);
    printf("peak_rss_kib %ld\n", peak_rss_kib());
    printf("checksum %llu\n", (unsigned long long)sink);

    free(stack);
    free(c.buckets);
    free(c.arena);
    return 0;
}

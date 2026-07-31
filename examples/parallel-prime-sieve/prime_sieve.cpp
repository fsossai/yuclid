#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

static std::vector<std::uint32_t> base_primes(std::uint64_t limit) {
    const auto root = static_cast<std::size_t>(std::sqrt(static_cast<long double>(limit)));
    std::vector<std::uint8_t> composite(root + 1, 0);
    std::vector<std::uint32_t> primes;
    for (std::size_t candidate = 2; candidate <= root; ++candidate) {
        if (composite[candidate])
            continue;
        primes.push_back(static_cast<std::uint32_t>(candidate));
        if (candidate <= root / candidate)
            for (std::size_t multiple = candidate * candidate; multiple <= root; multiple += candidate)
                composite[multiple] = 1;
    }
    return primes;
}

static std::uint64_t count_segment(
    std::uint64_t low,
    std::uint64_t high,
    const std::vector<std::uint32_t>& primes
) {
    std::vector<std::uint8_t> composite(static_cast<std::size_t>(high - low), 0);
    for (const std::uint64_t prime : primes) {
        if (prime * prime >= high && prime > high / prime)
            break;
        std::uint64_t first = ((low + prime - 1) / prime) * prime;
        first = std::max(first, prime * prime);
        for (std::uint64_t multiple = first; multiple < high; multiple += prime)
            composite[static_cast<std::size_t>(multiple - low)] = 1;
    }
    std::uint64_t count = 0;
    for (std::uint64_t value = low; value < high; ++value)
        if (value >= 2 && !composite[static_cast<std::size_t>(value - low)])
            ++count;
    return count;
}

static long peak_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return -1;
#if defined(__APPLE__)
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " LIMIT WORKERS static|dynamic SEGMENT_KIB\n";
        return 2;
    }
    const std::uint64_t limit = std::stoull(argv[1]);
    const unsigned workers = static_cast<unsigned>(std::stoul(argv[2]));
    const std::string schedule = argv[3];
    const std::uint64_t segment_size = std::stoull(argv[4]) * 1024;
    if (limit < 2 || workers == 0 || segment_size == 0 ||
        (schedule != "static" && schedule != "dynamic")) {
        std::cerr << "invalid argument\n";
        return 2;
    }

    const auto primes = base_primes(limit);
    const std::uint64_t segments = (limit + segment_size) / segment_size;
    std::atomic<std::uint64_t> next{0};
    std::atomic<std::uint64_t> total{0};
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            std::uint64_t local = 0;
            if (schedule == "dynamic") {
                while (true) {
                    const auto segment = next.fetch_add(1, std::memory_order_relaxed);
                    if (segment >= segments)
                        break;
                    const auto low = segment * segment_size;
                    const auto high = std::min(limit + 1, low + segment_size);
                    local += count_segment(low, high, primes);
                }
            } else {
                for (std::uint64_t segment = worker; segment < segments; segment += workers) {
                    const auto low = segment * segment_size;
                    const auto high = std::min(limit + 1, low + segment_size);
                    local += count_segment(low, high, primes);
                }
            }
            total.fetch_add(local, std::memory_order_relaxed);
        });
    }
    for (auto& thread : threads)
        thread.join();
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << elapsed.count() << ' '
              << static_cast<double>(limit - 1) / elapsed.count() << ' '
              << peak_rss_kib() << ' ' << total.load() << '\n';
    return 0;
}

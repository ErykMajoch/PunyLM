#include <catch2/catch_test_macros.hpp>
#include <punylm/core/rng.h>

#include <cmath>

TEST_CASE("RNG singleton returns the same instance", "[rng]") {
    auto &r1 = punylm::RNG::global();
    auto &r2 = punylm::RNG::global();
    CHECK(&r1 == &r2);
}

TEST_CASE("RNG produces deterministic sequence with fixed seed", "[rng]") {
    punylm::RNG::global().seed(42);
    float a1 = punylm::RNG::global().normal(0.0f, 1.0f);
    float a2 = punylm::RNG::global().normal(0.0f, 1.0f);
    float a3 = punylm::RNG::global().normal(0.0f, 1.0f);

    punylm::RNG::global().seed(42);
    float b1 = punylm::RNG::global().normal(0.0f, 1.0f);
    float b2 = punylm::RNG::global().normal(0.0f, 1.0f);
    float b3 = punylm::RNG::global().normal(0.0f, 1.0f);

    CHECK(a1 == b1);
    CHECK(a2 == b2);
    CHECK(a3 == b3);
}

TEST_CASE("Different seed produces different sequence", "[rng]") {
    punylm::RNG::global().seed(100);
    float v1 = punylm::RNG::global().normal(0.0f, 1.0f);

    punylm::RNG::global().seed(200);
    float v2 = punylm::RNG::global().normal(0.0f, 1.0f);

    CHECK(v1 != v2);
}

TEST_CASE("Normal distribution produces finite values", "[rng]") {
    punylm::RNG::global().seed(12345);

    float sum = 0.0f;
    constexpr int N = 1000;

    for (int i = 0; i < N; ++i) {
        float v = punylm::RNG::global().normal(0.0f, 1.0f);
        CHECK(std::isfinite(v));
        sum += v;
    }

    float mean = sum / static_cast<float>(N);
    CHECK(std::abs(mean) < 0.2f);
}

TEST_CASE("Uniform distribution respects bounds", "[rng]") {
    punylm::RNG::global().seed(54321);

    constexpr float low = -2.0f;
    constexpr float high = 3.0f;
    constexpr int n = 1000;

    for (int i = 0; i < n; ++i) {
        float v = punylm::RNG::global().uniform(low, high);
        CHECK(v >= low);
        CHECK(v < high);
    }
}

TEST_CASE("engine() returns underlying mt19937_64", "[rng]") {
    auto &rng = punylm::RNG::global();
    auto &eng = rng.engine();
    CHECK(&eng == &rng.engine());
}

#include <catch2/catch_test_macros.hpp>
#include <punylm/core/storage.h>

#include <type_traits>

TEST_CASE("Default storage is empty CPU storage", "[storage]") {
    punylm::Storage s;
    CHECK(s.size() == 0);
    CHECK(s.device() == punylm::Device::CPU);
    CHECK(s.empty());
    CHECK(s.data() == nullptr);
}

TEST_CASE("CPU storage allocation is zero-initialised", "[storage]") {
    punylm::Storage s(10, punylm::Device::CPU);
    CHECK(s.size() == 10);
    CHECK(!s.empty());
    REQUIRE(s.data() != nullptr);

    auto v = s.to_host();
    REQUIRE(v.size() == 10);
    for (auto x: v)
        CHECK(x == 0.0f);
}

TEST_CASE("from_host and to_host round-trip", "[storage]") {
    punylm::Storage s(4, punylm::Device::CPU);
    std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
    s.from_host(expected);
    auto result = s.to_host();
    REQUIRE(result == expected);
}

TEST_CASE("Zero resets all values", "[storage]") {
    punylm::Storage s(5, punylm::Device::CPU);
    s.from_host({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    s.zero();
    auto v = s.to_host();
    for (auto x: v)
        CHECK(x == 0.0f);
}

TEST_CASE("Move construction transfers ownership", "[storage]") {
    punylm::Storage s1(4, punylm::Device::CPU);
    s1.from_host({1.0f, 2.0f, 3.0f, 4.0f});

    punylm::Storage s2(std::move(s1));
    CHECK(s2.size() == 4);
    CHECK(s2.to_host() == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));

    CHECK(s1.empty());
    CHECK(s1.size() == 0);
    CHECK(s1.data() == nullptr);
}

TEST_CASE("Move assignment transfers ownership", "[storage]") {
    punylm::Storage s1(4, punylm::Device::CPU);
    s1.from_host({1.0f, 2.0f, 3.0f, 4.0f});

    punylm::Storage s2;
    s2 = std::move(s1);
    CHECK(s2.size() == 4);
    CHECK(s2.to_host() == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));

    CHECK(s1.empty());
    CHECK(s1.size() == 0);
    CHECK(s1.data() == nullptr);
}

TEST_CASE("Self-move assignment is safe", "[storage]") {
    punylm::Storage s(4, punylm::Device::CPU);
    s.from_host({1.0f, 2.0f, 3.0f, 4.0f});

    auto &r = s;
    s = std::move(r);
    CHECK(s.size() == 4);
    CHECK(s.to_host() == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST_CASE("Zero size storage is empty", "[storage]") {
    punylm::Storage s(0, punylm::Device::CPU);
    CHECK(s.empty());
    CHECK(s.size() == 0);
    CHECK(s.data() == nullptr);
    REQUIRE_NOTHROW(s.zero());
    REQUIRE_NOTHROW(s.to_host());
}

TEST_CASE("from_host throws on size mismatch", "[storage]") {
    punylm::Storage s(4, punylm::Device::CPU);
    REQUIRE_THROWS_AS(s.from_host({1.0f, 2.0f, 3.0f}), punylm::Error);
    REQUIRE_THROWS_AS(s.from_host({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}), punylm::Error);
}

TEST_CASE("Storage is move-only (copy deleted)", "[storage]") {
    CHECK(!std::is_copy_constructible_v<punylm::Storage>);
    CHECK(!std::is_copy_assignable_v<punylm::Storage>);
    CHECK(std::is_move_constructible_v<punylm::Storage>);
    CHECK(std::is_move_assignable_v<punylm::Storage>);
}

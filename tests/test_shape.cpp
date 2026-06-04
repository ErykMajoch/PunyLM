#include <catch2/catch_test_macros.hpp>
#include <punylm/core/shape.h>

TEST_CASE("Numel computes product of dimensions", "[shape]") {
    CHECK(punylm::numel({}) == 1);
    CHECK(punylm::numel({5}) == 5);
    CHECK(punylm::numel({2, 3}) == 6);
    CHECK(punylm::numel({4, 3, 2}) == 24);
    CHECK(punylm::numel({1, 1, 1}) == 1);
    CHECK(punylm::numel({10, 10}) == 100);
}

TEST_CASE("last_dim returns last dimension or 1 for empty shape", "[shape]") {
    CHECK(punylm::last_dim({}) == 1);
    CHECK(punylm::last_dim({5}) == 5);
    CHECK(punylm::last_dim({2, 3}) == 3);
    CHECK(punylm::last_dim({4, 3, 2}) == 2);
    CHECK(punylm::last_dim({10}) == 10);
}

TEST_CASE("rows_of computes numel / last_dim", "[shape]") {
    CHECK(punylm::rows_of({}) == 1);
    CHECK(punylm::rows_of({5}) == 5);
    CHECK(punylm::rows_of({2, 3}) == 2);
    CHECK(punylm::rows_of({4, 3, 2}) == 12);
    CHECK(punylm::rows_of({6, 1}) == 6);
}

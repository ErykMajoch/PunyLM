#include <catch2/catch_test_macros.hpp>
#include <punylm/core/tensor.h>

TEST_CASE("Default constructed tensor is undefined", "[tensor]") {
    punylm::Tensor t;
    CHECK_FALSE(t.defined());
}

TEST_CASE("Zeros factory creates zero-filled tensor with correct shape", "[tensor]") {
    auto t = punylm::Tensor::zeros({2, 3});
    REQUIRE(t.defined());
    CHECK(t.shape() == punylm::Shape({2, 3}));
    CHECK(t.numel() == 6);
    CHECK(t.device() == punylm::Device::CPU);

    auto v = t.to_host();
    REQUIRE(v.size() == 6);
    for (auto x: v)
        CHECK(x == 0.0f);
}

TEST_CASE("Zeros factory with requires_grad", "[tensor]") {
    auto t = punylm::Tensor::zeros({3}, punylm::Device::CPU, true);
    CHECK(t.requires_grad());

    auto t2 = punylm::Tensor::zeros({3}, punylm::Device::CPU, false);
    CHECK_FALSE(t2.requires_grad());
}

TEST_CASE("from_host factory initialises from vector", "[tensor]") {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    auto t = punylm::Tensor::from_host({2, 2}, data);
    CHECK(t.shape() == punylm::Shape({2, 2}));
    CHECK(t.numel() == 4);

    auto v = t.to_host();
    CHECK(v == data);
}

TEST_CASE("from_host with requires_grad", "[tensor]") {
    auto t = punylm::Tensor::from_host({3}, {1.0f, 2.0f, 3.0f}, punylm::Device::CPU, true);
    CHECK(t.requires_grad());
    CHECK(t.to_host() == std::vector<float>({1.0f, 2.0f, 3.0f}));
}

TEST_CASE("randn factory creates tensor with correct shape and grad flag", "[tensor]") {
    auto t = punylm::Tensor::randn({4, 2}, 1.0f);
    CHECK(t.shape() == punylm::Shape({4, 2}));
    CHECK(t.numel() == 8);
    CHECK(t.requires_grad());

    auto v = t.to_host();
    REQUIRE(v.size() == 8);
}

TEST_CASE("Backward seeds gradient with ones on leaf tensor", "[tensor]") {
    auto t = punylm::Tensor::from_host({3}, {1.0f, 2.0f, 3.0f}, punylm::Device::CPU, true);
    t.backward();
    auto g = t.grad_to_host();
    REQUIRE(g.size() == 3);
    CHECK(g == std::vector<float>({1.0f, 1.0f, 1.0f}));
}

TEST_CASE("zero_grad resets gradient to zeros", "[tensor]") {
    auto t = punylm::Tensor::from_host({2}, {1.0f, 2.0f}, punylm::Device::CPU, true);
    t.backward();
    t.zero_grad();
    auto g = t.grad_to_host();
    REQUIRE(g.size() == 2);
    for (auto x: g)
        CHECK(x == 0.0f);
}

TEST_CASE("Grad is lazily allocated", "[tensor]") {
    auto t = punylm::Tensor::zeros({5}, punylm::Device::CPU, true);
    auto g = t.grad();
    REQUIRE(g != nullptr);
}

TEST_CASE("Tensor data accessor returns writable pointer", "[tensor]") {
    auto t = punylm::Tensor::zeros({3}, punylm::Device::CPU, false);
    float *ptr = t.data();
    REQUIRE(ptr != nullptr);
    ptr[0] = 42.0f;
    ptr[1] = 43.0f;
    ptr[2] = 44.0f;
    CHECK(t.to_host() == std::vector<float>({42.0f, 43.0f, 44.0f}));
}

TEST_CASE("Tensor device is CPU by default", "[tensor]") {
    auto t = punylm::Tensor::zeros({1, 1});
    CHECK(t.device() == punylm::Device::CPU);
}

#include <catch2/catch_test_macros.hpp>
#include <punylm/core/device.h>

TEST_CASE("Device enum values", "[device]") {
    REQUIRE(static_cast<int>(punylm::Device::CPU) == 0);
    REQUIRE(static_cast<int>(punylm::Device::CUDA) == 1);
}

TEST_CASE("device_name returns correct strings", "[device]") {
    CHECK(std::string(punylm::device_name(punylm::Device::CPU)) == "CPU");
    CHECK(std::string(punylm::device_name(punylm::Device::CUDA)) == "CUDA");
}

TEST_CASE("Error inherits from std::runtime_error", "[device]") {
    punylm::Error e("test message");
    REQUIRE(std::string(e.what()).find("test message") != std::string::npos);
    REQUIRE(dynamic_cast<const std::runtime_error*>(&e) != nullptr);
}

TEST_CASE("PUNYLM_CHECK does not throw when condition is true", "[device]") {
    REQUIRE_NOTHROW(PUNYLM_CHECK(true, "should not throw"));
}

TEST_CASE("PUNYLM_CHECK throws punylm::Error when condition is false", "[device]") {
    REQUIRE_THROWS_AS(PUNYLM_CHECK(false, "failure occurred"), punylm::Error);
}

TEST_CASE("PUNYLM_CHECK error message includes PunyLM prefix and user message", "[device]") {
    constexpr auto error_string = "My Error!";
    try {
        PUNYLM_CHECK(false, error_string);
    } catch (const punylm::Error &e) {
        std::string msg(e.what());
        CHECK(msg.find("PunyLM: ") == 0);
        CHECK(msg.find(error_string) != std::string::npos);
    }
}

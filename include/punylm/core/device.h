#pragma once
#include <stdexcept>
#include <string>

#define PUNYLM_CHECK(condition, message) [[unlikely]] if (!(condition)) throw ::punylm::Error(std::string("PunyLM: ") + (message));

namespace punylm {

    enum class Device { CPU, CUDA };

    inline const char* device_name(const Device device) {
        return device == Device::CPU ? "CPU" : "CUDA";
    }

    struct Error : std::runtime_error {
        explicit Error(const std::string& message) : std::runtime_error(message) {}
    };

} // namespace punylm


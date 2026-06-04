#include <punylm/core/device.h>

#include <iostream>

using enum punylm::Device;

int main() {
    constexpr auto device = CPU;
    std::cout << "Hello World! Device: " << punylm::device_name(device) << '\n';
    PUNYLM_CHECK(0, "Hello World!");
}

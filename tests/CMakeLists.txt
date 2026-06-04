include(Catch)

add_executable(punylm_tests
        test_device.cpp
        test_shape.cpp
        test_storage.cpp
        test_rng.cpp
        test_tensor.cpp)
target_link_libraries(punylm_tests PRIVATE punylm::punylm Catch2::Catch2WithMain)
punylm_set_warnings(punylm_tests)
catch_discover_tests(punylm_tests)
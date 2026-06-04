include(Catch)

add_executable(punylm_tests
    test_main.cpp)
target_link_libraries(punylm_tests PRIVATE punylm::punylm Catch2::Catch2WithMain)
punylm_set_warnings(punylm_tests)
catch_discover_tests(punylm_tests)
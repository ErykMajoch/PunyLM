set(PUNYLM_LIB_SOURCES
    tokeniser.cpp
)

add_library(punylm STATIC ${PUNYLM_LIB_SOURCES})

target_include_directories(punylm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(punylm
    PUBLIC
        Eigen3::Eigen
)

target_compile_features(punylm PUBLIC cxx_std_20)

add_library(PunyLM::punylm ALIAS punylm)

add_executable(punylm_run main.cpp)
target_link_libraries(punylm_run PRIVATE punylm Eigen3::Eigen)
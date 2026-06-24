cmake_minimum_required(VERSION 3.20)

project(genetic_algo
    VERSION 1.0
    LANGUAGES CXX
)

# ----------------------------------
# C++ Standard
# ----------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------
# Threads (required for std::thread)
# ----------------------------------
find_package(Threads REQUIRED)

# ----------------------------------
# neural_network library
# ----------------------------------

# Since implementation is in headers/tpp files,
# INTERFACE is usually the best choice.
add_library(neural_network INTERFACE)

target_include_directories(neural_network
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

# ----------------------------------
# problems library
# ----------------------------------
add_library(problems
    src/problems/price_direction.cpp
)

target_include_directories(problems
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# ----------------------------------
# ga library
# thread_pool.cpp is the only compiled source; the rest is header/tpp only.
# ----------------------------------
add_library(ga
    src/ga/thread_pool.cpp
)

target_include_directories(ga
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ga
    PUBLIC
        Threads::Threads
)

# ----------------------------------
# Executable
# ----------------------------------
add_executable(genetic_algo
    src/main.cpp
)

target_link_libraries(genetic_algo
    PRIVATE
        neural_network
        problems
        ga
)

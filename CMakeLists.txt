cmake_minimum_required(VERSION 3.20)

project(DFT
    VERSION 1.0
    LANGUAGES CXX
)

# ============================================================
# C++ STANDARD
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# SOURCE FILES
# ============================================================

file(GLOB_RECURSE DFT_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# ============================================================
# DFT
# ============================================================

add_executable(DFT
    ${DFT_SOURCES}
)

# ============================================================
# INCLUDE DIRECTORIES
# ============================================================

file(GLOB_RECURSE DFT_INCLUDE_DIRS
    LIST_DIRECTORIES true
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
)

target_include_directories(DFT PRIVATE
    src
    ${DFT_INCLUDE_DIRS}
)
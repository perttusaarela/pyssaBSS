cmake_minimum_required(VERSION 3.15)
project(joint_diag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    src/pyssaBSS/joint_diag/wrapper.cpp
    src/pyssaBSS/joint_diag/cJADE.cpp
)

install(TARGETS _core
        LIBRARY DESTINATION pyssaBSS/joint_diag)
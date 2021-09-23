cmake_minimum_required(VERSION 3.18)
project(fastfilters2)

find_package(Halide REQUIRED)
find_package(Python REQUIRED COMPONENTS Development.Module)

add_executable(gen gen.cpp)
target_link_libraries(gen PRIVATE Halide::Generator)
set_target_properties(gen PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_halide_library(filters2d FROM gen)

Python_add_library(_fastfilters2 MODULE WITH_SOABI fastfilters2.c)
target_link_libraries(_fastfilters2 PRIVATE Halide::Runtime filters2d)
set_target_properties(_fastfilters2 PROPERTIES C_STANDARD 90 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

install(TARGETS _fastfilters2 DESTINATION ${CMAKE_CURRENT_LIST_DIR}/src)

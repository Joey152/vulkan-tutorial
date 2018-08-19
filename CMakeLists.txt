cmake_minimum_required(VERSION 3.10)
project(vulkan-tutorial)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE sources src/*.cpp)

add_executable(vulkan-tutorial ${sources})

find_package(Vulkan REQUIRED)
target_include_directories(vulkan-tutorial PRIVATE ${Vulkan_INCLUDE_DIR})
target_link_libraries(vulkan-tutorial ${Vulkan_LIBRARY})

find_package(glfw3 REQUIRED)
target_link_libraries(vulkan-tutorial glfw)

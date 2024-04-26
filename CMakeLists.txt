cmake_minimum_required(VERSION 3.8)
project(SimpleMonteCarloComputations LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall")

set(SRC_LIST main.cc)

add_executable(${PROJECT_NAME} ${SRC_LIST})

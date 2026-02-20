cmake_minimum_required(VERSION 3.22)
project(SwarmAutonomySimulator)

# Include GTest
# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/6ec14dfd8c409d05fba94e18e3a02df35b874353.zip
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

# dependency files and directories
set(DEPENDENCY_INCLUDES
  ../data-collector/include
  ../data-collector/third-party
  ../trajectory-generator/include
  ../trajectory-generator/third-party
)
set(DEPENDENCIES
  # GTest::gtest
  ${CMAKE_SOURCE_DIR}/../data-collector/build/dataCollector.lib
  ${CMAKE_SOURCE_DIR}/../trajectory-generator/build/trajectoryGenerator.lib
)

# Make shared library
set(SOURCES
    ./src/agent.cpp
    ./src/controller.cpp
)

add_library(simulator STATIC ${SOURCES})
target_include_directories(simulator 
  PUBLIC 
    ./include
    ${DEPENDENCY_INCLUDES}
)
target_link_libraries(simulator 
  PUBLIC 
    ${DEPENDENCIES}
)

# Make and link simulation executable
add_executable(all_states_known ./src/all_states_known.cpp)
target_include_directories(all_states_known PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(all_states_known PUBLIC simulator ${DEPENDENCIES})
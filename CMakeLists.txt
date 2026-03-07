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

find_package(Eigen3 REQUIRED NO_MODULE)

# dependency files and directories
set(DEPENDENCY_INCLUDES
  ../controller/include
  ../data-collector/include
  ../data-collector/third-party
  ../trajectory-generator/include
  ../trajectory-generator/third-party
  ../state-estimator/include
)
set(DEPENDENCIES
  # GTest::gtest
  Eigen3::Eigen
  ${CMAKE_SOURCE_DIR}/../data-collector/build/libdataCollector.a
  ${CMAKE_SOURCE_DIR}/../trajectory-generator/build/libtrajectoryGenerator.a
  ${CMAKE_SOURCE_DIR}/../state-estimator/build/libstateEstimator.a
)

# Make shared library
set(SOURCES
    ./src/agent.cpp
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

# Make and link simulation executable
add_executable(one_smart ./src/one_smart.cpp)
target_include_directories(one_smart PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(one_smart PUBLIC simulator ${DEPENDENCIES})

# Make and link controller execulatble
add_executable(follow ./src/follow.cpp)
target_include_directories(follow PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(follow PUBLIC simulator ${DEPENDENCIES})

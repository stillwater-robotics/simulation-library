cmake_minimum_required(VERSION 3.22)
project(SwarmAutonomySimulator)

# Include GTest
# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# dependency files and directories
set(DEPENDENCY_INCLUDES
  ../controller/include
  ../data-collector/include
  ../data-collector/third-party
  ../trajectory-generator/include
  ../trajectory-generator/third-party
)
set(DEPENDENCIES
  ${CMAKE_SOURCE_DIR}/../data-collector/build/libdataCollector.so
  ${CMAKE_SOURCE_DIR}/../trajectory-generator/build/libtrajectoryGenerator.so
)

# Make shared library
set(SOURCES
    ./src/agent.cpp
)

add_library(simulator SHARED ${SOURCES})
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
add_executable(all_neighbours_ground_truth_known ./src/all_neighbours_ground_truth_known.cpp)
target_include_directories(all_neighbours_ground_truth_known PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(all_neighbours_ground_truth_known PUBLIC simulator ${DEPENDENCIES})

# Make and link simulation executable
add_executable(one_smart_ground_truth ./src/one_smart_ground_truth.cpp)
target_include_directories(one_smart_ground_truth PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(one_smart_ground_truth PUBLIC simulator ${DEPENDENCIES})

# Make and link controller execulatble
add_executable(follow ./src/follow.cpp)
target_include_directories(follow PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries(follow PUBLIC simulator ${DEPENDENCIES})

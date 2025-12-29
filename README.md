# swarm-autonomy-simulation
This repository contains code for simulating the MCU autonomy related software. It includes C++ code to run simulations and python code to visualize results.

## include and src
C++ code to run simulations.
- agent.h/cpp: simulates a complete swarm agent, each instance is given an ID number that is only used by the simulation environement when saving data.
- controller.h/cpp: mock controller used by the agents.

After a simulation is complete the data will be saved to the following file structure:

```
sim_data_<unix_time>
  <swarm_id>_states.csv
  <swarm_id>_desired_poses.csv
  <swarm_id>_trajectories
    <swarm_id>_<time>.csv
    <swarm_id>_<time2>.csv
  <next_swarm_id>_states.csv
  <next_swarm_id>_desired_poses.csv
  <next_swarm_id>_trajectories
    <next_swarm_id>_<time>.csv
    <next_swarm_id>_<time2>.csv
  ...
```

Where `unix_time` is the time the simulation started at, `swarm_id` is the ID number of the agent that generated that data and `time`/`time2` is the time since simulation start the saved trajectory was generated at.

### Build Instructions (Tested on Linux)
1. Go to package root.
2. run `cmake -B build -G Ninja`
3. Go to the new `build` directory
4. run `ninja`


## sim_tools
Python code used to visulize simulation results.
- plot.py: code to plot trajectories


## state_estimator
- sensor_benchmarker.py: Computes the uncertainty of different sensor configurations for state estimation.
- sensor_specs.json: Specifications for various sensors used in the benchmarking.
- state_estimator.py: Simulates state estimation using an Extended Kalman Filter (EKF) and Rauch-Tung-Striebel (RTS) smoother.
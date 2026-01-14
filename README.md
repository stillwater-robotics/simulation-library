# swarm-autonomy-simulation

## cpp_state_estimator
State estimator class for Pico 2 W with example .ino usage file.

## py_state_estimator
Uses an old version of the vehicle model with thrust and sway propellors before we moved to the side-by-side thruster model.
- sensor_benchmarker.py: Computes the uncertainty of different sensor configurations for state estimation.
- sensor_specs.json: Specifications for various sensors used in the benchmarking.
- state_estimator.py: Simulates state estimation using an Extended Kalman Filter (EKF) and Rauch-Tung-Striebel (RTS) smoother.
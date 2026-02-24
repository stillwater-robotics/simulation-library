#include "agent.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <random>

// Helper function to convert Eigen state vector to State struct
State matrixToState(Eigen::Matrix<float, 7, 1> x){
    State state;
    
    state.pose.x = x(0);
    state.pose.y = x(1);
    state.pose.z = x(2);
    state.pose.theta = x(3);
    state.velocity.x = x(4);
    state.velocity.y = x(5);
    state.velocity.z = x(6);

    return state;
}

Agent::Agent(float id_, State initialState, uint32_t sim_start_, float speed, float dive_time, int replan_chance){
    pointGenerator = PointGenerator(5.0f, 3.0f, 0.5f);
    trajectoryGenerator = TrajectoryGenerator(speed, dive_time);
    controller = Controller();

    odds = replan_chance;
    id = id_;
    trueState = initialState;
    traj_gen_time = 0;
    traj_time = 0;
    desired = trueState.pose;

    sim_start = sim_start_;
    std::stringstream directory;
    directory << "sim_data_" << sim_start << "/";

    std::stringstream state_filename;
    state_filename << id << "_true_states.csv";
    true_state_writer = Writer(directory.str() + state_filename.str());

    std::stringstream est_state_filename;
    est_state_filename << id << "_estimated_states.csv";
    estimated_state_writer = Writer(directory.str() + est_state_filename.str());

    std::stringstream desired_filename;
    desired_filename << id << "_desired_poses.csv";
    desired_state_writer = Writer(directory.str() + desired_filename.str());
}

void Agent::Plan(std::vector<State> states, float time){
    traj_gen_time = time;

    int random_num = rand() % 101;
    if (random_num >= odds && traj_gen_time >= 0.01){
        return;
    }

    std::vector<Pose> poses;
    for(auto it=states.begin(); it < states.end(); ++it){
        poses.push_back((*it).pose);
    }
    desired = pointGenerator.Update(trueState.pose, poses);
    trajectoryGenerator.Update(trueState.pose, desired);

    traj_time = time;
    return;
}

void Agent::Update(float time){
    float dt = 0.1f; 
    float current_traj_time = time - traj_time;

    State desiredState = trajectoryGenerator.GetDesiredState(current_traj_time);
    float v_desired = sqrtf(desiredState.velocity.x * desiredState.velocity.x + 
                            desiredState.velocity.y * desiredState.velocity.y);

    // Since the controller isn't done yet, use inverse kinematics to determine approximate control inputs
    float target_vL = v_desired - (desiredState.velocity.theta * L_BASE / 2.0f);
    float target_vR = v_desired + (desiredState.velocity.theta * L_BASE / 2.0f);
    float accel_L = (target_vL - stateEstimator.x(IDX_VL)) / dt;
    float accel_R = (target_vR - stateEstimator.x(IDX_VR)) / dt;
  
    trueState = controller.Update(desiredState); // Right now, this just sets the real state to the desired state (no control inputs)
    stateEstimator.Predict(accel_L, accel_R, 0, dt); // Assume zero vertical acceleration

    // Create noisy measurements for GPS and Pressure
    static std::default_random_engine generator;
    std::normal_distribution<float> gps_noise(0.0, stateEstimator.std_gps);
    std::normal_distribution<float> press_noise(0.0, stateEstimator.std_press);

    float noisy_gps_x = trueState.pose.x + gps_noise(generator);
    float noisy_gps_y = trueState.pose.y + gps_noise(generator);
    float noisy_press_z = trueState.pose.z + press_noise(generator);

    stateEstimator.UpdateGPS(noisy_gps_x, noisy_gps_y);
    stateEstimator.UpdatePressure(noisy_press_z);
  
    return;
}

State Agent::ReadState(){
    return matrixToState(stateEstimator.x);
}

void Agent::WriteTrueState(float time){
    true_state_writer.Write(trueState, time);
}

void Agent::WriteEstimatedState(float time){
    State estState = ReadState();
    estimated_state_writer.Write(estState, time);
}

void Agent::WriteDesiredState(float time){
    desired_state_writer.Write(desired, time);
}

void Agent::WriteDesiredTrajectory(){
    std::stringstream directory;
    directory << "sim_data_" << sim_start << "/" << id << "_trajectories/";

    std::stringstream filename;
    filename << id << "_" << std::setfill('0') <<std::setw(10) << int(traj_gen_time*100) << ".csv";
    WriteTrajectory(trajectoryGenerator.trajectory, directory.str() + filename.str(), 0.1);
}
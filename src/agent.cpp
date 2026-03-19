#include "agent.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <random>

Agent::Agent(
        float id_, 
        State initialState, 
        uint32_t sim_start_, 
        float speed, 
        float dive_time, 
        int replan_chance, 
        float timestep_,
        float distance,
        float dive_depth,
        float error_threshold
    ) :
    id(static_cast<int>(id_)),
    trueState(initialState),
    timestep(timestep_),
    sim_start(sim_start_)
    {

    pointGenerator = PointGenerator(distance, dive_depth, error_threshold);
    trajectoryGenerator = TrajectoryGenerator(speed, dive_time);
    stateEstimator = StateEstimator(Pose(0,0,0), timestep_);

    odds = replan_chance;
    traj_gen_time = 0;
    traj_time = 0;
    desired = trueState.pose;

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
    trajectoryGenerator.Update(trueState, desired);

    traj_time = time;
    return;
}

void Agent::NoMove(float time){
    traj_gen_time = time;

    desired = trueState.pose;
    trajectoryGenerator.Update(trueState, desired);

    traj_time = time;
    return;
}

void Agent::SetGoal(Pose desired_state){
    desired = desired_state;
    trajectoryGenerator.Update(trueState, desired);
    
    return;
}

void Agent::Update(float time){
    float dt = timestep; 
    float current_traj_time = time - traj_time;

    State desiredState = trajectoryGenerator.GetDesiredState(current_traj_time);
    float v_desired = sqrtf(desiredState.velocity.x * desiredState.velocity.x + 
                            desiredState.velocity.y * desiredState.velocity.y);

    Input input = controller(stateEstimator.GetState(), desiredState, trajectoryGenerator.GetDesiredAcceleration(current_traj_time));
    trueState = dynamics(trueState, input, timestep);
    stateEstimator.Predict(input);

    static std::default_random_engine generator(1);
    std::normal_distribution<float> gps_noise(0.0, 1.5);
    std::normal_distribution<float> press_noise(0.0, 0.1);
    stateEstimator.UpdateGPS(trueState.pose.x + gps_noise(generator), trueState.pose.y + gps_noise(generator));  
    stateEstimator.UpdatePressure(trueState.pose.z + press_noise(generator));
  
    Input acc = acc_from_input(trueState, input);
    stateEstimator.UpdateIMU(acc.left, acc.right, acc.ballast);
    return;
}

State Agent::ReadState(){
    return stateEstimator.GetState();
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
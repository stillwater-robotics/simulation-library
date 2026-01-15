#include "agent.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>

Agent::Agent(float id_, State initialState, uint32_t sim_start_, float speed, float dive_time, int replan_chance){
    pointGenerator = PointGenerator(2.0f, 3.0f, 0.0f);
    trajectoryGenerator = TrajectoryGenerator(speed, dive_time);
    controller = Controller();

    odds = replan_chance;
    id = id_;
    currentState = initialState;
    traj_gen_time = 0;
    traj_time = 0;
    desired = currentState.pose;

    sim_start = sim_start_;
    std::stringstream directory;
    directory << "sim_data_" << sim_start << "/";

    std::stringstream state_filename;
    state_filename << id << "_states.csv";
    state_writer = Writer(directory.str() + state_filename.str());

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
    desired = pointGenerator.Update(currentState.pose, poses);
    trajectoryGenerator.Update(currentState.pose, desired);

    traj_time = time;
    return;
}

void Agent::Update(float time){
    currentState = controller.Update(trajectoryGenerator.GetDesiredState(time-traj_time));
    return;
}

State Agent::ReadState(){
    // TODO add simulated state estimation error
    return currentState;
}

void Agent::WriteState(float time){
    state_writer.Write(ReadState(), time);
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
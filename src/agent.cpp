#include "agent.h"

#include <sstream>

#include <iostream>  // TODO REMOVE

Agent::Agent(float id_, State initialState, uint32_t sim_start_, float speed, float dive_time){
    // TODO expose
    pointGenerator = PointGenerator(1.0f, 1.0f, 1.0f);
    trajectoryGenerator = TrajectoryGenerator(speed, dive_time);
    controller = Controller();

    id = id_;
    currentState = initialState;
    time = 0;
    traj_time = 0;
    desired = currentState.pose;

    sim_start = sim_start_;
    std::stringstream state_filename;
    state_filename << "sim_data_" << sim_start << "_" << id << "_states.csv";
    state_writer = Writer(state_filename.str());

    std::stringstream desired_filename;
    desired_filename << "sim_data_" << sim_start << "_" << id << "_desired_poses.csv";
    std::cout << desired_filename.str() << std::endl;
    desired_state_writer = Writer(desired_filename.str());
}

void Agent::Plan(std::vector<State> states){
    std::vector<Pose> poses;
    for(auto it=states.begin(); it < states.end(); ++it){
        poses.push_back((*it).pose);
    }
    desired = pointGenerator.Update(currentState.pose, poses);
    trajectoryGenerator.Update(currentState.pose, desired);

    traj_time = time;
    return;
}

void Agent::Update(float time_step){
    controller.Update(trajectoryGenerator.GetDesiredState(time-traj_time));
    time += time_step;
    return;
}

State Agent::ReadState(){
    // TODO add simulated state estimation error
    return currentState;
}

void Agent::WriteState(){
    state_writer.Write(ReadState(), time);
}

void Agent::WriteDesiredState(){
    desired_state_writer.Write(desired, time);
}

void Agent::WriteDesiredTrajectory(){
    std::stringstream filename;
    filename << "sim_data_" << sim_start << "_" << id << "_trajectory_" << traj_time << ".csv";
    WriteTrajectory(trajectoryGenerator.trajectory, filename.str(), 0.1);
}
#include "agent.h"

#include <chrono>
#include <cmath>
#include <iostream>

#define NUM_AGENTS 10
#define SPEED 0.5
#define DIVE_TIME 5
#define REPLAN_ODDS 100 ///(0.5*NUM_AGENTS)

#define SIM_TIME 60
#define CONTROL_TIME_STEP 0.1
#define PLAN_TIME_STEP 10*CONTROL_TIME_STEP

State get_start_state(int id){
    return State(Pose(id*pow(-1, id), id, 0, id*3.14/2), Pose(0, 0, 0, 0));
}

int main() {
    // Initilize the simulation
    uint32_t sim_start = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()
                   ).count();

    std::cout << "Writing sim data to: sim_data_" << sim_start << std::endl;
    
    std::vector<Agent> agents;
    for (int i=1; i<=NUM_AGENTS; i++){
        agents.push_back(Agent(i, get_start_state(i), sim_start, SPEED, DIVE_TIME, REPLAN_ODDS));
    }
    for (float i=0; i<=SIM_TIME; i+=CONTROL_TIME_STEP){
        if (fmod(i, PLAN_TIME_STEP) < CONTROL_TIME_STEP){
            // Replan all agents
            for (auto it=agents.begin(); it < agents.end(); ++it){
                std::vector<State> states;
                for (auto it2=agents.begin(); it2 < agents.end(); ++it2){
                    if (it->id == it2->id){
                        continue;
                    }
                    states.push_back(it2->ReadState());
                }
                it->Plan(states, i);
                it->WriteDesiredState(i);
                it->WriteDesiredTrajectory();
            }
        }

        // Update all agents
        for (auto it=agents.begin(); it < agents.end(); ++it){
            it->Update(i);
            it->WriteState(i);
        }
    }
}
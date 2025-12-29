#include "agent.h"

#include <chrono>
#include <cmath>
#include <iostream>

#define NUM_AGENTS 4
#define SPEED 1
#define DIVE_TIME 5

#define SIM_TIME 3
#define CONTROL_TIME_STEP 0.1
#define PLAN_TIME_STEP 10*CONTROL_TIME_STEP

State get_start_state(int id){
    return State(Pose(id, id, 0, 0), Pose(0, 0, 0, 0));
}

void replan_all(std::vector<Agent> agents){
    for (auto it=agents.begin(); it < agents.end(); ++it){
        std::vector<State> states;
        for (auto it2=agents.begin(); it2 < agents.end(); ++it2){
            if (it->id == it2->id){
                continue;
            }
            states.push_back(it2->ReadState());
        }
        it->Plan(states);
        it->WriteDesiredState();
        it->WriteDesiredTrajectory();
        std::cout << "wrote replan to files" << std::endl;
    }
}

void update_all(std::vector<Agent> agents){
    for (auto it=agents.begin(); it < agents.end(); ++it){
        it->Update(CONTROL_TIME_STEP);
        it->WriteState();
        std::cout << "wrote update to files" << std::endl;
    }
}

int main() {
    // Initilize the simulation
    uint32_t sim_start = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()
                   ).count();
    
    std::vector<Agent> agents;
    for (int i=1; i<=NUM_AGENTS; i++){
        agents.push_back(Agent(i, get_start_state(i), sim_start, SPEED, DIVE_TIME));
    }
    
    for (float i=0; i<=SIM_TIME; i+=CONTROL_TIME_STEP){
        std::cout << "Simulation time: " << i << std::endl;
        if (fmod(i, PLAN_TIME_STEP) < CONTROL_TIME_STEP){
            replan_all(agents);
        }
        update_all(agents);
    }
}
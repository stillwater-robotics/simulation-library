#include "agent.h"

#include <chrono>
#include <cmath>
#include <iostream>

#define NUM_AGENTS 20
#define SPEED 0.5
#define DIVE_TIME 5
#define REPLAN_ODDS 100 //0.5*NUM_AGENTS

#define SIM_TIME 30
#define CONTROL_TIME_STEP 0.1
#define PLAN_TIME_STEP 10*CONTROL_TIME_STEP

#define SEED 10

State get_start_state(int id){
    return State(
        Pose(rand()%20-10, rand()%20-10, 0, rand()/180*M_PI),
        Pose(0, 0, 0, 0)
    );
}

int main() {
    srand(SEED);

    // Initilize the simulation
    uint32_t sim_start = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()
                   ).count();

    std::cout << "Writing sim data to: sim_data_" << sim_start << std::endl;
    
    std::vector<Agent> agents;
    for (int i=1; i<=NUM_AGENTS; i++){
        agents.push_back(
            Agent(
                i,
                get_start_state(i),
                sim_start,
                SPEED,
                DIVE_TIME,
                REPLAN_ODDS,
                CONTROL_TIME_STEP
            )
        );
    }
    for (float i=0; i<=SIM_TIME; i+=CONTROL_TIME_STEP){
        if (fmod(i, PLAN_TIME_STEP) < CONTROL_TIME_STEP){
            // Replan agent 0
            std::vector<State> states;
            for (auto it=agents.begin()+1; it < agents.end(); ++it){
                states.push_back(it->ReadState());
            }
            agents[0].Plan(states, i);
            agents[0].WriteDesiredState(i);
            agents[0].WriteDesiredTrajectory();

            // Replan the rest
            for (auto it=agents.begin()+1; it < agents.end(); ++it){
                it->NoMove(i);
                it->WriteDesiredState(i);
                it->WriteDesiredTrajectory();
            }
        }

        // Update agent 0
        agents[0].Update(i);
        for (auto it=agents.begin(); it < agents.end(); ++it){
            it->WriteTrueState(i);
        }
    }
}
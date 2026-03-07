#include "agent.h"
#include "controller.h"
#include "state_estimator.h"

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>


#define SIM_TIME 10
#define CONTROL_TIME_STEP 0.1

#define SPEED 0.5
#define DIVE_TIME 5
#define REPLAN_ODDS 100
#define DISTANCE 2
#define DIVE_DEPTH 3
#define ERROR_THRESHOLD 0

int main(){
    uint32_t sim_start = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()
                   ).count();

    State current(Pose(0, 0, 0, 0), Pose(0, 0, 0, 0));
    Pose end(5, 0, 0, 0);

    Agent agent(1,
        current, 
        sim_start, 
        SPEED, 
        DIVE_TIME, 
        REPLAN_ODDS, 
        CONTROL_TIME_STEP,
        DISTANCE,
        DIVE_DEPTH,
        ERROR_THRESHOLD
    );

    agent.SetGoal(end);
    agent.WriteDesiredTrajectory();
    agent.WriteDesiredState(0);

    for (float i=0; i<=SIM_TIME; i+=CONTROL_TIME_STEP){
        agent.Update(i);
        agent.WriteTrueState(i);
        agent.WriteEstimatedState(i);
    }

    std::cout << "Saving data to: sim_data_" << sim_start << std::endl;
}
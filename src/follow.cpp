#include "agent.h"
#include "controller.h"

#include <chrono>
#include <iostream>
#include <sstream>


#define SIM_TIME 10
#define CONTROL_TIME_STEP 0.1

int main(){
    TrajectoryGenerator generator(5, 10);
    State current(Pose(0, 0, 0, 1), Pose(0.1, 0.1, 0, 0));
    Pose end(-1, -1, 0, 0);
    generator.Update(current, end);
    
    // Write Data
    uint32_t sim_start = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()
                   ).count();

    std::stringstream folder;
    folder << "follow_data_" << sim_start << "/";
    Writer state_writer(folder.str() + "1_states.csv");
    Writer desired_writer(folder.str() + "1_desired_poses.csv");

    desired_writer.Write(end, 0);
    WriteTrajectory(generator.trajectory, folder.str() + "1_trajectories/1_0000000000.csv");
    
    Writer state_writer2(folder.str() + "2_states.csv");
    Writer desired_writer2(folder.str() + "2_desired_poses.csv");

    desired_writer2.Write(end, 0);
    WriteTrajectory(generator.trajectory, folder.str() + "2_trajectories/2_0000000000.csv");
    

    for (float i=0; i < SIM_TIME ; i+= CONTROL_TIME_STEP){
        state_writer.Write(current, i);
        state_writer2.Write(generator.GetDesiredState(i), i);

        std::cout << "Time: " << i << std::endl;
        Input input = controller(
            current, 
            generator.GetDesiredState(i), 
            generator.GetDesiredAcceleration(i)
        );

        State new_state = dynamics(current, input, CONTROL_TIME_STEP);

        current = new_state;
    }

    std::cout << "Saving data to: " << folder.str() << std::endl;
}
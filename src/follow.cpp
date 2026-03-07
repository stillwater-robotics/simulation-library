#include "agent.h"
#include "controller.h"
#include "state_estimator.h"

#include <chrono>
#include <iostream>
#include <random>
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
    Writer state_writer(folder.str() + "1_true_states.csv");
    Writer est_state_writer(folder.str() + "1_estimated_states.csv");
    Writer desired_writer(folder.str() + "1_desired_poses.csv");

    desired_writer.Write(end, 0);
    WriteTrajectory(generator.trajectory, folder.str() + "1_trajectories/1_0000000000.csv");
    
    Writer state_writer2(folder.str() + "2_true_states.csv");
    Writer est_state_writer2(folder.str() + "2_estimated_states.csv");
    Writer desired_writer2(folder.str() + "2_desired_poses.csv");

    desired_writer2.Write(end, 0);
    WriteTrajectory(generator.trajectory, folder.str() + "2_trajectories/2_0000000000.csv");
    

    StateEstimator stateEstimator;
    static std::default_random_engine random_generator(1); // 1 is seed

    for (float i=0; i < SIM_TIME ; i+= CONTROL_TIME_STEP){
        state_writer.Write(current, i);
        state_writer2.Write(generator.GetDesiredState(i), i);
        est_state_writer2.Write(generator.GetDesiredState(i), i);

        State desiredState = generator.GetDesiredState(i);
        Pose desiredAcceleration = generator.GetDesiredAcceleration(i);
        
        float v_desired = sqrtf(desiredState.velocity.x * desiredState.velocity.x + 
                            desiredState.velocity.y * desiredState.velocity.y);

        // use inverse kinematics to determine approximate control inputs
        float target_vL = v_desired - (desiredState.velocity.theta * L_BASE / 2.0f);
        float target_vR = v_desired + (desiredState.velocity.theta * L_BASE / 2.0f);
        float accel_L = (target_vL - stateEstimator.x(IDX_VL)) / CONTROL_TIME_STEP;
        float accel_R = (target_vR - stateEstimator.x(IDX_VR)) / CONTROL_TIME_STEP;
    
        stateEstimator.Predict(accel_L, accel_R, 0, CONTROL_TIME_STEP); // Assume zero vertical acceleration

        // Create noisy measurements for GPS and Pressure
        std::normal_distribution<float> gps_noise(0.0, stateEstimator.std_gps);
        std::normal_distribution<float> press_noise(0.0, stateEstimator.std_press);

        float noisy_gps_x = current.pose.x + gps_noise(random_generator);
        float noisy_gps_y = current.pose.y + gps_noise(random_generator);
        float noisy_press_z = current.pose.z + press_noise(random_generator);

        stateEstimator.UpdateGPS(noisy_gps_x, noisy_gps_y);
        stateEstimator.UpdatePressure(noisy_press_z);

        est_state_writer.Write(matrixToState(stateEstimator.x), i);

        std::cout << "Time: " << i << std::endl;
        Input input = controller(
            matrixToState(stateEstimator.x), 
            desiredState, 
            desiredAcceleration
        );

        State new_state = dynamics(current, input, CONTROL_TIME_STEP);

        current = new_state;
    }

    std::cout << "Saving data to: " << folder.str() << std::endl;
}
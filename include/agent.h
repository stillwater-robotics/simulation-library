#include "common.h"
#include "controller.h"
#include "csv.h"
#include "point_generator.h"
#include "trajectory_generator.h"
#include "state_estimator.h"

/**
 * @brief Simulation agent.
 */
class Agent
{
private:
    StateEstimator stateEstimator;
    State trueState;

    PointGenerator pointGenerator;
    TrajectoryGenerator trajectoryGenerator;
    Controller controller;

    float traj_gen_time;
    float traj_time;
    Pose desired;
    int odds;

    uint32_t sim_start;
    Writer true_state_writer;
    Writer estimated_state_writer;
    Writer desired_state_writer;

public:
    int id;

    /**
    * @brief Constructs an agent in a given state
    * 
    * @param id_ ID number assigned to the agent, only for simulation purposes
    * @param initialState starting state of the robot
    * @param sim_start start time of the simulation
    * @param speed average speed while moving in the plane
    * @param dive_time time spent at measurement depth
    * @param replan_chance percent chance the system plans a new route
    */
    Agent(float id_, State initialState, uint32_t sim_start_, float speed, float dive_time, int replan_chance);

    /**
     * @brief Updates the desired trajectory, contains code that will be
     * used on the robot.
     * 
     * @param states States the neighboring agents
     */
    void Plan(std::vector<State> states, float time); 

    /**
     * @brief Updates the state of the robot
     * 
     * @param time_step Time in seconds since last update
    */
    void Update(float time);

    // returns the current state of the agent
    State ReadState();

    // writes the agents information
    void WriteTrueState(float time);
    void WriteEstimatedState(float time);
    void WriteDesiredState(float time);
    void WriteDesiredTrajectory();
};
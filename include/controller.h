#include "common.h"

/**
 * @brief Emulates the low level controller. This code will not run on the
 * real agent.
 * 
 */
class Controller
{
private:
    
public:
    Controller();

    /**
     * @brief Updates the robot state based on the desired state.
     * 
     * @param desired desired robot state.
     * @return State updated robot state.
     */
    State Update(State desired);
};


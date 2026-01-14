#ifndef STATE_ESTIMATOR_H
#define STATE_ESTIMATOR_H
#include <BasicLinearAlgebra.h> 
using namespace BLA;

// System Constants
const float L_BASE = 0.5; 
const int HISTORY_SIZE = 50; // Buffer size for RTS Smoother (Limited by RAM)

// State Index Definitions for readability
#define IDX_X 0
#define IDX_Y 1
#define IDX_Z 2
#define IDX_THETA 3
#define IDX_VL 4
#define IDX_VR 5
#define IDX_VZ 6

struct StateSnapshot {
    Matrix<7> x;      // State Vector
    Matrix<7,7> P;    // Covariance Matrix
    Matrix<7,7> F;    // Jacobian of process model (needed for RTS)
    bool hasGPS;      // Was GPS used in this step?
};

class StateEstimator {
public:
    // State Vector: [x, y, z, theta, vL, vR, vZ]
    Matrix<7> x; 
    
    // Prediction Covariance Matrix
    Matrix<7,7> P;

    // Noise
    float std_gps = 1.5;
    float std_press = 0.2;
    float std_accel = 0.5;

    // History Buffer for RTS Smoother
    StateSnapshot history[HISTORY_SIZE];
    int history_idx = 0;
    bool buffer_full = false;

    StateEstimator() {
        x.Fill(0);
        P.Fill(0);
        for(int i=0; i<7; i++) P(i,i) = 0.1; 
    }

    void predict(float aL, float aR, float aZ, float dt) {
        // Jacobian F (Linearized Model)
        Matrix<7,7> F; 
        F.SetIdentity();
        
        float vL = x(IDX_VL);
        float vR = x(IDX_VR);
        float th = x(IDX_THETA);
        
        // Partial derivatives 
        F(IDX_X, IDX_VL) = 0.5 * cos(th) * dt;
        F(IDX_X, IDX_VR) = 0.5 * cos(th) * dt;
        F(IDX_Y, IDX_VL) = 0.5 * sin(th) * dt;
        F(IDX_Y, IDX_VR) = 0.5 * sin(th) * dt;
        F(IDX_Z, IDX_VZ) = dt;
        F(IDX_X, IDX_THETA) = -0.5 * (vL + vR) * sin(th) * dt;
        F(IDX_Y, IDX_THETA) =  0.5 * (vL + vR) * cos(th) * dt;
        F(IDX_THETA, IDX_VL) = (-1.0/L_BASE) * dt;
        F(IDX_THETA, IDX_VR) = (1.0/L_BASE) * dt;
        
        // x = x + x_dot * dt
        x(IDX_X) += (0.5 * cos(th) * (vL + vR)) * dt;
        x(IDX_Y) += (0.5 * sin(th) * (vL + vR)) * dt;
        x(IDX_Z) += x(IDX_VZ) * dt;
        x(IDX_THETA) += (1.0/L_BASE * (-vL + vR)) * dt;
        
        // Update Velocities based on Control Inputs (Accelerations)
        x(IDX_VL) += aL * dt;
        x(IDX_VR) += aR * dt;
        x(IDX_VZ) += aZ * dt;

        // P = F * P * F^T + Q
        Matrix<7,7> Q; 
        Q.SetIdentity(); 
        Q *= (std_accel * dt * std_accel * dt); // Simple Q approximation

        P = F * P * ~F + Q;

        // 4. Save to History for RTS
        saveToHistory(F);
    }

    void updatePressure(float z_meas) {
        // only observe z
        Matrix<1,7> H; H.Fill(0); H(0, IDX_Z) = 1;
        
        Matrix<1,1> R; R(0,0) = std_press * std_press;

        Matrix<1> z; z(0) = z_meas;
        Matrix<1> y = z - (H * x);
        
        // K = P*H^T * (H*P*H^T + R)^-1
        Matrix<1,1> S = H * P * ~H + R;
        Matrix<1,1> S_inv = S.Inverse(); 
        Matrix<7,1> K = P * ~H * S_inv;

        // Update
        x += K * y;
        Matrix<7,7> I; I.SetIdentity();
        P = (I - K * H) * P;
    }

    void updateGPS(float gps_x, float gps_y) {
        // observe x and y
        Matrix<2,7> H; H.Fill(0);
        H(0, IDX_X) = 1;
        H(1, IDX_Y) = 1;

        Matrix<2,2> R; R.Fill(0);
        R(0,0) = std_gps * std_gps;
        R(1,1) = std_gps * std_gps;

        Matrix<2> z; 
        z(0) = gps_x; 
        z(1) = gps_y;
        
        Matrix<2> y = z - (H * x); 
        
        Matrix<2,2> S = H * P * ~H + R;
        Matrix<2,2> S_inv = Invert(S);
        Matrix<7,2> K = P * ~H * S_inv;

        x += K * y;
        Matrix<7,7> I; I.SetIdentity();
        P = (I - K * H) * P;
    }

    void runRTS_Smoother() { // this is simplified to allow compute on pico
        int current = history_idx - 1; 
        if(current < 0) current = HISTORY_SIZE - 1;

        for(int i = 0; i < HISTORY_SIZE - 1; i++) {
            int next = current; 
            int prev = current - 1; 
            if(prev < 0) prev = HISTORY_SIZE - 1;

            StateSnapshot& s_next = history[next];
            StateSnapshot& s_prev = history[prev];
            current = prev;
        }
    }

private:
    void saveToHistory(Matrix<7,7>& F_k) {
        history[history_idx].x = x;
        history[history_idx].P = P;
        history[history_idx].F = F_k;
        history_idx = (history_idx + 1) % HISTORY_SIZE;
    }
};

#endif
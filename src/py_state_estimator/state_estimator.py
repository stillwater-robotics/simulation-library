import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# =============================
# Simulation parameters
# =============================
DT = 0.1
T = 200
N = int(T / DT)
GPS_DEPTH_LIMIT = 0.2

process_noise_std = np.array([0.01, 0.01, 0.005, 0.001, 0.002])
gps_std = 1.5 # m
pressure_std = 0.2 # m
imu_accel_std = 0.5 # m/s^2

Q = np.diag(process_noise_std**2)
R_full = np.diag([gps_std**2, gps_std**2, pressure_std**2, imu_accel_std**2])

# =============================
# Models
# =============================
def f(x, u):
    v_surge, yaw_rate, u_z = u
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + DT * v_surge * np.cos(x[3])
    x_next[1] = x[1] + DT * v_surge * np.sin(x[3])
    x_next[2] = x[2] + DT * x[4]
    x_next[3] = x[3] + DT * yaw_rate
    x_next[4] = x[4] + DT * (-0.1 * x[4] + u_z)
    return x_next

def h_full(x):
    return np.array([x[0], x[1], x[2], x[4]])

def H_full(x):
    H = np.zeros((4,5))
    H[0,0] = 1
    H[1,1] = 1
    H[2,2] = 1
    H[3,4] = 1
    return H

def H_no_gps(x):
    H = np.zeros((2,5))
    H[0,2] = 1
    H[1,4] = 1
    return H

# =============================
# EKF Class
# =============================
class EKF:
    def __init__(self, x0, P0, Q, R_full):
        self.x_hat = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R_full = R_full
        self.state_dim = len(x0)
        self.x_preds = []
        self.P_preds = []
        self.F_preds = []
        self.x_ests = []
        self.P_ests = []

    def step(self, u, y_meas, gps_available):
        # Linearized F
        v_surge = u[0]
        F = np.eye(self.state_dim)
        F[0,3] = -DT * v_surge * np.sin(self.x_hat[3])
        F[1,3] = DT * v_surge * np.cos(self.x_hat[3])
        F[2,4] = DT
        F[4,4] = 1 - 0.1 * DT
        
        self.F_preds.append(F.copy())

        # Predict
        x_pred = f(self.x_hat, u)
        P_pred = F @ self.P @ F.T + self.Q

        # Measurement
        if gps_available:
            H = H_full(x_pred)
            R = self.R_full
            y_pred = h_full(x_pred)
            y_used = y_meas
        else:
            H = H_no_gps(x_pred)
            R = np.diag([pressure_std**2, imu_accel_std**2])
            y_pred = np.array([x_pred[2], x_pred[4]])
            y_used = np.array([y_meas[2], y_meas[3]])

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_hat = x_pred + K @ (y_used - y_pred)
        self.P = (np.eye(self.state_dim) - K @ H) @ P_pred

        # Store for RTS
        self.x_preds.append(x_pred.copy())
        self.P_preds.append(P_pred.copy())
        self.x_ests.append(self.x_hat.copy())
        self.P_ests.append(self.P.copy())    # P_{k+1|k+1}

        return self.x_hat.copy(), self.P.copy()

# =============================
# RTS Smoother Class
# =============================
class RTS:
    def __init__(self, ekf):
        self.ekf = ekf

    def smooth(self):
        N = len(self.ekf.x_ests)
        x_smooth = [self.ekf.x_ests[-1].copy()]
        P_smooth = [self.ekf.P_ests[-1].copy()]

        for k in range(N-2, -1, -1):
            P_filt_k = self.ekf.P_ests[k]        # P_{k|k}  <-- FIX
            P_pred_next = self.ekf.P_preds[k+1]  # P_{k+1|k}
            F = self.ekf.F_preds[k]              # F_k

            # RTS gain
            G = P_filt_k @ F.T @ np.linalg.inv(P_pred_next)

            # Smoothed state
            x_next = x_smooth[0]
            x_s = self.ekf.x_ests[k] + G @ (x_next - self.ekf.x_preds[k+1])
            x_smooth.insert(0, x_s.copy())
            
            P_next_s = P_smooth[0]               # P_{k+1|N}
            P_s = P_filt_k + G @ (P_next_s - P_pred_next) @ G.T
            P_smooth.insert(0, P_s.copy())

        return np.array(x_smooth)

# =============================
# Initialization
# =============================
x0 = np.zeros(5)
P0 = np.eye(5)*0.5
ekf = EKF(x0, P0, Q, R_full)

true_traj = np.zeros((N,5))
est_traj = np.zeros((N,5))
gps_flags = np.zeros(N, dtype=bool)
meas_noise_log = np.zeros((N, 4))

# =============================
# Simulation loop
# =============================
for k in range(N):
    t = k * DT

    # Controls
    v_surge = 1.0
    yaw_rate = 0.05 * np.sin(0.1*t)
    u_z = 0.02 if k < ((N-10)/2) else -0.05 
    u = np.array([v_surge, yaw_rate, u_z])

    # True state
    if k == 0:
        x_true = np.zeros(5)
    x_true = f(x_true, u) + np.random.normal(0, process_noise_std)
    if x_true[2] < 0.00: # Can't go above water
        x_true[2] = 0.00
    true_traj[k] = x_true.copy()

    # Measurement
    y_true = h_full(x_true)
    noise_sample = np.random.normal(0, [gps_std, gps_std, pressure_std, imu_accel_std])
    y_meas = y_true + noise_sample
    meas_noise_log[k, :] = noise_sample

    gps_available = x_true[2] <= GPS_DEPTH_LIMIT
    gps_flags[k] = gps_available

    # EKF step
    x_est, P_est = ekf.step(u, y_meas, gps_available)
    est_traj[k] = x_est.copy()

# =============================
# RTS smoothing
# =============================
rts = RTS(ekf)
smoothed_traj = rts.smooth()

# =============================
# Diagnostics & Quantification
# =============================
print('\n=== Measurement noise statistics (configured vs sampled) ===')
print(f'Configured stds -> GPS: {gps_std:.3f} m, Pressure: {pressure_std:.3f} m, IMU accel: {imu_accel_std:.3f} m/s^2')
meas_means = meas_noise_log.mean(axis=0)
meas_stds = meas_noise_log.std(axis=0)
meas_mins = meas_noise_log.min(axis=0)
meas_maxs = meas_noise_log.max(axis=0)
names = ['GPS_x', 'GPS_y', 'Pressure_z', 'IMU_az']
for i, name in enumerate(names):
    print(f'{name}: sampled mean={meas_means[i]:+.4f}, std={meas_stds[i]:.4f}, min={meas_mins[i]:+.4f}, max={meas_maxs[i]:+.4f}')

# Estimation errors (est - true)
errs = est_traj - true_traj
abs_errs = np.abs(errs)
state_names = ['x (m)', 'y (m)', 'z (m)', 'theta (rad)', 'vz (m/s)']
print('\n=== Estimation error statistics (est - true) over whole run ===')
for i, sname in enumerate(state_names):
    print(f'{sname}: mean={errs[:,i].mean():+.4f}, std={errs[:,i].std():.4f}, min={errs[:,i].min():+.4f}, max={errs[:,i].max():+.4f}')

# Position error magnitude (xy)
pos_err = np.sqrt((errs[:,0])**2 + (errs[:,1])**2)
print('\nPosition error (XY) statistics:')
print(f' RMS={np.sqrt(np.mean(pos_err**2)):.4f} m, mean={pos_err.mean():.4f} m, std={pos_err.std():.4f} m, min={pos_err.min():.4f} m, max={pos_err.max():.4f} m')

# Convergence time after surfacing: detect transitions where gps_flags goes False->True
threshold = 2.0 * gps_std  # convergence threshold (assumption)
surf_indices = np.where((~gps_flags[:-1]) & (gps_flags[1:]))[0] + 1
convergence_times = []
print(f'\nAssumed convergence threshold after surfacing: {threshold:.3f} m (2*GPS std)')
if len(surf_indices) == 0:
    print('No surfacing events detected (gps never regained after being lost).')
else:
    for si in surf_indices:
        # find earliest index >= si where pos_err <= threshold
        below_idx = np.where(pos_err[si:] <= threshold)[0]
        if below_idx.size == 0:
            conv_time = np.nan
            print(f'Surfacing at t={si*DT:.2f}s: did NOT converge before end of run')
        else:
            conv_idx = si + below_idx[0]
            conv_time = (conv_idx - si) * DT
            print(f'Surfacing at t={si*DT:.2f}s: convergence in {conv_time:.2f} s (at t={(conv_idx*DT):.2f}s)')
        convergence_times.append(conv_time)
    # summary
    conv_arr = np.array([c for c in convergence_times if not np.isnan(c)])
    if conv_arr.size>0:
        print(f'Average convergence time over surfacings that converged: {conv_arr.mean():.2f} s')
    else:
        print('No surfacing convergences completed within run.')

# Quantify drift underwater: consider contiguous segments where gps_flags==False
underwater_mask = ~gps_flags
drift_rates = []
drift_mags = []
start = None
for k in range(N):
    if underwater_mask[k] and start is None:
        start = k
    if (not underwater_mask[k] or k==N-1) and start is not None:
        end = k if not underwater_mask[k] else k
        duration = (end - start) * DT
        if duration <= 0:
            start = None
            continue
        # drift defined as change in position error magnitude from start to end
        drift = pos_err[end-1] - pos_err[start]
        rate = drift / duration
        drift_mags.append(drift)
        drift_rates.append(rate)
        print(f'Underwater segment {start}..{end-1} (t={start*DT:.1f}-{(end-1)*DT:.1f}s): duration={duration:.1f}s, drift={drift:.4f} m, rate={rate:.4f} m/s')
        start = None

if len(drift_rates)>0:
    print(f'Average underwater drift rate: {np.mean(drift_rates):+.6f} m/s, std={np.std(drift_rates):.6f} m/s')
else:
    print('No underwater segments detected; no drift to report.')

# =============================
# RTS smoothed path standard error
# =============================
try:
    smooth_resid = smoothed_traj - true_traj
    n_s = smooth_resid.shape[0]
    smooth_std = smooth_resid.std(axis=0)
    smooth_mean = smooth_resid.mean(axis=0)
    smooth_stderr = smooth_std / np.sqrt(n_s)

    print('\n=== RTS Smoothed Path Residuals (smoothed - true) ===')
    for i, sname in enumerate(state_names):
        print(f'{sname}: mean={smooth_mean[i]:+.4f}, std={smooth_std[i]:.4f}, stderr={smooth_stderr[i]:.6f}')

    # XY position errors
    smooth_pos_err = np.sqrt(smooth_resid[:,0]**2 + smooth_resid[:,1]**2)
    pos_std = smooth_pos_err.std()
    pos_mean = smooth_pos_err.mean()
    pos_stderr = pos_std / np.sqrt(n_s)
    print(f"\nRTS Smoothed XY position error: mean={pos_mean:.4f} m, std={pos_std:.4f} m, stderr={pos_stderr:.6f} m")
except Exception as e:
    print('Failed to compute RTS smoothed path standard error:', e)


# =============================
# 3D Plot
# =============================
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# EKF trajectory red/blue
for i in range(1, N):
    color = 'r' if gps_flags[i] else 'b'
    ax.plot(est_traj[i-1:i+1,0], est_traj[i-1:i+1,1], est_traj[i-1:i+1,2],
            color=color, linewidth=2)

# True trajectory
ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], 'k--', linewidth=2, label='True Path')

# Smoothed trajectory (green)
ax.plot(smoothed_traj[:,0], smoothed_traj[:,1], smoothed_traj[:,2], 'g-', linewidth=2, label='RTS Smoothed')

# Legend
legend_lines = [Line2D([0],[0], color='r', lw=2),
                Line2D([0],[0], color='b', lw=2),
                Line2D([0],[0], color='k', lw=2, linestyle='--'),
                Line2D([0],[0], color='g', lw=2)]
legend_labels = ['EKF (GPS avail)', 'EKF (No GPS)', 'True Path', 'RTS Smoothed']
ax.legend(legend_lines, legend_labels)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Depth [m]')
ax.invert_zaxis()  # positive Z = underwater
ax.set_title('UUV 3D Trajectory with RTS Smoothing')
ax.view_init(elev=30, azim=-60)
plt.show()

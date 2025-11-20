from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def plot_trajectory(filepath: Path, title: str = "Trajectory"):
    trajectory = np.loadtxt(filepath, delimiter=",")

    if trajectory.shape[1] != 10:
        raise ValueError("CSV data is not a valid trajectory.")

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_title(title)
    ax.plot(trajectory[:, 1], trajectory[:, 2], -trajectory[:, 3], label="Trajectory")
    ax.plot(
        trajectory[0, 1], trajectory[0, 2], -trajectory[0, 3], label="Start", marker="o"
    )
    ax.plot(
        trajectory[-1, 1],
        trajectory[-1, 2],
        -trajectory[-1, 3],
        label="Desired",
        marker="o",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


def plot_z_over_time(filepath: Path, title: str = "Trajectory"):
    trajectory = np.loadtxt(filepath, delimiter=",")

    if trajectory.shape[1] != 10:
        raise ValueError("CSV data is not a valid trajectory.")

    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.5, 1.5]) 

    ax0 = plt.subplot(gs[0])
    line0, = ax0.plot(trajectory[:, 0], -trajectory[:, 3], color='b')
    ax0.set_title(title)
    ax0.set_ylabel("Position (m)")

    ax1 = plt.subplot(gs[1], sharex = ax0)
    line1, = ax1.plot(trajectory[:, 0], -trajectory[:, 6], color='g')
    ax1.set_ylabel("Velocity (m/s)")

    ax2 = plt.subplot(gs[2], sharex = ax0)
    line2 =  ax2.plot(trajectory[:, 0], -trajectory[:, 9], color='r')
    ax2.set_ylabel(r'Acceleration (m/$s^2$)')
    ax2.set_xlabel("Time (s)")

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.show()


def main():
    plot_z_over_time(
        "/home/sophie/Capstone/trajectory-generator/build/dive.csv", "Dive Trajectory"
    )
    plot_z_over_time(
        "/home/sophie/Capstone/trajectory-generator/build/dive.csv",
        "Dive Trajectory - First Derivative",
        1,
    )
    plot_z_over_time(
        "/home/sophie/Capstone/trajectory-generator/build/dive.csv",
        "Dive Trajectory - Second Derivative",
        2,
    )
    plot_trajectory(
        "/home/sophie/Capstone/trajectory-generator/build/dive.csv",
        "Dive Trajectory",
    )
    plot_trajectory(
        "/home/sophie/Capstone/trajectory-generator/build/spline.csv",
        "X-Y Trajectory",
    )


if __name__ == "__main__":
    main()

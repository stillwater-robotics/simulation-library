from pathlib import Path
import click
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

class Agent:
    def __init__(self, root: Path, id: int, colour: str = "k") -> None:
        self.root = root
        self.id = id
        self.colour = colour

        # Load True and Estimated states
        self.true_states = np.loadtxt(root / f"{id}_true_states.csv", delimiter=",")
        self.est_states = np.loadtxt(root / f"{id}_estimated_states.csv", delimiter=",")
        self.desired_poses = np.loadtxt(root / f"{id}_desired_poses.csv", delimiter=",")

        # Dynamically determine trajectory length from first file
        files = sorted([str(file) for file in (root / f"{id}_trajectories").iterdir()])
        sample_traj = np.loadtxt(files[0], delimiter=",")
        traj_len = sample_traj.shape[0]

        self.trajectories = np.zeros([self.desired_poses.shape[0], traj_len, 10])
        for i, file in enumerate(files):
            self.trajectories[i] = np.loadtxt(file, delimiter=",")

def animate(agents: list[Agent], timestep: float, replan_timestep: float, lims: int = 15):
    fig, ax = plt.subplots()
    ax.set(xlim=[-lims, lims], ylim=[-lims, lims])
    ax.set_aspect('equal') # Keep the arena square

    # Plot True States (Bright/Solid)
    true_points = [
        ax.plot(agent.true_states[0, 1], agent.true_states[0, 2], "o", 
                color=agent.colour, label=f"Agent {agent.id} True")[0]
        for agent in agents
    ]
    
    # Plot Estimated States (Muted/Transparent)
    est_points = [
        ax.plot(agent.est_states[0, 1], agent.est_states[0, 2], "o", 
                color=agent.colour, alpha=0.3)[0]
        for agent in agents
    ]

    desired_states = [
        ax.plot(agent.desired_poses[0, 1], agent.desired_poses[0, 2], "^", 
                color=agent.colour)[0]
        for agent in agents
    ]

    trajectories = [
        ax.plot(agent.trajectories[0, :, 1], agent.trajectories[0, :, 2], 
                color=agent.colour, linewidth=1)[0]
        for agent in agents
    ]

    def update(frame: int):
        # Update both true and estimated positions
        for i in range(len(agents)):
            true_points[i].set_data([agents[i].true_states[frame, 1]], 
                                    [agents[i].true_states[frame, 2]])
            est_points[i].set_data([agents[i].est_states[frame, 1]], 
                                   [agents[i].est_states[frame, 2]])

        # Update planning-based elements
        if frame * timestep % replan_timestep == 0:
            new_frame = int(frame * timestep / replan_timestep)

            if new_frame >= agents[0].desired_poses.shape[0]:
                return true_points + est_points + desired_states + trajectories

            for i in range(len(agents)):
                desired_states[i].set_data(
                    [agents[i].desired_poses[new_frame, 1]], 
                    [agents[i].desired_poses[new_frame, 2]]
                )
                trajectories[i].set_data(
                    agents[i].trajectories[new_frame, :, 1],
                    agents[i].trajectories[new_frame, :, 2],
                )

        return true_points + est_points + desired_states + trajectories

    return ani.FuncAnimation(
        fig=fig,
        func=update,
        frames=agents[0].true_states.shape[0],
        interval=timestep * 1000,
        blit=True
    )

@click.command()
@click.option('-f', '--folder', required=True, type=click.Path(exists=True), help="simulation data folder")
@click.option('-l', '--limits', default=15, type=int, help="frame bounds")
def main(folder: str, limits: int):
    folder_path = Path(folder)
    # Count agent subdirectories
    agent_dirs = [d for d in folder_path.iterdir() if d.is_dir() and d.name.split('_')[0].isdigit()]
    num_agents = len(agent_dirs)
    
    timestep = 0.1
    replan_timestep = 1.0
    colours = ["red", "blue", "green", "orange", "cyan", "magenta", "black"]

    agents = [Agent(folder_path, i, colours[(i-1) % len(colours)]) for i in range(1, 1 + num_agents)]

    animation = animate(agents, timestep, replan_timestep, limits)
    plt.title("True (Solid) vs Estimated (Muted) Positions")
    plt.show()
    
    plot_average_error(agents, timestep)
    
def plot_average_error(agents: list[Agent], timestep: float):
    num_timesteps = agents[0].true_states.shape[0]
    errors = np.zeros((len(agents), num_timesteps))

    for i, agent in enumerate(agents):
        true_pos = agent.true_states[:, 1:3]
        est_pos = agent.est_states[:, 1:3]
        errors[i] = np.linalg.norm(true_pos - est_pos, axis=1)

    avg_error = np.mean(errors, axis=0)
    time = np.arange(num_timesteps) * timestep

    plt.figure()
    plt.plot(time, avg_error, label="Mean Estimation Error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.title("Average Estimation Error Across All Robots")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
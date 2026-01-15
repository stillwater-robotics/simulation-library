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

        self.states = np.loadtxt(root / f"{id}_states.csv", delimiter=",")
        self.desired_poses = np.loadtxt(root / f"{id}_desired_poses.csv", delimiter=",")

        # TODO replace 101 with expression
        self.trajectories = np.zeros([self.desired_poses.shape[0], 101, 10])
        files = [str(file) for file in (root / f"{id}_trajectories").iterdir()]
        files.sort()
        for i, file in enumerate(files):
            self.trajectories[i] = np.loadtxt(file, delimiter=",")


def animate(agents: list[Agent], timestep: float, replan_timestep: float, lims: int = 15):
    fig, ax = plt.subplots()

    ax.set(xlim=[-lims, lims], ylim=[-lims, lims])

    states = [
        ax.plot(agent.states[0, 1], agent.states[0, 2], f"o{agent.colour}")[0]
        for agent in agents
    ]
    desired_states = [
        ax.plot(
            agent.desired_poses[0, 1], agent.desired_poses[0, 2], f"^{agent.colour}"
        )[0]
        for agent in agents
    ]
    trajectories = [
        ax.plot(agent.trajectories[0, :, 1], agent.trajectories[0, :, 2], agent.colour)[
            0
        ]
        for agent in agents
    ]

    def update(frame: int):
        for i in range(len(states)):
            states[i].set_data(agents[i].states[frame, 1], agents[i].states[frame, 2])

        if frame * timestep % replan_timestep == 0:
            new_frame = int(frame * timestep / replan_timestep)

            if new_frame >= agents[0].desired_poses.shape[0]:
                print(
                    f"""Warning: requested index {new_frame} from desired states with length {
                        agents[0].desired_poses.shape[0]}"""
                )
                return (states, desired_states, trajectories)

            for i in range(len(desired_states)):
                desired_states[i].set_data(
                    agents[i].desired_poses[new_frame, 1],
                    agents[i].desired_poses[new_frame, 2],
                )
                trajectories[i].set_data(
                    agents[i].trajectories[new_frame, :, 1],
                    agents[i].trajectories[new_frame, :, 2],
                )

        return (states, desired_states, trajectories)

    return ani.FuncAnimation(
        fig=fig,
        func=update,
        frames=agents[0].states.shape[0],
        interval=timestep * 1000,
    )

@click.command()
@click.option('-f', '--folder', help="simulation data folder")
@click.option('-l', '--limits', default=15, type=int, help="frame bounds")
def main(folder: str, limits: int):
    num_agents = len([file for file in Path(folder).iterdir() if file.is_dir()])
    timestep = 0.1
    replan_timestep = 1

    colours = ["r", "b", "g", "y", "c", "m", "k"]

    agents = [Agent(Path(folder), i, colours[i % len(colours)]) for i in range(1, 1 + num_agents)]

    animation = animate(agents, timestep, replan_timestep, limits)
    plt.show()


if __name__ == "__main__":
    main()

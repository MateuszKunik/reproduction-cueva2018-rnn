import torch
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice

def plot_square(size: float, center: tuple[float, float]):
    half_size = size / 2
    x_center, y_center = center

    x_coordinates = [
        x_center - half_size,
        x_center + half_size,
        x_center + half_size,
        x_center - half_size,
        x_center - half_size,
    ]
    y_coordinates = [
        y_center - half_size,
        y_center - half_size,
        y_center + half_size,
        y_center + half_size,
        y_center - half_size,
    ]

    plt.plot(x_coordinates, y_coordinates)


def plot_simulation(simulation_data, arena):
    plot_square(arena.size, center=arena.center())
    
    plt.scatter(
        x=[record["x"] for record in simulation_data],
        y=[record["y"] for record in simulation_data],
        color="black"
    )
    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Rodent movement simulation")
    plt.show()


def generate_dataset(controller, num_samples, num_steps, num_inputs, num_outputs):
    trajectories = np.empty(
        (num_samples, num_steps, num_inputs + num_outputs),
        dtype=float,
    )

    for sample_id in range(num_samples):
        trajectory = controller.generate(num_steps)

        trajectories[sample_id] = np.asarray(
            [tuple(islice(step.values(), 1, None)) for step in trajectory],
            dtype=float,
        )

    inputs, targets = np.split(trajectories, indices_or_sections=2, axis=num_inputs)

    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets

def generate_dataset_fast(controller, num_samples, num_steps, num_inputs, num_outputs):
    trajectories = np.empty(
        (num_samples, num_steps, num_inputs + num_outputs),
        dtype=float,
    )

    for sample_id in range(num_samples):
        trajectory = controller.generate_fast(num_steps)

        #TODO: islice performance?
        trajectories[sample_id] = np.asarray(
            [tuple(islice(step.values(), 1, None)) for step in trajectory],
            dtype=float,
        )

    inputs, targets = np.split(trajectories, indices_or_sections=2, axis=num_inputs)
    
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets


def generate_dataset_fast_old(controller,
                          num_samples,
                          num_steps,
                          num_inputs,
                          num_outputs,
                          device):

    traj = torch.empty(
        num_samples,
        num_steps,
        num_inputs + num_outputs,
        dtype=torch.float32,
        device=device,
    )#.detach()

    # temporary CPU buffer for simulation
    buf = np.empty((num_steps, num_inputs + num_outputs),
                   dtype=np.float32)

    for s in range(num_samples):
        controller.generate_array(num_steps, buf)

        traj[s].copy_(torch.from_numpy(buf),
                      non_blocking=True) #kolejna nieblokująca, można jeszcze próbować przez zrównoleglenie tego na procesorze

    inputs, targets = torch.split(
        traj,
        [num_inputs, num_outputs],
        dim=2
    )

    return inputs, targets

def compute_spatial_axis_bounds(arena_center, arena_size):
    half_size = arena_size / 2.0
    return arena_center - half_size, arena_center + half_size


def compute_spatial_bounds(arena_center, arena_size):
    x_center, y_center = arena_center
    x_min, x_max = compute_spatial_axis_bounds(x_center, arena_size)
    y_min, y_max = compute_spatial_axis_bounds(y_center, arena_size)

    return ((x_min, x_max), (y_min, y_max))


def compute_spatial_bin_edges(min_position, max_position, num_bins):
    return np.linspace(min_position, max_position, num_bins + 1)


def compute_spatial_bin_edges_from_bounds(spatial_bounds, num_bins):
    (x_min, x_max), (y_min, y_max) = spatial_bounds

    x_edges = compute_spatial_bin_edges(x_min, x_max, num_bins)
    y_edges = compute_spatial_bin_edges(y_min, y_max, num_bins)

    return x_edges, y_edges


def discretize_spatial_positions(positions, min_position, max_position, num_bins):
    normalized_positions = (positions - min_position) / (max_position - min_position)

    return np.clip(
        np.floor(normalized_positions * num_bins).astype(int), 0, num_bins - 1
    )


def compute_mean_activity_map(x_discretized, y_discretized, unit_activity, num_bins):
    activity_map = np.zeros((num_bins, num_bins), dtype=float)
    occupancy_map = np.zeros((num_bins, num_bins), dtype=int)

    np.add.at(activity_map, (x_discretized, y_discretized), unit_activity)
    np.add.at(occupancy_map, (x_discretized, y_discretized), 1)

    mean_activity_map = activity_map / np.maximum(occupancy_map, 1)

    return mean_activity_map, occupancy_map


def compute_unit_activity_map(positions, unit_activity, spatial_bounds, num_bins):
    x_position, y_position = positions[:, 0], positions[:, 1]
    (x_min, x_max), (y_min, y_max) = spatial_bounds

    x_discretized = discretize_spatial_positions(
        x_position, x_min, x_max, num_bins
    )
    y_discretized = discretize_spatial_positions(
        y_position, y_min, y_max, num_bins
    )
    activity_map, occupancy_map = compute_mean_activity_map(
        x_discretized, y_discretized, unit_activity, num_bins
    )

    return activity_map, occupancy_map


def plot_activity_map(activity_map, x_edges, y_edges, title=None):
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        x_edges, y_edges, activity_map, cmap="viridis", shading="flat"
    )
    fig.colorbar(mesh, ax=ax, label="unit mean activity")
    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title)

    plt.show()
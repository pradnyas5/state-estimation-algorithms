# Project: Non Linear Kalman Filter
# Task III: Visualization
# Author: Pradnya Sushil Shinde

import matplotlib.pyplot as plt
import math

def visualizeData(ground_truth, positions, orientations, time_stamps, file_name):
    # Plot 3D trajectory
    gt_coords = [(gt["x"], gt["y"], gt["z"]) for gt in ground_truth]
    
    est_coords = [(position[0], position[1], position[2]) for position in positions]
   
    x_gt = [gt[0] for gt in gt_coords]
    y_gt = [gt[1] for gt in gt_coords]
    z_gt = [gt[2] for gt in gt_coords]

    yaw_gt = [gt["yaw"] for gt in ground_truth]
    pitch_gt = [gt["pitch"] for gt in ground_truth]
    roll_gt = [gt["roll"] for gt in ground_truth]

    gt_time_stamps = [gt["TimeID"] for gt in ground_truth]
    
    x_est = [position[0] for position in est_coords]
    y_est = [position[1] for position in est_coords]
    z_est = [position[2] for position in est_coords]

    yaw_est = [orientation[2] for orientation in orientations]
    pitch_est = [orientation[1] for orientation in orientations]
    roll_est = [orientation[0] for orientation in orientations]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Orientation vs Time: Ground Truth & Estimated Positions")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Yaw")
    axs[0].set_title("Yaw")
    axs[0].set_ylim(-math.pi / 2, math.pi / 2)
    axs[0].plot(gt_time_stamps, yaw_gt, label="Ground Truth", color = "green")
    axs[0].plot(time_stamps, yaw_est, label="Estimated", color="red")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch")
    axs[1].set_ylim(-math.pi / 2, math.pi / 2)
    axs[1].plot(gt_time_stamps, pitch_gt, label="Ground Truth", color = "green")
    axs[1].plot(time_stamps, pitch_est, label="Estimated", color="red")
    axs[1].legend()

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Roll")
    axs[2].set_title("Roll")
    axs[2].set_ylim(-math.pi / 2, math.pi / 2)
    axs[2].plot(gt_time_stamps, roll_gt, label="Ground Truth", color = "green")
    axs[2].plot(time_stamps, roll_est, label="Estimated", color="red")
    axs[2].legend()

    
    fig.savefig(f"./Results/{file_name}_orientations.png")

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Position vs Time: Ground Truth & Estimated Data")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X")
    axs[0].set_title("X")
    axs[0].plot(gt_time_stamps, x_gt, label="Ground Truth", color = "green")
    axs[0].plot(time_stamps, x_est, label="Estimated", color="red")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Y")
    axs[1].set_title("Y")
    axs[1].plot(gt_time_stamps, y_gt, label="Ground Truth", color = "green")
    axs[1].plot(time_stamps, y_est, label="Estimated", color="red")
    axs[1].legend()

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Z")
    axs[2].set_title("Z")
    axs[2].plot(gt_time_stamps, z_gt, label="Ground Truth", color = "green")
    axs[2].plot(time_stamps, z_est, label="Estimated", color="red")
    axs[2].legend()

    
    fig.savefig(f"./Results/{file_name}_positions.png")

def visualizeTrajectory(traj_coords, type):
    cmap = 'inferno'
    colors = range(len(traj_coords))
    x_traj = [traj[0] for traj in traj_coords]
    y_traj = [traj[1] for traj in traj_coords]
    z_traj = [traj[2] for traj in traj_coords]

    fig = plt.figure(figsize=(10,6), layout="tight")
    axes = plt.axes(projection="3d")

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title(type)
    
    sc = axes.scatter3D(x_traj, y_traj, z_traj, c=colors, cmap=cmap, linewidth=0.5)
    fig.colorbar(sc, ax=axes, orientation='vertical', label='Point Index')

    return fig


def savePlot(fig, path, file_name, type):

    output = path + file_name + type + ".png"
    fig.savefig(output)
    

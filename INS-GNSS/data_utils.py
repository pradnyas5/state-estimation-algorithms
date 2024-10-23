import csv
from haversine import haversine
import matplotlib.pyplot as plt
import numpy as np

def read_trajectory_data(filename):
    trajectory_data = {}
    
    with open(filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for idx, row in enumerate(csvreader):
    
            time = row['time']
            data = {
                'true_lat': float(row['true_lat']),
                'true_lon': float(row['true_lon']),
                'true_alt': float(row['true_alt']),
                'true_roll': float(row['true_roll']),
                'true_pitch': float(row['true_pitch']),
                'true_yaw': float(row['true_heading']),
                'gyro_x': float(row['gyro_x']),
                'gyro_y': float(row['gyro_y']),
                'gyro_z': float(row['gyro_z']),
                'acc_x': float(row['accel_x']),
                'acc_y': float(row['accel_y']),
                'acc_z': float(row['accel_z']),
                'z_lat': float(row['z_lat']),
                'z_lon': float(row['z_lon']),
                'z_alt': float(row['z_alt']),
                'z_VN': float(row['z_VN']),
                'z_VE': float(row['z_VE']),
                'z_VD': float(row['z_VD'])
            }
            
            trajectory_data[time] = data
    
    return trajectory_data

def computeError(filtered_positions, true_positions, savePath, method):
    """
    Calculate the haversine error.
    :return: haversine error
    """
    # calculate the error
    filtered_positions = filtered_positions.T
    # calculate the distance error
    dist = []
    for i in range(len(filtered_positions)):
        dist.append(haversine((filtered_positions[i][0], filtered_positions[i][1]), (true_positions[i][0], true_positions[i][1])))
    
    # Plot the haversine distancevs time
    plt.figure(figsize=(14, 5))
    plt.plot(dist)
    plt.xlabel("Time")
    plt.ylabel("Haversine Error")
    plt.title("Haversine Error vs Time")
    plt.savefig(savePath + method + "_HaversineError.png")
    # plt.show()
    # np.mean(err)
    return np.mean(dist)


def plot(filtered_positions, true_positions, savePath, method):
    plt.figure()
   
    plt.plot(filtered_positions[:, 0], filtered_positions[:, 1], label="Filtered Data")
    plt.plot(true_positions[:, 0], true_positions[:, 1], label="True Data")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Trajectory")
    plt.legend()
    plt.savefig(savePath+method+"_Trajectory.png")

    # Compare each state in subfigures
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(filtered_positions[:, 0], label="Filtered Data")
    plt.plot(true_positions[:, 0], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Latitude")
    plt.title("Latitude")
    plt.legend()
    # plt.savefig(savePath+method+"_Latitude.png")

    plt.subplot(1, 3, 2)
    plt.plot(filtered_positions[:, 1], label="Filtered Data")
    plt.plot(true_positions[:, 1], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Longitude")
    plt.title("Longitude")
    plt.legend()
    # plt.savefig(savePath+method+"_Longitude.png")

    plt.subplot(1, 3, 3)
    plt.plot(filtered_positions[:, 2], label="Filtered Data")
    plt.plot(true_positions[:, 2], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Altitude")
    plt.title("Altitude")
    plt.legend()
    plt.savefig(savePath+method+"_Data.png")


    plt.figure(figsize=(14, 5))
    plt.plot(filtered_positions[:, 0] - true_positions[:, 0], label="Latitude Error")
    plt.plot(filtered_positions[:, 1] - true_positions[:, 1], label="Longitude Error")
    plt.plot(filtered_positions[:, 2] - true_positions[:, 2], label="Altitude Error")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Positional Error")
    plt.legend()
    plt.savefig(savePath+method+"_Error.png")
    
    # plt.show()

    cmap_filter = 'winter'
    colors_filter = range(len(filtered_positions))
    # print(filtered_positions.shape)

    # cmap_true = 'winter'
    # colors_true = range(len(true_positions))

    fig = plt.figure(figsize=(10,6), layout="tight")
    axes = plt.axes(projection="3d")

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title(type)
    
    sc_filter = axes.scatter3D(filtered_positions[:,0], filtered_positions[:,1], filtered_positions[:,2], c=colors_filter, cmap=cmap_filter, linewidth=0.5, label = 'Filtered Positions Trajectory')
    # sc_true = axes.scatter3D(true_positions[:,0], true_positions[:,1], true_positions[:,2], c=colors_true, cmap=cmap_true, linewidth=0.5)
    fig.colorbar(sc_filter, ax=axes, orientation='vertical', label='Filtered Point Index')
    # fig.colorbar(sc_true, ax=axes, orientation='vertical', label='Point Index')
    fig.savefig(savePath + method + "Trajectory.png")

    




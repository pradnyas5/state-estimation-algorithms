import numpy as np
from data_utils import*
import os
from ukf import*
import pm as PM
from tqdm import tqdm


model = PM.PropagationModel()

path_to_csv = "trajectory_data.csv"
savePath = "./Results/"

trajectory_data = read_trajectory_data(path_to_csv)

timestamps = list(trajectory_data.keys())
true_positions = [[trajectory_data[time]['true_lat'], trajectory_data[time]['true_lon'], trajectory_data[time]['true_alt']] for time in trajectory_data]
true_positions = np.array(true_positions)
true_orientations = [[trajectory_data[time]['true_roll'], trajectory_data[time]['true_pitch'], trajectory_data[time]['true_yaw']] for time in trajectory_data]
gyro_readings = [[trajectory_data[time]['gyro_x'], trajectory_data[time]['gyro_y'], trajectory_data[time]['gyro_z']] for time in trajectory_data]
acc_readings = [[trajectory_data[time]['acc_x'], trajectory_data[time]['acc_y'], trajectory_data[time]['acc_z']] for time in trajectory_data]
GNSS_pos = [[trajectory_data[time]['z_lat'], trajectory_data[time]['z_lon'], trajectory_data[time]['z_alt']] for time in trajectory_data]
GNSS_vel = [[trajectory_data[time]['z_VN'], trajectory_data[time]['z_VE'], trajectory_data[time]['z_VD']] for time in trajectory_data]


if not os.path.exists(savePath):
    os.makedirs(savePath)

methods = {"FF":(1, 1e-3, 0.1),
        "FB": (5, 1e-2, 1e-2) # 5, 1e-7, 1e-7, 3, 1e-1, 1e-1,  1, 1e-1, 1e-1
       }

prev_time = 0.0
prevPos = GNSS_pos[0]
prevRot = true_orientations[0]
prevVel = GNSS_vel[0]

# Initial State of Position, Rotation, and Velocity should be zero.
prevPos = [0, 0, 0]
prevRot = [0, 0, 0]
prevVel = [0, 0, 0]


for method in methods:
    # Initialise the filter for the specific method: 
    # "FF": Feed Forward --- uses Error State Model
    # "FB": Feed Back --- uses Full State Model

    filter = UKF(method, 
                 state_covariance = methods[method][0],  # P
                 process_noise = methods[method][1], # Q
                 measurement_noise = methods[method][2]) # R

    filtered_positions = np.zeros((3, len(timestamps)))
    if method == "FF":
        x = np.hstack((prevPos, prevRot, prevVel, [1, 1, 1]))  # initial state vector - 12 x 1
        print("----Process started for Feed Forward Model----")
    elif method == "FB":
        x = np.hstack((prevPos, prevRot, prevVel, [1, 1, 1, 1, 1, 1]))  # initial state vector - 15 x 1
        print("----Process started for Feed Back Model----")

    print("----Filtering Data----")
    
    with tqdm(total=len(timestamps)) as pbar: # To track progress
        for idx, time in enumerate(timestamps):

            # # Uncomment to process selected batch of data
            # if idx == 100:
            #     break
            # Calculate dt
            pbar.update(1)

            dt = int(time) - prev_time
            prev_time = int(time)

            # Define control inputs

            u = np.vstack((gyro_readings[idx], acc_readings[idx]))
            # Measurements
            z = np.hstack((GNSS_pos[idx], GNSS_vel[idx]))

            x, P = filter.predict(model, x, u, GNSS_pos[idx])    # predict the next state
            x, P = filter.update(x, z)         # refine the state estimate
        
            filtered_positions[:,idx] = x[0:3]

    print("----Computing Error----")
    err = computeError(filtered_positions, true_positions, savePath, method)
    # print("HAVERSINE ERROR " + method +" Model: ", err)
    print("----Finished, Plotting Results----")
    plot(filtered_positions.T, true_positions, savePath, method)

    # filtered_positions = filtered_positions.T
    # print("LATITUDE ERROR " + method + " Model: ", np.mean(filtered_positions[:,0]-true_positions[:,0]))
    # print("LONGITUDE ERROR " + method + " Model: ", np.mean(filtered_positions[:,1]-true_positions[:,1]))
    # print("ALTITUDE ERROR " + method + " Model: ", np.mean(filtered_positions[:,2]-true_positions[:,2]))

    


        

            

            

            



        


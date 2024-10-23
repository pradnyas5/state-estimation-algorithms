# Project: Non Linear Kalman Filter
# Task IV: Non Linear Kalman Filter Implementation
# Author: Pradnya Sushil Shinde

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from tasks.CovarianceEstimation import validate_gt


class EKF:
    def __init__(self, R):
        """
        Initialize the EKF with necessary parameters.

        Args:
            dt: Time step of the system.
            process_noise_covariance: Covariance matrix of process noise.
            measurement_noise_covariance: Covariance matrix of measurement noise.
        """

        # State = [x, y, z, roll, pitch, yaw, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
        self.dims = 15 
        # Process Noise Covariance
        self.Q = np.eye(15)*7 # -- 15x15

        # Estimate Covariance Matrix
        self.P = np.eye(15)*7 # -- 15x15

        # Measurement Noise Covariance
        # self.R = np.eye(6)*((0.05)**2)
        # self.R = np.array([[ 0.16120252, 0.04454244 ,-0.01102814 ,-0.06662617 , 0.12712535, -0.01602593],
        #                     [ 0.04454244 , 0.0303485 ,  0.0034852 , -0.02942521 , 0.04777935 , 0.00137034],
        #                     [-0.01102814 , 0.0034852 ,  0.07066456,  0.00467304,  0.00192919,  0.05175611],
        #                     [-0.06662617, -0.02942521,  0.00467304,  0.14878254, -0.02530463, -0.0107711 ],
        #                     [ 0.12712535,  0.04777935,  0.00192919, -0.02530463 , 0.16553259 ,-0.00537219],
        #                     [-0.01602593,  0.00137034,  0.05175611, -0.0107711,  -0.00537219,  0.12759402]])
        
        self.R = np.array([[ 0.01364079, 0.00266133, -0.00350008, -0.00389962,  0.00469374,  0.00029292],
                           [ 0.00266133, 0.01189318,  0.00255956, -0.00888527,  0.00683178,  0.00069422],
                           [-0.00350008, 0.00255956,  0.01353899,  0.0005029,   0.00487014,  0.00164411],
                           [-0.00389962, -0.00888527,  0.0005029,   0.00875641, -0.0040138,  -0.00063906],
                           [ 0.00469374,  0.00683178,  0.00487014, -0.0040138,   0.01135403,  0.0003331],
                           [ 0.00029292,  0.00069422,  0.00164411, -0.00063906,  0.0003331,   0.00126122]])
        # self.R = R
        self.g = np.array([0, 0, 9.81]).reshape((3,1))  # Gravity vector
        self.bg = np.zeros((3,1))
        self.ba = np.zeros((3,1))
    
        self.dt = 0

        # Initialise the state vector
        self.x = np.zeros((self.dims, 1))

        self.filtered_position = np.zeros((3,1))

    def ProcessModel(self, x, dt, u):
        # Process Model: xdot = [pdot; G(q)_inv*u_w; g + R(q)*u_a; n_bg; n_ba]
        # We will extract the terms for process model

        # Current State Velocity
        p, q, p_dot, bg, ba = x[0:3,:], x[3:6,:], x[6:9,:], x[9:12,:], x[12:15,:]
        # print(p)
        phi, theta, psi = p[0][0], p[1][0], p[2][0]

        p_dot1, p_dot2, p_dot3 = p_dot
        wx, wy, wz, ax, ay, az = u[0,0], u[1,0], u[2,0], u[3,0], u[4,0], u[5,0]


        G_q_inv = np.array([[np.cos(theta), 0, -np.cos(phi) * np.sin(theta)],
                            [0, 1, np.sin(phi)],
                            [np.sin(theta), 0, np.cos(phi) * np.cos(theta)]])
        R_q = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(theta) * np.sin(psi),
                         -np.cos(phi) * np.sin(psi),
                         np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)],
                        [np.cos(psi) * np.sin(phi) * np.sin(theta) + np.cos(theta) * np.sin(psi),
                         np.cos(phi) * np.cos(psi),
                         np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)],
                        [-np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)]])
        

        # Define x_dot
        p_dot = np.array([p_dot1, p_dot2, p_dot3]).reshape((3,1))
        u_w = np.array([wx, wy, wz]).reshape((3,1))
        u_a = np.array([ax, ay, az]).reshape((3,1))
        x_dot = np.vstack([p_dot, np.dot(G_q_inv, u_w), self.g + np.dot(R_q, u_a), self.bg, self.ba])

        # Compute Jacobian of x_dot w.r.t. x to obtain A
        A = self.ComputeJacobianA(x, x_dot)

        # Construct F matrix
        F = np.eye(15) + A * dt

        # Define x_hat
        x = x + x_dot * dt

        return F, x
    
    def ComputeJacobianA(self, x, x_dot):
        # Compute Jacobian matrix
        A = np.zeros((15, 15))

        # Compute each row of the Jacobian matrix
        for i in range(3):
            for j in range(3):
                A[i, j+6] = -x_dot[3+j][0] if i == j else 0
            A[i, 9+i] = -1
        for i in range(3):
            A[i+3, 3+i] = 1
        A[6:9, :3] = np.dot(self.skew_symmetric_matrix(x[3:6,0]), np.eye(3))
        A[6:9, 9:12] = -np.eye(3)
        A[9:12, 9:12] = -np.eye(3)

        return A
    
    def skew_symmetric_matrix(self, v):

        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def predict(self, x, P, u, dt):
        """
        Predict the next state based on the motion model.

        Args:
            x: Previous State Vector
            P: Previous Estimate Covariance Matrix
            u: Control input vector (current state accelerometer and gyroscope measurements)
            dt: transition time
        """
        F, self.x = self.ProcessModel(x, dt, u)

        self.P = F @ P @ F.T + self.Q

        return self.x, self.P
        

    def update(self, x, z, P):
        """
        Update the state estimate based on the measurement. 
        """
        # Measurement model
        H = self.MeasurementJacobian()
        H = H.astype(np.float64)

        P = P.astype(np.float64)
        R = self.R.astype(np.float64)

        # Compute Kalman Gain
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        # Update Step
        y = z - H @ x

        x += K@y
        P = (np.eye(15) - K@H)@P
        return x, P

    
    def MeasurementJacobian(self):
        # Construct H matrix
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        return H

    def runEKF(self, data, time_stamps, position_estimates, orientation_estimates, key):
        """
        Process one iteration of prediction and update.

        Args:
            u: Control input vector (accelerometer and gyroscope measurements).
            z: Measurement vector (e.g., from GPS or IMU).
        """
        # Initialize state for 1st estimate
        curr_time = time_stamps[0]
        prev_time = 0.0
        curr_position = position_estimates[0]
        curr_orientation = orientation_estimates[0]
        
        # self.x = self.get_state(curr_time, prev_time, self.x, curr_position, curr_orientation)
        
        # The initial velocity should be set to zero
        # self.x[6:9] = np.zeros((3,1))

        # Update the previous time stamp after initilaising the state vector
        # prev_time = time_stamps[0]

        # Initialize a  contol input array
        u = np.zeros((6,1))
        filtered_positions = []

        # Iterate over the data file
        # Since we already are processing the data only for the "detected tags"
        # We can use our position and orientations estimates from Task I 
        for idx, dict in enumerate(data):
            # if idx == 0:
            #     continue

            # Start from 2nd timestamp
            curr_time = dict["TimeID"]

            # Define the current state based on the estimates
            self.x = self.get_state(curr_time, prev_time, self.x, position_estimates[idx], orientation_estimates[idx])

            dt = curr_time - prev_time
            prev_time = curr_time
    
            # Set the current measurent readings
            u[0:3,:] = dict["AngVel"].reshape(3,1) # -- Gyroscope Readings
            u[3:6,:] = dict["Acc"].reshape(3,1) # -- Accelerometer Readings
            

            self.x, self.P = self.predict(self.x, self.P, u, dt)

            z = np.vstack((position_estimates[idx].reshape(3,1),
                          orientation_estimates[idx].reshape(3,1)))

            self.x, self.P = self.update(self.x, z, self.P)

            # self.filtered_position[0:3,:] = self.x[0:3,:]
            # filtered_positions[idx] = self.filtered_position.reshape((1,3))
            filtered_positions.append(self.x[0:3,:].reshape(1,3))

        return filtered_positions

    def get_state(self, curr_time, prev_time, prev_State, positions, orientations):
        """
        Get current estimated state vector.
        """

        self.x[0:3,:] = positions.reshape((3,1))
        self.x[3:6,:] = orientations.reshape((3,1))

        self.dt = curr_time - prev_time

        dp = self.x[0:3,:] - prev_State[0:3,:]
        vel = dp/self.dt
        
        self.x[6:9,:] = vel
        self.x[9:12,:] = self.bg
        self.x[12:15,:] = self.ba

        return self.x
    

def visualize_filter_results(data_estimates, ground_truth,filtered_positions, estimated_positions, key):
    # Load .mat file
    
    ground_truth_position = [(gt["x"], gt["y"], gt["z"]) for gt in ground_truth]
    
    # estimated_positions = [(position[0], position[1], position[2]) for position in estimated_positions]
    # Convert lists to numpy arrays
    estimated_positions = np.array(estimated_positions)
    filtered_positions = np.array(filtered_positions)
    ground_truth_position = np.array(ground_truth_position)
    

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth position
    ax.plot(ground_truth_position[:, 0], ground_truth_position[:, 1], ground_truth_position[:, 2], label='Ground Truth', color='red')

    # Plot estimated position
    ax.plot(estimated_positions[:,0], estimated_positions[:,1], estimated_positions[:,2], label='Pose Estimates', color='green')

    # Plot filtered position
    ax.plot(filtered_positions[:,0,0], filtered_positions[:,0,1], filtered_positions[:,0,2], label='EKF Results', color='orange')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ground Truth vs Estimated vs Filtered Position: ({str(key)})')
    ax.legend()
    plt.savefig(f"./Results/{str(key)}_EKF.png")


def plotErr(est_position, est_orientation, filter_position, time, key):

    x_est_err = [err[0] for i, err in enumerate(est_position)]
    y_est_err = [err[1] for i, err in enumerate(est_position)]
    z_est_err = [err[2] for i, err in enumerate(est_position)]

    x_filter_err = [err[0] for i, err in enumerate(filter_position)]
    y_filter_err = [err[1] for i, err in enumerate(filter_position)]
    z_filter_err = [err[2] for i, err in enumerate(filter_position)]

    try:
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle("RMSE EKF: Filtered & Estimated Positions")

        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("X")
        axs[0].set_title("X")   
        axs[0].plot(time, x_est_err, label="Estimated ", color = "green")
        axs[0].plot(time, x_filter_err, label="Filtered", color="red")   
        axs[0].legend()

        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Y")
        axs[1].set_title("Y")
        axs[1].plot(time, y_est_err, label="Estimated", color = "green")
        axs[1].plot(time, y_filter_err, label="Filtered", color="red")
        axs[1].legend()

        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Z")
        axs[2].set_title("Z")
        axs[2].plot(time, z_est_err, label="Estimated", color = "green")
        axs[2].plot(time, z_filter_err, label="Filtered", color="red")
        axs[2].legend()
        fig.savefig(f"./Results/{key}_position_RMSE.png")
    except ValueError:
        print("Invalid Shapes")


def computeRMSE(ground_truth, positions_estimates, orientation_estimates, filtered_results, time_stamps, key):

    gt_positions = []
    gt_orientations = []

    filtered_results = np.array(filtered_results)

    for idx, estimate in enumerate(positions_estimates):
        estTime = time_stamps[idx]
        try:
            gt = validate_gt(ground_truth, estTime)
        except ValueError:
            continue
        gt_position = gt[0:3]
        # print(gt_position)
        gt_positions.append(gt_position.reshape((1,3)))

        gt_orientation = gt[3:6]
        gt_orientations.append(gt_orientation.reshape((1,3)))

    # Squared Differences
    filter_positions = np.array(filtered_results)
    # filter_orientations = [orientation[0, 3:6] for orientation in filtered_results]
    # print(filter_orientations)

    positions_estimates = np.array(positions_estimates)
    orientation_estimates = np.array(orientation_estimates)
    gt_positions = np.array(gt_positions)
    gt_orientations = np.array(gt_orientations)

    try:
        pos_estimate_error = np.sqrt(np.mean((gt_positions - positions_estimates) ** 2, axis=1))
        or_estimate_error = np.sqrt(np.mean((gt_orientations - orientation_estimates) ** 2, axis=1))
        # print("Postion Error", pos_estimate_error)
        pos_filter_error = np.sqrt(np.mean((gt_positions - filter_positions) ** 2, axis=1))
        # or_filter_error = np.sqrt(np.mean((gt_orientations - filter_orientations) ** 2, axis=1))

        # totalRMSE_filter = np.mean(np.sqrt(pos_filter_error**2 + or_filter_error**2))
        # print("TOTAL", totalRMSE_filter)
        totalRMSE_est = np.mean(np.sqrt(pos_estimate_error**2 + or_estimate_error**2))


        plotErr(pos_estimate_error, or_estimate_error, pos_filter_error, time_stamps, key)
    except ValueError:
        print("Invalid Shape of Errors")

   

   
   

   
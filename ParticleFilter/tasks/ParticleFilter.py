# Project 4: Particle Filter
# Author: Pradnya Sushil Shinde

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from tasks.CovarianceEstimation import validate_gt
import math
from sklearn.cluster import KMeans

class ParticleFilter:
    def __init__(self, num_particles, R):
        """
        Initialize the Particle Filter.

        Args:
            num_particles (int): Number of particles.
            initial_state (array): Initial state of the quadrotor.
            process_noise_covariance (array): Covariance matrix of process noise.
            observation_noise_covariance (array): Covariance matrix of observation noise.
        """
        self.num_particles = num_particles
        self.state_dim = 15
        self.R = np.array([[ 0.01364079, 0.00266133, -0.00350008, -0.00389962,  0.00469374,  0.00029292],
                           [ 0.00266133, 0.01189318,  0.00255956, -0.00888527,  0.00683178,  0.00069422],
                           [-0.00350008, 0.00255956,  0.01353899,  0.0005029,   0.00487014,  0.00164411],
                           [-0.00389962, -0.00888527,  0.0005029,   0.00875641, -0.0040138,  -0.00063906],
                           [ 0.00469374,  0.00683178,  0.00487014, -0.0040138,   0.01135403,  0.0003331],
                           [ 0.00029292,  0.00069422,  0.00164411, -0.00063906,  0.0003331,   0.00126122]])
        
        # self.R = np.eye(6)*(5**2)
        
        self.GyroNoise = 0.5
        self.AccNoise = 100
        

    def getInitState(self):
        """
        Get current estimated state vector.
        """

        x_range = (0.0, 3.0)  # Based on estimates of X
        y_range = (0.0, 3.0)  # Based on estimates of Y
        z_range = (0.0, 1.5)  # Based on estimates of Z

        roll_range = (-np.pi/2, np.pi/2)
        pitch_range = (-np.pi/2, np.pi/2)
        yaw_range = (-np.pi/2, np.pi/2)

        lowRanges = np.array([x_range[0], y_range[0], z_range[0], roll_range[0], pitch_range[0], yaw_range[0]])
        highRanges = np.array([x_range[1], y_range[1], z_range[1], roll_range[1], pitch_range[1], yaw_range[1]])

        particles = np.random.uniform(low=lowRanges, high=highRanges, size=(self.num_particles, 6))    

        particles = np.concatenate((particles, np.zeros((self.num_particles, 9))), axis=1)

        particles = np.expand_dims(particles, axis=-1)

        return particles

    def predict(self, particles, measured_inputs, dt):
        """
        Prediction step of the Particle Filter.

        Args:
            measured_inputs (array): Measured inputs (e.g., angular velocity and linear acceleration).
            dt (float): Time step.

        Returns:
            None
        """
        # Sample from process noise distribution
        process_noise = np.zeros((self.num_particles, 6,1))
        process_noise[:, 0:3] = np.random.normal(scale=self.GyroNoise, size=(self.num_particles, 3, 1))
        process_noise[:, 3:6] = np.random.normal(scale=self.AccNoise, size=(self.num_particles, 3, 1))

        uw = np.tile(measured_inputs[0:3,:], (self.num_particles, 1, 1))
        ua = np.tile(measured_inputs[3:6,:], (self.num_particles, 1, 1))

        ua = ua + process_noise[:, 3:6]
        # print("Process Noise", len(process_noise[0]))

        # Update particles using process model and measured inputs plus noise
        xdot = self.process_model(particles, ua, uw)
        xdot[:, 3:6] = xdot[:, 3:6] + process_noise[:, :3]

        particles = particles + xdot*dt

        return particles


    def update(self, particles, measurements):
        """
        Update step of the Particle Filter.

        Args:
            measurements (array): Measurements obtained from sensors.

        Returns:
            None
        """
        # Measurement Matrix
        H = self.MeasurementJacobian()

        # Extract the diagonal covariances
        R_diag = np.diag(self.R).reshape((1,6))

        # Update measuremnts with the help of particles
        zParticles = ((H @ particles).reshape((self.num_particles, 6)) + R_diag)
        # zParticles = zParticles.reshape((self.num_particles, 6))
        zParticles = np.concatenate((zParticles, np.zeros((self.num_particles, 9))), axis=1) 


        weights = np.exp(-0.5*np.sum((zParticles[:,0:6] - measurements[0:6])**2, axis=1))
    
        weights = weights / np.sum(weights)

        return weights


    def MeasurementJacobian(self):
        # Construct H matrix
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        return H
    
    def process_model(self, xhat, ua, uw):
        """
        Process model of the quadrotor.

        Args:
            state (array): Current state of the quadrotor.
            measured_inputs (array): Measured inputs (e.g., angular velocity and linear acceleration).
            process_noise (array): Process noise for the particle.
            dt (float): Time step.

        Returns:
            array: Updated state of the quadrotor.
        """
        # Unpack state variables
        xdot = np.zeros((self.num_particles, 15, 1))
        # print("Initial State: ", xdot)

        phi, theta, psi = xhat[:,3], xhat[:,4], xhat[:,5]

        R_q = np.zeros((self.num_particles, 3, 3, 1))
        G_q = np.zeros((self.num_particles, 3, 3, 1))

        R_q[:, 0, 0] = np.cos(psi) * np.cos(theta) - np.sin(psi) * np.sin(theta) * np.sin(phi)
        R_q[:, 0, 1] = -np.cos(phi) * np.sin(psi)
        R_q[:, 0, 2] = np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)
        R_q[:, 1, 0] = np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta)
        R_q[:, 1, 1] = np.cos(psi) * np.cos(phi)
        R_q[:, 1, 2] = np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)
        R_q[:, 2, 0] = -np.cos(phi) * np.sin(theta)
        R_q[:, 2, 1] = np.sin(phi)
        R_q[:, 2, 2] = np.cos(phi) * np.cos(theta)
        R_q = R_q.reshape((self.num_particles, 3, 3))

        G_q[:, 0, 0] = np.cos(theta)
        G_q[:, 0, 2] = -np.cos(phi) * np.sin(theta)
        G_q[:, 1, 1] = 1
        G_q[:, 1, 2] = np.sin(phi)
        G_q[:, 2, 0] = np.sin(theta)
        G_q[:, 2, 2] = np.cos(phi) * np.cos(theta)
        G_q = G_q.reshape((self.num_particles, 3, 3))

        xdot[:, 0:3] = xhat[:, 6:9]
        xdot[:, 3:6] = np.linalg.inv(G_q) @ (uw - xhat[:, 9:12])
        xdot[:, 6:9] = np.array([0, 0, -9.81]).reshape((3,1)) + R_q @ (ua - xhat[:, 12:15])

        return xdot
    
    def getEstimatedState(self, particles, weights):
        """
        Get the estimated state of the quadrotor based on particle weights.

        Returns:
            array: Estimated state of the quadrotor.
        """
        # Maximum Weight Estimation
        # max_weight_idx = np.argmax(weights)
        # estimatedState = particles[max_weight_idx]

        # Avg  Estimation
        # estimatedState = np.mean(particles, axis=0)

        # # Weighted Avg Estimation
        estimatedState = np.sum(particles*weights.reshape(self.num_particles, 1, 1), axis=0)

        return estimatedState
    
    def systematicResampling(self, particles, weights):
        # Systematic Resampling
        reParticles = np.zeros((self.num_particles, 15, 1))
        weights = weights / np.sum(weights)
        w_i = weights[0]
        i = 0
        r = np.random.uniform(0, 1/self.num_particles)

        for j in range(0, self.num_particles):
            u = r + (j*1/self.num_particles)
            while w_i < u:
                i+=1
                w_i+=weights[i]
            reParticles[j] = particles[i]
    
        return reParticles
    
    def runPF(self, data, time_stamps, position_estimates, orientation_estimates, key):
        """
        Run the Particle Filter to estimate the state of the quadrotor.

        Args:
            data (list): List of dictionaries containing sensor data.
            time_stamps (array): Array of time stamps corresponding to sensor data.
            position_estimates (array): Array of position estimates.

        Returns:
            array: Filtered positions estimated by the Particle Filter.
        """
        
        filtered_positions = []
        u = np.zeros((6,1))
        prev_time = 0.0

        particles = self.getInitState()
        prev_particles = np.zeros((self.num_particles, 6, len(position_estimates)))

        for idx, dict in enumerate(data):
            # if idx == 0:
            #     continue

            # Calculate dt
            curr_time = dict["TimeID"]
            dt = curr_time - prev_time
            prev_time = curr_time

            # IMU Readings
            u[0:3,:] = dict["AngVel"].reshape(3,1) # -- Gyroscope Readings
            u[3:6,:] = dict["Acc"].reshape(3,1) # -- Accelerometer Readings
            particles = self.predict(particles, u, dt)

            z = np.concatenate((position_estimates[idx], orientation_estimates[idx], [0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=0)
            # print("Measurements:", z)
            
            weights = self.update(particles, z)
            weights = weights / np.sum(weights)
        
            estimatedState = self.getEstimatedState(particles, weights)
            estimatedState = estimatedState[0:6,:].reshape((1,6))
            # print("Estimated State: ", estimatedState)

            prev_particles[:, :, idx] = particles[:, :6, :].reshape((self.num_particles, 6,))   

            particles = self.systematicResampling(particles, weights)

            filtered_positions.append(estimatedState)
        print(np.shape(filtered_positions))

        return filtered_positions


def visualize_filter_results(data_estimates, ground_truth, filtered_positions, estimated_positions, key):
   
    ground_truth_position = [(gt["x"], gt["y"], gt["z"]) for gt in ground_truth]

    estimated_positions = np.array(estimated_positions)
    filtered_positions = np.array(filtered_positions)
    ground_truth = np.array(ground_truth_position)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth position as scatter plot
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', color='red')

    # Plot estimated position as scatter plot
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Pose Estimates', color='green', s=2)

    # Plot filtered position as scatter plot
    ax.scatter(filtered_positions[:, 0, 0], filtered_positions[:, 0, 1], filtered_positions[:, 0, 2], label='EKF Results', color='orange', s=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ground Truth vs Estimated vs Filtered Position: ({str(key)})')
    ax.legend()
    plt.savefig(f"./Results/{str(key)}_PF.png")


def comparePose(ground_truth, positions, orientations, time_stamps, filtered_positions,file_name):

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

    x_filter = [position[0][0] for position in filtered_positions]
    y_filter = [position[0][1] for position in filtered_positions]
    z_filter = [position[0][2] for position in filtered_positions]

    roll_filter = [position[0][3] for position in filtered_positions]
    pitch_filter = [position[0][4] for position in filtered_positions]
    yaw_filter = [position[0][5] for position in filtered_positions]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Orientation vs Time: Ground Truth & Estimated Positions")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Yaw")
    axs[0].set_title("Yaw")
    axs[0].set_ylim(-math.pi / 2, math.pi / 2)
    axs[0].plot(gt_time_stamps, yaw_gt, label="Ground Truth", color = "green")
    axs[0].plot(time_stamps, yaw_est, label="Estimated", color="red")
    axs[0].plot(time_stamps, yaw_filter, label="Filtered", color="orange")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch")
    axs[1].set_ylim(-math.pi / 2, math.pi / 2)
    axs[1].plot(gt_time_stamps, pitch_gt, label="Ground Truth", color = "green")
    axs[1].plot(time_stamps, pitch_est, label="Estimated", color="red")
    axs[1].plot(time_stamps, pitch_filter, label="Filtered", color="orange")
    axs[1].legend()

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Roll")
    axs[2].set_title("Roll")
    axs[2].set_ylim(-math.pi / 2, math.pi / 2)
    axs[2].plot(gt_time_stamps, roll_gt, label="Ground Truth", color = "green")
    axs[2].plot(time_stamps, roll_est, label="Estimated", color="red")
    axs[2].plot(time_stamps, roll_filter, label="Filtered", color="orange")
    axs[2].legend()

    
    fig.savefig(f"./Results/{file_name}_orientations.png")

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Position vs Time: Ground Truth & Estimated Data")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X")
    axs[0].set_title("X")
    axs[0].plot(gt_time_stamps, x_gt, label="Ground Truth", color = "green")
    axs[0].plot(time_stamps, x_est, label="Estimated", color="red")
    axs[0].plot(time_stamps, x_filter, label="Filtered", color="orange")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Y")
    axs[1].set_title("Y")
    axs[1].plot(gt_time_stamps, y_gt, label="Ground Truth", color = "green")
    axs[1].plot(time_stamps, y_est, label="Estimated", color="red")
    axs[1].plot(time_stamps, y_filter, label="Filtered", color="orange")
    axs[1].legend()

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Z")
    axs[2].set_title("Z")
    axs[2].plot(gt_time_stamps, z_gt, label="Ground Truth", color = "green")
    axs[2].plot(time_stamps, z_est, label="Estimated", color="red")
    axs[2].plot(time_stamps, z_filter, label="Filtered", color="orange")
    axs[2].legend()

    
    fig.savefig(f"./Results/{file_name}_positions.png")


def plotErr(est_position, est_orientation, filter_position, filter_orientation, time, key):

    x_est_err = [err[0] for i, err in enumerate(est_position)]
    y_est_err = [err[1] for i, err in enumerate(est_position)]
    z_est_err = [err[2] for i, err in enumerate(est_position)]

    x_filter_err = [err[0] for i, err in enumerate(filter_position)]
    y_filter_err = [err[1] for i, err in enumerate(filter_position)]
    z_filter_err = [err[2] for i, err in enumerate(filter_position)]

    try:
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle("RMSE PF: Filtered & Estimated Positions")

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
    filter_positions = [position[0, 0:3] for position in filtered_results]
    filter_orientations = [orientation[0, 3:6] for orientation in filtered_results]
    # print(filter_orientations)

    positions_estimates = np.array(positions_estimates)
    orientation_estimates = np.array(orientation_estimates)
    gt_positions = np.array(gt_positions)
    gt_orientations = np.array(gt_orientations)

    
    pos_estimate_error = np.sqrt(np.mean((gt_positions - positions_estimates) ** 2, axis=1))
    or_estimate_error = np.sqrt(np.mean((gt_orientations - orientation_estimates) ** 2, axis=1))
    # print("Postion Error", pos_estimate_error)
    pos_filter_error = np.sqrt(np.mean((gt_positions - filter_positions) ** 2, axis=1))
    or_filter_error = np.sqrt(np.mean((gt_orientations - filter_orientations) ** 2, axis=1))

    totalRMSE_filter = np.mean(np.sqrt(pos_filter_error**2 + or_filter_error**2))
    print("TOTAL", totalRMSE_filter)
    totalRMSE_est = np.mean(np.sqrt(pos_estimate_error**2 + or_estimate_error**2))


    plotErr(pos_estimate_error, or_estimate_error, pos_filter_error, or_filter_error, time_stamps, key)

   

   
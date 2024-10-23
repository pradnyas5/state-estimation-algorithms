# Kalman Filter Implementation
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

# Define Kalman Filter class
class KalmanFilter:
    def __init__(self):
        self.m = 0.027 # Mass of the Drone
    
    def computeF_G(self, A, B, dt):
        # To derive discrete time model from the continuous-time model
        # State Space: xdot = Ax + Bu
        # where x = [p pdot] and p is position

        # F = I + A*dt
        F = np.eye(6) + A*dt

        # G = (dt*I + A*((dt*dt)/2))*B
        G = (np.eye(6)*dt + (A*(dt**2))) @ B

        return F, G

    def predict(self,F, G, xhat_n, u_n, P_n, Q):
        # Prediction step
        xhat_pred = F @ xhat_n + G @ u_n
        p_pred = F @ P_n @ F.T + Q
        return xhat_pred, p_pred

    def computeKGain(self, xhat_pred, p_pred, z_n, H, R):
        # Update step
        K = p_pred @ H.T @ np.linalg.inv(H @ p_pred @ H.T + R) 
        y = z_n - H @ xhat_pred 
        return y, K

# Read data from CSV file
def read_data(data_file):
    data = np.loadtxt(data_file, delimiter=',')
    t = data[:,0]
    u = data[:,1:4]  # Input (force)
    z = data[:,4:7]  # Measurement (position or velocity) 
    return t, u, z

# Run Kalman filter on data
def run_kalman_filter(t, u, z, R, Q, StateParam):
    # Initialise a KalmanFilter class object
    kf = KalmanFilter() 
    n = len(t)
    A = np.array([[0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1/kf.m, 0, 0],
                    [0, 1/kf.m, 0],
                    [0, 0, 1/kf.m]])
    P = np.zeros((n, 6, 6)) # Estimate Uncertainity 
    P[0] = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # Initial Estimate Uncertainity
    xhat = np.zeros((n, 6))

    if StateParam == 'Position':
        xhat[0] = np.concatenate([z[0], np.zeros(3)])  # Initial State: [x, y, z, 0, 0, 0]
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    elif StateParam == 'Velocity':
        xhat[0] = np.concatenate([np.zeros(3),z[0]]) # Initial State: [0, 0, 0, xdot, ydot, zdot]
        H = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    for i in range(1,n):
        dt = t[i] - t[i-1]
        F, G = kf.computeF_G(A, B, dt)

        # Prediction Step
        xhat_n = xhat[i-1]
        u_n = u[i-1]
        P_n = P[i-1]
        xhat_pred, p_pred = kf.predict(F, G, xhat_n, u_n, P_n, Q)
    
        z_i = z[i]
        y, K = kf.computeKGain(xhat_pred, p_pred, z_i, H, R)

        # Update Step
        xhat[i] = xhat_pred + K @ y  # Update State
        P[i] = (np.eye(6) - K @ H) @ p_pred # Update Estimate

    return xhat, P

def visualize(xhat):
    p_hat = xhat[:, :3]
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(p_hat[:, 0], p_hat[:, 1], p_hat[:, 2], s=3, color = (204/255, 119/255, 34/255))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Main function
def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--StateParam", default='Position', help="Provide the State Parameter for which Kalman Filter should be implemented.")
    Parser.add_argument("--FileType", default="mocap", help="Choose from: mocap, low_noise, high_noise, velocity")
    Parser.add_argument("--Visualize", default="Yes", help="Bool Value for Visualizing Results(Yes/No)")
    
    Args = Parser.parse_args()
    StateParam = Args.StateParam
    FileType = Args.FileType
    Viz = Args.Visualize
    # Read data
    base_file = "kalman_filter_data_" 
    data_file = base_file + FileType + '.txt'
    t, u, z = read_data(data_file)

    # Q: Process Noise Variance
    # R: Measurement Noise Variance
    # LOW Q: Low Variation in State Transition, HIGH Q: High Variation in State Transition
    # LOW R: Expects Less Noise, HIGH R: Expects High Noise
    if FileType == 'mocap':
        Q = np.eye(6)*((0.008)**2)
        R = np.eye(3)*((1)**2)
    elif FileType == 'low_noise':
        Q = np.eye(6)*((0.01)**2)
        R = np.eye(3)*((0.5)**2)
    elif FileType == 'high_noise':
        Q = np.eye(6)*((0.01)**2)
        R = np.eye(3)*((1)**2)
    elif FileType == 'velocity':
        Q = np.eye(6)*((0.05)**2)
        R = np.eye(3)*((0.5)**2) 

    xhat, P = run_kalman_filter(t, u, z, R, Q, StateParam)

    if Viz == "Yes":
        visualize(xhat)


if __name__ == "__main__":
    main()

import numpy as np
from scipy.linalg import sqrtm

class UKF:

    def __init__(self, method = "FB", state_covariance = 5, process_noise = 1e-3, measurement_noise = 0.5):

        self.method= method
        state_dims = 12 if self.method == "FF" else 15
        self.n = state_dims
       
        self.P = np.eye(self.n)*(state_covariance**2)
        self.Q = np.eye(self.n)*(process_noise**2)
        self.R = np.eye(6,6)*measurement_noise

        self.x = np.zeros((self.n, 1))

        self.kappa = 1.0
        self.alpha = 1.0
        self.beta = 0.4

        self.bg = np.zeros((3, 1))
        self.ba = np.zeros((3, 1))


    def computeSigmaPts(self, x, P):
        """
        Generate sigma points.
        Input:
            x : mean.
            P : covariance matrix.
        Return:
            Sigma Points.
        """
       
        sigma_points = np.zeros((2 * self.n + 1, self.n))
    
        sigma_points[0] = x

        l = self.alpha ** 2 * (self.n + self.kappa) - self.n      # scaling factor
        cov = sqrtm((self.n + l) * P)
      
        for i in range(1, self.n + 1):
            sigma_points[i] = x + cov[i - 1]
            sigma_points[self.n + i] = x - cov[i - 1]

        return sigma_points.T

    def correctCovar(self, covariance, j=1e-2):
        # Check if symmetric, returns bool
        sym = np.allclose(covariance, covariance.T)
        
        try:
            #Check if positive definite
            np.linalg.cholesky(covariance)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False

        # If positive definite and symmetric, return
        if sym and positive_definite:
            return covariance

        covariance = (covariance + covariance.T) / 2

        eig_values, eig_vectors = np.linalg.eig(covariance)
        eig_values[eig_values < 0] = 0
        eig_values += j

        covariance = eig_vectors.dot(np.diag(eig_values)).dot(eig_vectors.T)

        return self.correctCovar(covariance, j=10 * j)

    def computeWeights(self, x):
        """
        Compute weights.
        """

        l = self.alpha ** 2 * (self.n + self.kappa) - self.n  # scaling factor
        meanWts = np.zeros(2 * self.n + 1) + 1 / (2 * (l + self.n))
        covWts = np.zeros(2 * self.n + 1) + 1 / (2 * (l + self.n))

        meanWts[0] = l / (self.n + l)
        covWts[0] = l / (self.n + l) + (1 - self.alpha ** 2 + self.beta)

        return meanWts, covWts

    def FeedBackModel(self, model, x, u):
        """
        Observation model for the full state.
        Input:
            model : Prapogation  Model Object.
            x : current state.
            u : control input.
        Return:
            Updated state.
        """

        prevPos, prevRot, prevVel = x[0:3], x[3:6], x[6:9]
        uw, ua = u[0], u[1]
        dt = 1

        # Use the Propagation Model to update state
        currRot = model.UpdateAtt(prevPos, prevRot, prevVel, uw, dt)
        currVel = model.UpdateVel(prevPos, prevRot, currRot, prevVel, ua, dt)
        currPos = model.UpdatePos(prevPos, prevVel, currVel, dt)

        bias = x[9:15]
        xhat = np.hstack((currPos, currRot, currVel, bias))
        return xhat

    def FeedForwardModel(self, model, x, u, zPos):
        """
        Observation model for the error state.
        Input:
            model : Prapogation Model Object.
            x : current state.
            u : control inputs.
            zPos : GNSS Position Reading.
        Return:
            Propagated state.
        """
        prev_pos, prev_rot, prev_vel = x[0:3], x[3:6], x[6:9]
        uw, ua = u[0], u[1]
        dt = 1
        # print("Control Input :", u)
        # print("Control Input gyro:", uw)
        currRot = model.UpdateAtt(prev_pos, prev_rot, prev_vel, uw, dt)
        currVel = model.UpdateVel(prev_pos, prev_rot, currRot, prev_vel, ua, dt)
        currPos = model.UpdatePos(prev_pos, prev_vel, currVel, dt)

        error = currPos - zPos
        currPos = currPos - error

        xhat = np.hstack((currPos, currRot, currVel, error))
    
        return xhat

    def computeObservation(self, x):
        """
        Compute observation for Feed-Back and Feed-Forward Model.
        Input:
            x : The current state.
        Returns:
            Observation Matrix.
        """
        # observation matrix
        if self.method == "FF":
            H = np.zeros((6, 12))
        elif self.method == "FB":
            H = np.zeros((6, 15))

        H[0:3, 0:3] = np.eye(3, 3)  # Position Observations
        H[3:6, 6:9] = np.eye(3, 3)  # Velocity Observations

        H = H @ x

        return H


    def predict(self, model, x, u, zPos):
        """
        Predict the next state.
        Input:
            model : the model object.
            x : initial state.
            u : imu, gyroscope input.
            P : initial state covariance matrix.
            Q : process noise.
            zPos : GNSS measurement for position.
        Return:
            Predicted mean x and state covariance matrix P.
        """

        # calculate the sigma points and weights
        sigma_pts = self.computeSigmaPts(x, self.P)
        mean_weights, cov_weights = self.computeWeights(x)

        if self.method == "FF":
            for i in range(2 * self.n + 1):
                # Compute predicted state for Feed-Forward
                sigma_pts[:, i] = self.FeedForwardModel(model, sigma_pts[:, i], u, zPos)
        elif self.method == "FB":
            for i in range(2 * self.n + 1):
                # Compute predicted state for Feed-Back
                sigma_pts[:, i] = self.FeedBackModel(model, sigma_pts[:, i], u)

        x = np.sum(mean_weights * sigma_pts, axis=1)

        D = sigma_pts - x[:,np.newaxis]
        self.P = D @ np.diag(cov_weights) @ D.T + self.Q

        return x, self.P


    def update(self, x, z):
        """
        Update the state.
        Input:
            x : initial state.
            z : observation.
        Return:
            updated state and state covariance.
        """
        # print("Z", z)
        # calculate the sigma points
        sigma = self.computeSigmaPts(x, self.P)
        w_mean, w_cov = self.computeWeights(x)

        # compute sigma for observation
        z_sigma = self.computeObservation(sigma)

        # compute observation mean
        z_mean = np.sum(w_mean * z_sigma, axis=1)
        # print("Z_mean", z_mean)

        # compute observation covariance

        dz = z_sigma - z_mean[:, np.newaxis]
        S = dz @ np.diag(w_cov) @ dz.T + self.R

        #  Compute cross covariance
        V = np.zeros((self.n, z.size))
        dx = sigma - x[:, np.newaxis]
        V += dx @ np.diag(w_cov) @ dz.T

        # Compute Kalman gain
        K = V @ np.linalg.inv(S)   
        # update state mean and covariance
        x += K @ (z - z_mean)
        self.P -= K @ S @ K.T
        self.P = self.correctCovar(self.P)

        return x, self.P
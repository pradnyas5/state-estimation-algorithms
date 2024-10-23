import numpy as np
from scipy.spatial.transform import Rotation as R
import earth
from data_utils import *


class PropagationModel:
    def __init__(self):
    
        self.We = 7.292115e-5  # Earth's rate of rotation
        self.prevOmega_en = np.array([0, 0, 0])
        self.prevOmega_ie = np.array([0, 0, 0])
        self.bg = np.array([1, 1, 1]) * 0.01
        self.ba= np.array([1, 1, 1]) * 0.01

    def UpdateAtt(self, prevPos, prevRot, prevVel, gyro_reading, dt):
        """
        Update the attitude using the angular velocity
        Input:
            prevPos: previous position
            prevRot: previous rotation
            prevVel: previous velocity
            gyro_reading: angular velocity measurement from the IMU
            dt: time step
        Return:
            currRot: updated rotation
        """
        lat, lon, alt = prevPos
        Rn, Re, Rp = earth.principal_radii(lat, alt)     
        Vn, Ve, Vd = prevVel

        # Correct the gyroscope bias
        gyro_reading = gyro_reading - self.bg   

        Omega_ie = np.array([[0, -self.We, 0], [self.We, 0, 0], [0, 0, 0]])

        w_en = np.array([(Ve/(Re + alt)), (-Vn/(Rn + alt)), (-Ve*np.tan(np.deg2rad(lat))/(Re + alt))])

        Omega_en = np.array([[0, w_en[2], -w_en[1]], [-w_en[2], 0, w_en[0]], [w_en[1], -w_en[0], 0]])

        Omega_ib = np.array([[0, -gyro_reading[2], gyro_reading[1]], [gyro_reading[2], 0, -gyro_reading[0]], [-gyro_reading[1], gyro_reading[0], 0]])

        # Convert to rotation matrix from ruler angles
        prevRotMat = R.from_euler('xyz', prevRot, degrees=True)
        prevRotMat = prevRotMat.as_matrix()

        # Update the rotation matrix
        currRotMat = prevRotMat * (np.eye(3) + Omega_ib * dt) - (Omega_ie + Omega_en) * prevRotMat * dt

        self.prevOmega_en = Omega_en
        self.prevOmega_ie = Omega_ie

        # Calculate current euler angles
        currRotMat = R.from_matrix(currRotMat)
        currRot = currRotMat.as_euler('xyz', degrees=True)

        return currRot

    def UpdateVel(self, prevPos, prevRot, currRot, prevVel, acc_reading, dt):
        """
        Update the velocity
        Input:
            prevPos: previous position
            prevRot: previous rotation
            currRot: updated rotation
            prevVel: previous velocity
            acc_reading: acceleration measurement from the IMU
            dt: time step
        Return:
            currVel: updated velocity
        """
        lat, lon, alt = prevPos

        acc_reading = acc_reading - self.ba   

        # Convert to rotation matrix
        prevRotMat = R.from_euler('xyz', prevRot, degrees=True).as_matrix()
        currRotMat = R.from_euler('xyz', currRot, degrees=True).as_matrix()

        Fn = 1/2 * (prevRotMat + currRotMat) @ acc_reading     

        Vn = prevVel + dt * (Fn + earth.gravity(lat, alt) - (self.prevOmega_en + 2 * self.prevOmega_ie) @ prevVel)

        return Vn

    def UpdatePos(self, prevPos, prevVel, currVel, dt):
        """
        Update the position using the velocity
        Input:
            prevPos: previous position
            prevVel: previous velocity
            currVel: updated velocity
            dt: time step
        Return:
            currPos: updated position
        """
        prev_Rn, prev_Re, prev_Rp = earth.principal_radii(prevPos[0], prevPos[2])      # Pass in lat and alt to calculate principla radii

        alt = prevPos[2] - dt/2 * (prevVel[2] + currVel[2])

        lat = prevPos[0] + dt/2 * ((prevVel[0]/(prev_Rn + prevPos[2])) + (currVel[0] / (prev_Rn + alt)))

        curr_Rn, curr_Re, curr_Rp = earth.principal_radii(lat, alt)                      # Current principal radii

        lon = prevPos[1] + dt/2 * ((prevVel[1]/((prev_Re + prevPos[2]) * np.cos(np.deg2rad(prevPos[0]))))
                                    + (currVel[1]/((curr_Re + alt) * np.cos(np.deg2rad(lat)))))

        return np.array([lat, lon, alt])

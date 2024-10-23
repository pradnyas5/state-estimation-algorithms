# Project 4: Particle Filter 
# Author: Pradnya Sushil Shinde

import numpy as np

def estimateCovariances(data, gt, positions, orientations):

    covariances = []

    for idx, position in enumerate(positions):

        estTime = data[idx]["TimeID"]

        # Define Estimates of State (position, orientation)
        estState = np.array(
            [position[0],
            position[1],
            position[2],
            orientations[idx][0],
            orientations[idx][1],
            orientations[idx][2]]).reshape(6,1)

        try:
            gtState= validate_gt(gt, estTime)
        except ValueError:
            continue

        # Calculate IMU Error State
        residuals = gtState - estState

        covariance = np.dot(residuals, residuals.T)
        covariances.append(covariance)

    sum = np.sum(covariances, axis=0)
    num = len(covariances)

    covariance = sum/num

    return covariance


def validate_gt(gt, estTime):

    gta = None
    gtb = None
    for idx, elem in enumerate(gt):
        # if idx == 0:
        #     continue
        if elem["TimeID"] > estTime:
            gta = gt[idx-1]
            gtb = gt[idx]
            break
    
    if gta == None or gtb == None:
        raise ValueError(".....Could not find a ground truth at the given timestamp!....")
         
    dta = estTime - gta["TimeID"]
    dtb = gtb["TimeID"] - estTime
    tdt = gtb["TimeID"] - gta["TimeID"]

    percA = dta/tdt
    percB = dtb/tdt

    va = np.array([gta["x"], gta["y"], gta["z"], gta["roll"], gta["pitch"], gta["yaw"]]).reshape(6,1)
    vb = np.array([gtb["x"], gtb["y"], gta["z"], gtb["roll"], gtb["pitch"], gtb["yaw"]]).reshape(6,1)

    # We will calculate weighted avg
    gtState = (1-percA)*(va) + (1-percB)*(vb)

    return gtState
    







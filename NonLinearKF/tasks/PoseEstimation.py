# Project 3: Non-Linear Kalman Filter 
# Task I: Pose Estimation
# Author: Pradnya Sushil Shinde

from cv2 import solvePnP, Rodrigues
import numpy as np
import cv2

def init_tagsCoords(tagMap):
    tagMap_coords= {}
    offset = 0.178 - 0.152
    for row_index, row in enumerate(tagMap):
        startX = 0.152 * row_index * 2
        for tag_index, tag in enumerate(row):
            startY = 0.152 * tag_index * 2
            if tag_index >= 3:
                startX += offset
            if tag_index >= 6:
                startY += offset

            top_left = (startX, startY, 0)
            top_right = (startX, startY + 0.152, 0)
            bottom_right = (startX + 0.152, startY + 0.152, 0)
            bottom_left = (startX + 0.152, startY, 0)

            tagMap_coords[tag] = (top_left,
                top_right,
                bottom_right,
                bottom_left)
    
    return tagMap_coords

def computeImgCoords(detectedTags):
    
    rows = len(detectedTags)*4
    X = np.zeros((rows,1))
    Y = np.zeros((rows,1))

    corners = ["p1", "p2", "p3", "p4"]

    for i, tag in enumerate(detectedTags):
        for j, corner in enumerate(corners):
            x = tag[corner][0]
            y = tag[corner][1]
            idx = 4*i + j
            X[idx] = x
            Y[idx] = y

    img_coords = np.array(np.hstack((X, Y)))
  
    return img_coords

	
def computeWorldCoords(detectedTags, tagMap_coords):

    rows = len(detectedTags)*4

    X = np.zeros((rows,1))
    Y = np.zeros((rows,1))
    Z = np.zeros((rows,1))

    mapRows = 12 
    mapCols = 9
    numTags = mapRows*mapCols
    corners = ["p1", "p2", "p3", "p4"] 

    for i, tag in enumerate(detectedTags):
        id = int(tag["id"])
        for j, corner in enumerate(corners):
            x = tagMap_coords[id][j][0] 
            y = tagMap_coords[id][j][1] 
            idx = 4*i + j
            X[idx] = x
            Y[idx] = y

    world_coords = np.array(np.hstack((X, Y, Z)))   
    return world_coords


def estimatePose(detectedTags, params):
    # Defining the Map of Tags as an array
    tagMap = [
            [0, 12, 24, 36, 48, 60, 72, 84, 96],
            [1, 13, 25, 37, 49, 61, 73, 85, 97],
            [2, 14, 26, 38, 50, 62, 74, 86, 98],
            [3, 15, 27, 39, 51, 63, 75, 87, 99],
            [4, 16, 28, 40, 52, 64, 76, 88, 100],
            [5, 17, 29, 41, 53, 65, 77, 89, 101],
            [6, 18, 30, 42, 54, 66, 78, 90, 102],
            [7, 19, 31, 43, 55, 67, 79, 91, 103],
            [8, 20, 32, 44, 56, 68, 80, 92, 104],
            [9, 21, 33, 45, 57, 69, 81, 93, 105],
            [10, 22, 34, 46, 58, 70, 82, 94, 106],
            [11, 23, 35, 47, 59, 71, 83, 95, 107],
        ]
    
    tagMap_coords = init_tagsCoords(tagMap)
    img_coords = computeImgCoords(detectedTags)
    world_coords = computeWorldCoords(detectedTags, tagMap_coords)
    camMat = params["K"]
    distCoeffs = params["Dst"]

    num_detected_tags = (np.shape(world_coords)[0])//4

    if num_detected_tags == 1:
        method = cv2.SOLVEPNP_IPPE_SQUARE
    elif num_detected_tags > 1:
        method = 0
    _, rvec, tvec = solvePnP(world_coords, img_coords, camMat, distCoeffs)


    # "rvec" and "tvec" here represent camera-based pose estimate
    # It is given that the camera faces downwards.
    # To transform the camera-based pose estimate from 
    # the camera frame to the robot frame, we will use the following parameters:
    XYZ = params["XYZ"]
    Yaw = params["Yaw"]

    # The camera points dwonwards, hence the z-axis tranformation as defined below:
    rotZ = np.array([[np.cos(Yaw), -np.sin(Yaw), 0],
                     [np.sin(Yaw), np.cos(Yaw), 0],
                     [0, 0, 1]])
    
    # As seen in the map, the x-axis is flipped 180 degree from its actual position, hence the x-axis transformation
    rotX = np.array([[1, 0, 0],
                     [0, -1, 0], 
                     [0, 0, -1]])
    
    # Total Rotation: R = Rx.Rz
    rot = np.dot(rotX, rotZ)

    # Transformation from Camera to Robot (drone)
    Hcd = np.array([
                    [rot[0][0], rot[0][1], rot[0][2], XYZ[0]],
                    [rot[1][0], rot[1][1], rot[1][2], XYZ[1]],
                    [rot[2][0], rot[2][1], rot[2][2], XYZ[2]],
                    [0, 0, 0, 1]
                    ])

    R, _ = Rodrigues(rvec)
    # Transformation from Camera to World 
    Hcw = np.hstack((R, tvec.reshape(-1,1)))
    Hcw = np.vstack((Hcw, [0, 0, 0, 1]))
 
    # Transformation from Drone to World
    Hdw = np.dot(np.linalg.inv(Hcw), Hcd)

    orientation_mat = Hdw[0:3,0:3]   
    orientation = computeEulerAngles(orientation_mat)
    # orientation = Orientation2RPY(orientation)

    position = Hdw[0:3, 3]

    return np.array(orientation), position


def computeEulerAngles(rvec):

    singularity = np.sqrt(rvec[0,0]**2 + rvec[1,0]**2)
    singular = singularity < 1e-6
    if not singular:
        roll = np.arctan2(rvec[2,1], rvec[2,2])
        pitch = np.arctan2(-rvec[2,0], singularity)
        yaw = np.arctan2(rvec[1,0], rvec[0,0])
    else:
        roll = np.arctan2(-rvec[1,2], rvec[1,1])
        pitch = np.arctan2(-rvec[2,0], singularity)
        yaw = 0

    return roll, pitch, yaw

# def Orientation2RPY(
#     orientation):
#     """
#     Takes a rotation matrix and
#     converts it to a tuple of yaw, pitch, and roll.
#     """
#     # Convert the 3x1 matrix to a rotation matrix
#     rotation = Rodrigues(orientation)[0]

#     yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
#     pitch = np.arctan2(
#         -rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
#     )
#     roll = np.arctan2(rotation[2, 1], rotation[2, 2])

#     return roll, pitch, yaw

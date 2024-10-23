# Project 4: Particle Filter
# Author: Pradnya Sushil Shinde
# Collection of helper functions to handle data

import cv2 
import numpy as np
from matplotlib import pyplot
import math
from scipy.io import loadmat
import re

def getParams():
    '''
    Assigns the paramters defined in the parameters.txt file
    Input: path to file 'parameters.txt'.
    Output: A dictionary, 'params' that stores all the parameters.
    '''

    params = {}

    cam_mat = np.array([[314.1779, 0, 199.4848],
                        [0, 314.2218, 113.7838],
                        [0, 0, 1]])
    distort_params = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])

    XYZ = np.array([-0.04, 0.0, -0.03])

    Yaw = math.pi/4 

    tag_ids = np.array([[0, 12, 24, 36, 48, 60, 72, 84, 96],
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
                       [11, 23, 35, 47, 59, 71, 83, 95, 107]])
    
    params['K'] = cam_mat
    params['Dst'] = distort_params
    params['XYZ'] = XYZ
    params['Yaw'] = Yaw
    params['IDs'] = tag_ids

    return params

def loadMat(file_path):
    
    mat = loadmat(file_path, simplify_cells = True)

    matData = mat["data"]
    matTime= mat["time"]
    matVicon = mat["vicon"]

    file_num = re.search(r'\d+', file_path).group()
    print("-----Initialized File: " + file_path + " -----")

    data = []

    if file_path == "./data/studentdata0.mat":
        key = "drpy"
    else:
        key = "omg"

    for i, value in enumerate(matData):
        # Iterate over each time stamp 
        dict = {}
        dict["TimeID"] = value["t"] 
        dict["Image"] = value["img"]
        dict["Orientation"] = value["rpy"]
        dict["AngVel"] = value[key]
        dict["Acc"] = value["acc"]

        tags = []

        if isinstance(value["id"], int):
            value["id"] = [value["id"]]
            for corner in ["p1", "p2", "p3", "p4"]:
                value[corner] = [[value[corner][0]], [value[corner][1]]]

        for idx, val in enumerate(value["id"]):
            # Iterate over each detected tag ID
            tag = {}
            tag["id"] = value["id"][idx]
            tag["p1"] = (value["p4"][0][idx], value["p4"][1][idx])
            tag["p2"] = (value["p3"][0][idx], value["p3"][1][idx])
            tag["p3"] = (value["p2"][0][idx], value["p2"][1][idx])
            tag["p4"] = (value["p1"][0][idx], value["p1"][1][idx])
            tags.append(tag)

        dict["DetectedTags"] = tags
        
        data.append(dict)

    ground_truth = []
    for idx, time in enumerate(matTime):
        ground_truth.append({
            "TimeID":time, 
            "x": matVicon[0][idx],
            "y": matVicon[1][idx],
            "z": matVicon[2][idx],
            "roll": matVicon[3][idx],
            "pitch": matVicon[4][idx],
            "yaw": matVicon[5][idx],
            "vx": matVicon[6][idx],
            "vy": matVicon[7][idx],
            "vz": matVicon[8][idx],
            "wx": matVicon[9][idx],
            "wy": matVicon[10][idx],
            "wz": matVicon[11][idx],
        })

    return data, ground_truth




    

    

    

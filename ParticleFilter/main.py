import cv2 
from matplotlib import pyplot
import numpy as np
import os
import argparse

from utils.data_utils import*
from tasks.PoseEstimation import*
from tasks.Visualization import*
from tasks.CovarianceEstimation import*
from tasks.ParticleFilter import *



def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--VisualizePose", default="No", help="Flag to visualize pose estimates")
    Parser.add_argument("--CompCovar", default="No", help="Flag to compute covranices, if True, calculate during runtime,else use precalculated covarinaces.")
    Parser.add_argument("--VisualizePF", default="No", help="Flag to visualize EKF results")
    Parser.add_argument("--Filter", default="No", help="Flag to implement PF")

    Args = Parser.parse_args()
    VisualizePose = Args.VisualizePose
    ComputeCovar = Args.CompCovar
    VisualizePF = Args.VisualizePF
    ImplementPF = Args.Filter

    params = getParams()

    savePath = "./Results/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    matFiles = {'0': "./data/studentdata0.mat",
            '1': "./data/studentdata1.mat",
            '2': "./data/studentdata2.mat",
            '3': "./data/studentdata3.mat",
            '4': "./data/studentdata4.mat",
            '5': "./data/studentdata5.mat",
            '6': "./data/studentdata6.mat",
            '7': "./data/studentdata7.mat",}
    
    covariances = []
    num_of_particles =  250
    # rmse = []
    rmse_sum = 0
    for key, value in enumerate(matFiles):
        file_path = matFiles[str(key)] 
        data, ground_truth = loadMat(file_path)

        file_name = file_path.split("/")[-1].split(".")[0]

        positions = []
        orientations = []
        time_stamps = []
        data_estimates =[]

        print("-----Estimating Pose-----")
        for idx, dict in enumerate(data):
            # Iterate over each dictionary in data related to each Image Frame
            tags = data[idx]["DetectedTags"]

            # Check if no tags are detcted
            if len(tags) == 0:
                continue

            # Process the data for the iterations where tags get detected
            data_estimates.append(dict)
            time = data[idx]["TimeID"]                 
            time_stamps.append(time)

            # Estimate pose of all the detected tags in the observed image frame
            orientation, position = estimatePose(tags, params)
            orientations.append(orientation)
            positions.append(position)
    
        # print("-----Estimating Covariance-----")
        covariance = estimateCovariances(data_estimates, ground_truth, positions, orientations)
        # print("Covarince Estimates for dataset " + file_name + ": ")
        # print(covariance)
        covariances.append(covariance)

        if ImplementPF == "Yes":
            print("-----Implementing PF-----")
            ekf = ParticleFilter(num_of_particles, covariance)
            filtered_positions = ekf.runPF(data_estimates, time_stamps, positions, orientations, key)
            

            if VisualizePF == "Yes":
                visualize_filter_results(data_estimates, ground_truth, filtered_positions, positions, key)
                comparePose(ground_truth, positions, orientations, time_stamps, filtered_positions,file_name)
                # plot_results(es)
                print("-----Filter Results saved successfully!-----")
        
        
        # computeRMSE(ground_truth, positions, orientations,filtered_positions,time_stamps, key)
        
        if VisualizePose == "Yes":

            visualizeData(ground_truth, positions, orientations, time_stamps, file_name)

            fig = visualizeTrajectory([(gt["x"], gt["y"], gt["z"]) for gt in ground_truth],
                                "Ground-Truth Trajectory")
            savePlot(fig, savePath, file_name, "_gt")

            fig = visualizeTrajectory([(position[0], position[1], position[2]) for position in positions], 
                                "Estimated Trajectory")
            savePlot(fig, savePath, file_name, "_est")

            print("-----Pose Estimates saved successfully!-----")

    # err = rmse_sum/7
    # print("AVG RMSE ERROR:", err)
    if ComputeCovar == "Yes":
        sum = np.sum(covariances, axis=0)
        num = len(covariances)
        covariance = sum/num
        print("Cumulative Covariance Estimate is: ")
        print(covariance)

        
        




if __name__ == "__main__": 

    main()
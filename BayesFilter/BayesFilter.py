#!/usr/bin/evn python
# Author: Pradnya Sushil Shinde

# System State: x_t 
# Measurement State: z_t
# Input to the System: u_t

# p(z_t = 1 | x_t = 1) = 0.6
# p(z_t = 0 | x_t = 1) = 0.4
# p(z_t = 0 | x_t = 0) = 0.8
# p(z_t = 1 | x_t = 0) = 0.2

import argparse

robot_state_push = {'OpenOpen':1, 'OpenClose':0.8, 'CloseOpen':0, 'CloseClose':0.2}
robot_state_do_nothing = {'OpenOpen':1, 'OpenClose':0, 'CloseOpen':0, 'CloseClose':1}
model_probabilities = [[0.8, 0.4], [0.2, 0.6]] # To access elements: define {model_probabilities[z_t][x_t]}

def prediction(action, bel_X0_open, bel_X0_close):
    # Defining a prediction Step that takes in 4 input arguments.
    # bel(X1) = p(X1 | U1 = do nothing, X0 = is open) bel(X0 = is open) + p(x1 | U1 = do nothing, X0 = is closed) bel(X0 = is closed)
    # U1 - Input to the System - Action
    # X1 - Posterior Belief (Close or Open)
    # X0 - Prior Belief (Close or Open)
    # print("Predicting state")
    if action == 'do nothing':
        # bel(X1) = p(X1 = Close| U1 = do nothing, X0 =Open) bel(X0 = is open) + p(x1 = Close | U1 = do nothing, X0 = Close) bel(X0 = is closed)
        bel_bar_X1_open =  robot_state_do_nothing['OpenOpen']*bel_X0_open + robot_state_do_nothing['OpenClose']*bel_X0_close
        bel_bar_X1_close = robot_state_do_nothing['CloseOpen']*bel_X0_open + robot_state_do_nothing['CloseClose']*bel_X0_close
    
    elif action == 'push':
        # bel(X1) = p(X1 = Open | U1 = push, X0 = Open) bel(X0 = is open) + p(X1 = Open | U1 = push, X0 = Close) bel(X0 = is closed)
        bel_bar_X1_open =  robot_state_push['OpenOpen']*bel_X0_open + robot_state_push['OpenClose']*bel_X0_close
        bel_bar_X1_close = robot_state_push['CloseOpen']*bel_X0_open + robot_state_push['CloseClose']*bel_X0_close
    else:
        print('No relevant input action!')
    
    # print("Prediction Step Completed!")

    return bel_bar_X1_open, bel_bar_X1_close

def measurement(sense, bel_bar_X1_open, bel_bar_X1_close):
    # bel(X1) = eta p(Z1 | X1) bel_bar(X1)
    # X1 - Posterior Belief (Close or Open)
    # Z1 - Sense Measurement
    # eta - Normalization Factor
    # print('Updating measurement based on predicted belief.')
    if sense == 'Open':
        # Calculate for when Robot senses the door to be 'Open'.
        # bel(X1) = eta p(Z1 = sense_open | X1) bel_bar(X1)
        bel_X1_open = model_probabilities[1][1]*bel_bar_X1_open
        bel_X1_close = model_probabilities[1][0]*bel_bar_X1_close

        eta = 1 /(bel_X1_open + bel_X1_close)

        bel_X1_open = eta*bel_X1_open
        bel_X1_close = eta*bel_X1_close

    elif sense == 'Close':
         # Calculate for when Robot senses the door to be 'Close'.
        # bel(X1) = eta p(Z1 = sense_close | X1) bel_bar(X1)
        bel_X1_open = model_probabilities[0][1]*bel_bar_X1_open
        bel_X1_close = model_probabilities[0][0]*bel_bar_X1_close
        
        eta = 1 /(bel_X1_open + bel_X1_close)

        bel_X1_open = eta*bel_X1_open
        bel_X1_close = eta*bel_X1_close
    
    # print("Update step compled!")

    return bel_X1_open, bel_X1_close

def compute_steady_state_belief(bel_open, bel_close, prev_bel_open, prev_bel_close, action, sense, threshold):
        while abs(bel_open - prev_bel_open) > threshold or abs(bel_close - prev_bel_close) > threshold:
            prev_bel_open, prev_bel_close = bel_open, bel_close
            bel_bar_open, bel_bar_close = prediction(action, bel_open, bel_close)
            bel_open, bel_close = measurement(sense, bel_bar_open, bel_bar_close)

        return bel_open, bel_close


def compute_iterations(its, bel_X1_open, bel_X1_close, action, sense):
        
        while bel_X1_open < 0.9999:
            bel_bar_X1_open, bel_bar_X1_close = prediction(action, bel_X1_open, bel_X1_close) 
            bel_X1_open, bel_X1_close = measurement(sense, bel_bar_X1_open, bel_bar_X1_close)
            its+=1
        return its


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--InputAction",
        default= 'do nothing',
        help="Sequence of Input ",
    )
    Parser.add_argument(
        "--SensedMeasurement",
        default= 'Open',
        help="Sense whether the door is Open or Close")

    Parser.add_argument(
        "--InitDoorOpenBel",
        default= 0.5,
        help="Initial belief that the door is Open (Optional)",
    )

    Args = Parser.parse_args()
    bel_init_open = Args.InitDoorOpenBel   # Provide an initial probability that the door is open.
    action = Args.InputAction # Provide an action to perform
    sense = Args.SensedMeasurement # Provide the  sensed measurement

    # Given inital belief of door open, calcualte initial belief of door close 
    bel_init_close = 1 - bel_init_open  

    # Set iterations to zero
    if sense == 'Open':
        its = 0 
        bel_X1_open =  bel_init_open 
        bel_X1_close = bel_init_close

        print("Computing iterations to perform action")
        its = compute_iterations(its, bel_X1_open, bel_X1_close, action, sense)
        print("Number of iterations to perform action: " + action + " is: " + str(its))

    elif sense == 'Close':
        bel_open, bel_close = bel_init_open, bel_init_close
        prev_bel_open, prev_bel_close = 0, 0
        threshold = 1e-4

        print("Computing steady state belief.")

        open_steady_state_bel, close_steady_state_bel = compute_steady_state_belief(bel_open, bel_close, prev_bel_open, prev_bel_close, action='push', sense='Close', threshold=threshold)
        print("Steady stae belief for door open is: "+ str(open_steady_state_bel))
        print("Steady stae belief for door close is: "+ str(close_steady_state_bel))
    


main()

# Import the dependencies
from math import *
import numpy as np
import sys
import csv
from numpy import genfromtxt
import pandas as pd
import os


#################### Comparison using Dynamic Time Warping ###############################
def DTW(A, B, window = sys.maxsize, d = lambda x,y: abs(x-y)):
    # create the cost matrix
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxsize * np.ones((M, N))

    # initialize the first row and column
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(A[0], B[j])
    # fill in the rest of the matrix
    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
    
    path.append((0,0))
    return cost[-1, -1], path

################# Comparison using Root Mean Squared Distance ###############################
def RMS(A, B):
    dist = np.sqrt(np.sum(np.square(np.subtract(A, B))))
    return dist

################ Main ##############
def main():

    #file name
    file_name = 'embeddings/embeddings-PRE-new.csv'
    #number of images
    noOfImages = 1000

    # Get the embeddings from csv
    arr_img = genfromtxt(file_name, delimiter=',')

    # Remove index
    arr_img = np.delete(arr_img, 0, 1)

    # Get image names
    img_names = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            img_names.append(row[0])
            
    # Loop through images
    for i in range(1,noOfImages):
        # get pre op name
        img = img_names[i]

        # Count of distance < 1
        count_less_than_1 = 0

        # Match array
        match_array = []
        
        # loop through post-op images
        for j in range(1,noOfImages):

            # get embeddings of both images
            A = arr_img[i]
            B = arr_img[j]

            # Perform DTW comparison betwen the embeddings and get the distance
            # distance, _ = DTW(A, B, window = 4)

            # Perform RMS comparison betwen the embeddings and get the distance
            distance = RMS(A, B)

            # Check for minimum distance
            # if distance < min_dist:
            #     min_dist = distance
            #     post_min = post

            # Check for if distane < 0.5
            if distance < 0.55 and distance != 0:
                count_less_than_1 +=1
                match_array.append(img_names[j])
        
        if count_less_than_1 > 0:
            print(img, count_less_than_1, sep=' : ')

        # if pre-op and post-op image match (testing)
        # if pre == post_min:
        #     count+=1

        #min_dist_arr.append(min_dist)
        #os.system('clear')
        # Progress tracker
        #print("Images matched :", count)
        #print("Processed : ", i, "/ 999")

if __name__ == '__main__':
    main()
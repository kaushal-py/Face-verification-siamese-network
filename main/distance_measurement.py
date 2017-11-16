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
    # Get the embeddings from csv
    arr_pre = genfromtxt('embeddings-PRE.csv', delimiter=',')
    arr_post = genfromtxt('embeddings-POST.csv', delimiter=',')

    # Remove index
    arr_pre = np.delete(arr_pre, 0, 1)
    arr_post = np.delete(arr_post, 0, 1)

    # Scale factor for embeddings
    arr_pre = np.dot(arr_pre, 10)
    arr_post = np.dot(arr_post, 10)

    # Get pre and post names
    post_names = []
    pre_names = []
    with open('embeddings-PRE.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            pre_names.append(row[0])
    with open('embeddings-POST.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            post_names.append(row[0])
            
    # Initialise count of images correctly recognised
    count = 0
    min_dist_arr = []
    # Loop through pre-op images
    for i in range(1,1000):
        # get pre op name
        pre = pre_names[i]
        pre = pre[11:18]

        # Set minimun distance between emdeddings to inf
        min_dist = 2000
        post_min = ""
        
        # loop through post-op images
        for j in range(1,836):
            # get post op name
            post = post_names[j]
            post = post[12:19]

            # get embeddings of pre and post op
            A = arr_pre[i]
            B = arr_post[j]

            # Perform DTW comparison betwen the embeddings and get the distance
            # distance, _ = DTW(A, B, window = 4)

            # Perform RMS comparison betwen the embeddings and get the distance
            distance = RMS(A, B)

            # Check for minimum distance
            if distance < min_dist:
                min_dist = distance
                post_min = post

        # if pre-op and post-op image match (testing)
        if pre == post_min:
            count+=1

        min_dist_arr.append(min_dist)
        os.system('clear')
        # Progress tracker
        print("Images matched :", count)
        print("Processed : ", i, "/ 999")
    
    print(max(min_dist_arr), min(min_dist_arr))

if __name__ == '__main__':
    main()
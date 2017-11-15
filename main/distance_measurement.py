from math import *
import numpy as np
import sys
import csv
from numpy import genfromtxt
import pandas as pd

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


arr_pre = genfromtxt('embeddings-PRE.csv', delimiter=',')
arr_post = genfromtxt('embeddings-POST.csv', delimiter=',')

arr_pre = np.delete(arr_pre, 0, 1)
arr_post = np.delete(arr_post, 0, 1)

# print(arr_pre)

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

# pre_names = np.delete(pre_names, 0, 0)
# post_names = np.delete(post_names, 0, 0)

# print(post_names[1],arr_post[1])
# print(pre_names)
# print(post_names)
cnt=0
for i in range(1,835):
    pre = pre_names[i]
    pre = pre[11:18]
    # print(pre)
    for j in range(1,835):
        post = post_names[j]
        post = post[12:19]
        # print("\t",post)
        if pre == post:
            A = arr_pre[i]
            B = arr_post[j]
            # print('PRE = ',pre)
            # print(A)
            # print('POST = ',post)
            # print(B)
            cost, path = DTW(A, B, window = 4)
            print ('Total Distance is ', cost)
            if cost > 6:
                cnt += 1
            print("Count",cnt)
            break
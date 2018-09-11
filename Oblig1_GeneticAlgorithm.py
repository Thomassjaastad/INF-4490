from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import itertools
import csv
from datetime import datetime
import time
#Solving Traveling Salesman Problem (TSP) with exhaustive search

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))

n = 6   #number of cities
data = np.asarray(data)
CitiesDist = data[1 : n + 1, 0 : n] #Skipping "string", name of cities
 
Cities = np.arange(1, n)  
Cities = np.insert(Cities, 0, 0, axis = 0)

#Problem producing same index
half = n//2
swapA = np.random.randint(n - half)
swapB = np.random.randint(n + half)

def Mutation(Genotype, i, j):
    #The index cannot occur twice 
    Genotype[i:j] = np.flip(Genotype[i:j], 0)
    return Genotype
print(Cities)
print (Mutation(Cities, swapA, swapB))

def DistConvert(Index, Dist):
    #Want to calculate the distance between n cities and return the total distance
    #
    #Remember to calculate to distance from end to start     
    DistValues = np.zeros(n)
    for i in range(n - 1):
        start = Index[i]
        end = Index[i + 1]
        DistValues[i] = Dist[start, end]  
    DistValuesFinal = Dist[Index[n-1],Index[0]]
    DistValues[n-1] = DistValuesFinal
    return np.sum(DistValues)
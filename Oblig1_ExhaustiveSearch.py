from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import itertools
import csv
from datetime import datetime
import time
#Solving Traveling Salesman Problem (TSP) with exhaustive search and hill climbing 

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))

n = 10   #number of cities
data = np.asarray(data)
CitiesDist = data[1 : n + 1, 0 : n] #Skipping "string", name of cities

Cities = np.arange(1, n)  
CitiesPermute = list(itertools.permutations(Cities))
CitiesPermute = np.asarray(CitiesPermute)
CitiesPermute = np.insert(CitiesPermute, 0, 0, axis = 1)

startTime = time.time()
start = datetime.now()

def DistConvert(Index, Dist):
    #Want to calculate the distance between n cities and return the total distance
    #Remember to calculate to distance from end to start     
    DistValues = np.zeros(n)
    for i in range(n - 1):
        start = Index[i]
        end = Index[i + 1]
        DistValues[i] = Dist[start, end]  
    DistValuesFinal = Dist[Index[n-1],Index[0]]
    DistValues[n-1] = DistValuesFinal
    return np.sum(DistValues) 

def ExhaustiveSearch(func, Index, DistMatrix, start, stop):
    increment = start
    Shortest = func(CitiesPermute[increment], CitiesDist)
    while increment < stop:
        ShortestDist = func(CitiesPermute[increment], CitiesDist)   
        if ShortestDist < Shortest:
            Shortest = ShortestDist 
        increment += 1    
    return Shortest

endTime = time.time()
end = datetime.now()

print (ExhaustiveSearch(DistConvert, CitiesPermute[0], CitiesDist, 0, len(CitiesPermute)))
print(end - start)
print (endTime - startTime)
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import csv
from datetime import datetime
import time
#Solving Traveling Salesman Problem (TSP) with exhaustive search and hill climbing 

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))

n = 10   #number of cities
data = np.asarray(data)
CitiesDist = data[1 : n + 1, 0 : n] #Skipping "string", name of cities

CityOrderStart = np.arange(n)
#Clock 
startTime = time.time()
start = datetime.now()

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

#print (DistConvert(cityOrder, CitiesDist))
def HillClimbing(func, n, CityOrder):
    #Chosing a random set of cities and minimizing distance 
    #Calculating distance from two cities 
    for i in range(1000):
        city1 = np.random.randint(n)
        city2 = np.random.randint(n)
        if city1 != city2:
            possibleCityOrder = CityOrder.copy() 
            possibleCityOrder = np.where(possibleCityOrder == city1, -1, possibleCityOrder)    
            possibleCityOrder = np.where(possibleCityOrder == city2, city1, possibleCityOrder)
            possibleCityOrder = np.where(possibleCityOrder == -1, city2 , possibleCityOrder)

            CurrentDistance = func(CityOrder, CitiesDist)
            NewDistance = func(possibleCityOrder, CitiesDist)

            if NewDistance < CurrentDistance:
                CurrentDistance = NewDistance
                CityOrder = possibleCityOrder    
    return CurrentDistance, list(CityOrder)


#Running 20 times to measure performance
MeasureDist = np.zeros(20)
TrueValue = 7486.31          #For ten cities

for i in range(20):
    MeasureDist[i] = HillClimbing(DistConvert, n, CityOrderStart)[0]
    
    if TrueValue in MeasureDist:
        print (MeasureDist[i])

MeanDist = sum(MeasureDist)/20    #20 is number of runs
StdDist = np.std(MeasureDist)  
#print (MeasureDist)

#print (HillClimbing(DistConvert, n, CityOrderStart))
#Clock end
endTime = time.time()
end = datetime.now()
#print (end - start)
#print (endTime- startTime)
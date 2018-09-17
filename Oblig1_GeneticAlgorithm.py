from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import itertools
import csv
from datetime import datetime
import time
import random
#Solving Traveling Salesman Problem (TSP) with GeneticAlgorithm

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))

n = 6   #number of cities
data = np.asarray(data)
CitiesDist = data[1 : n + 1, 0 : n] #Skipping "string", name of cities
Cities = np.arange(n)  
#Cities = np.insert(Cities, 0, 0, axis = 0)
CitiesPermute = list(itertools.permutations(Cities))
CitiesPermute = np.asarray(CitiesPermute)
Cit = np.random.shuffle(CitiesPermute)

#CitiesPermute = np.insert(CitiesPermute, 0, 0, axis = 1)

population = CitiesPermute[1 : 6, :]
population = list(population)
#print (population)
#print (random.sample(population, 2))

startTime = time.time()
 
def Fitness(Index, Dist):
    #Want to calculate the distance between n cities and return the total distance
    #Remember to calculate to distance from end to start at the end of loop    
    DistValues = np.zeros(n)
    for i in range(n - 1):
        start = Index[i]
        end = Index[i + 1]
        DistValues[i] = Dist[start, end]  
    DistValuesFinal = Dist[Index[n - 1],Index[0]]
    DistValues[n - 1] = DistValuesFinal
    return 1/np.sum(DistValues), DistValues

def NormFitness(func, Index, Dist, populationSet):
    SumPopulation = 0
    FitnessOne = np.zeros(len(populationSet))
    for i in range(len(populationSet)):
        SumPopulation += func(populationSet[i], CitiesDist)[0] 
        FitnessOne[i] = func(populationSet[i], CitiesDist)[0]
    NormalizedFitness = FitnessOne/SumPopulation
    return NormalizedFitness

def TournamentSelection(lamb, k, FitnessCriteria):  #k < lamb
    current_member = 1
    mating_pool = []
    individual = random.sample(population, k)
    while current_member < lamb:     
        EvalFitness = NormFitness(Fitness, individual, CitiesDist, individual)
    
        print(np.argmax(EvalFitness))
        #mating_pool.append(individual[argmax(EvalFitness)])
        current_member += 1
    print (EvalFitness)
    #print (individual)
    return mating_pool
#print (population)
TournamentSelection(4, 3, 0.3)
exit()
#print (NormFitness(Fitness, population, CitiesDist))
def Mutation(Genotype):
    #Genotype is in this case a permutation/route, in this case also called parents
    half = n//2
    swapA = np.random.randint(0, half)
    swapB = np.random.randint(half, n)
    if swapA == swapB:
        raise ValueError('A and B are equal')
    Genotype[swapA : swapB] = np.flip(Genotype[swapA : swapB], 0) #Not good 
    return Genotype, swapA, swapB 

#Make sure that population size is the same. Get rid of bad genotypes to store better genotypes

def PMX(ParentA, ParentB, start, end):
    ParentA = ParentA.tolist()
    ParentB = ParentB.tolist()
    offspring = [None]*len(ParentA)
    offspring[start : end] = ParentA[start : end]
    
    #A gene in ParentB[step] is occupied by a gene in offspring[step].
    #Take the gene occupied in offspring and locate that specific gene in ParentB.
    #Then place the gene with corresponding index as in ParentB in offspring     

    for step, gene in enumerate(ParentB[start : end]):
        step += start
        #gene from parentB
        if gene not in offspring:
            while offspring[step] != None: 
                step = ParentB.index(ParentA[step])
            offspring[step] = gene
    #The remaining genes in offspring which are empty gets filled with the genes from ParentB
    #offspring consist of a mix of ParentA and ParentB
    for step, gene in enumerate(offspring):
        if gene == None:
            offspring[step] = ParentB[step]
    return offspring

#PMX() returns floats, need ints for iteration in DistConvert(). intPMX is list of integers

#print (PMX(population[0], population[1], Mutation(Cities)[1], Mutation(Cities)[2]))
#intPMX = list(map(int, floats))
#intPMX = np.array(intPMX)
#print(Fitness(intPMX, CitiesDist), intPMX) 

#Running 20 times to measure performance
#MeanDist = np.mean(MeasureDist)    #20 is number of runs
#StdDist = np.std(MeasureDist)  
#BestDist = min(MeasureDist)
#WorstDist = max(MeasureDist)
endTime = time.time()
#Clock end and printing
#print ('Runtime for', n, 'cities is', 't =', (endTime - startTime),'s')
#print ('HillClimbing: Mean value for 20 runs and', n, 'cities is', MeanDist, 'with standard deviation', StdDist)
#print ('Best tour', BestDist, ';', 'Worst', WorstDist)


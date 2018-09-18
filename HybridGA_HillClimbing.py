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

n = 10   #number of cities
data = np.asarray(data)
CitiesDist = data[1 : n + 1, 0 : n] #Skipping "string", name of cities
Cities = np.arange(n)  
CitiesPermute = list(itertools.permutations(Cities))
CitiesPermute = np.asarray(CitiesPermute)
Cit = np.random.shuffle(CitiesPermute)

startTime = time.time()
 
def Fitness(Index, Dist):
	#Want to calculate the distance between n cities and return the total distance
	#Remember to calculate to distance from end to start at the end of loop    
	DistValues = np.zeros(n)
	for i in range(n - 1):
		start = Index[i]
		end = Index[i + 1]
		DistValues[i] = Dist[start, end]  
	DistValuesFinal = Dist[Index[n - 1], Index[0]]
	DistValues[n - 1] = DistValuesFinal
	return 1/np.sum(DistValues), DistValues

def NormFitness(func, Dist, populationSet):
	SumPopulation = 0
	FitnessOne = np.zeros(len(populationSet))
	for i in range(len(populationSet)):
		SumPopulation += func(populationSet[i], CitiesDist)[0] 
		FitnessOne[i] = func(populationSet[i], CitiesDist)[0]
	NormalizedFitness = FitnessOne/SumPopulation
	return NormalizedFitness

#population = CitiesPermute[1 : 10, :]
#print (NormFitness(Fitness, CitiesDist, population), np.sum(NormFitness(Fitness, CitiesDist, population)))

def TournamentSelection(lamb, k):  
	current_member = 1
	mating_pool = []
	while current_member <= lamb:
		individuals = random.sample(population, k)    
		EvalFitness = NormFitness(Fitness, CitiesDist, individuals)   	
		mating_pool.append(individuals[np.argmax(EvalFitness)])
		current_member += 1
	return mating_pool
	
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

def Mutation(Genotype):
	half = n//2
	swapA = np.random.randint(0, half)
	swapB = np.random.randint(half, n + 1)
	if swapA == swapB:
		raise ValueError('A and B are equal') 
	Genotype[swapA : swapB] = np.flip(Genotype[swapA : swapB], 0) #Must be ints 
	return Genotype, swapA, swapB 

def HillClimbing(func, n, CityOrder):
    #Chosing a random set of cities and minimizing distance 
    #Calculating distance from two cities 
    for i in range(10):
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

#Solver
population = CitiesPermute[1 : 300, :]
population = list(population)
Hillpopulation = []
for i in range(len(population)):
    Hillpopulation.append(HillClimbing(Fitness, n, population[i])[1])
#print (Hillpopulation)
means = []
dev = []
best = []
worst = []
runs = 20
for k in range(runs):	
	for t in range(10):
		candidate = TournamentSelection(50, 2)
		NumbOffspring = len(candidate)//2
		NextGen = np.zeros((NumbOffspring, n))
		NextGenMutate = np.zeros((NumbOffspring, n))

		for i in range(NumbOffspring):
			offspring = (PMX(candidate[i], candidate[i + 1], Mutation(candidate[i])[1], Mutation(candidate[i + 1])[2]))
			intoffspring = list(map(int, offspring))
			NextGen[i] = intoffspring
			NextGenMutate[i] = Mutation(intoffspring)[0]

		New_generation = np.append(NextGen, NextGenMutate, axis = 0)
		NewFitness = np.zeros((len(New_generation)))

		for i in range (len(New_generation)):
			IntNewGen = list(map(int, New_generation[i]))
			NewFitness[i] = Fitness(IntNewGen, CitiesDist)[0]
	best.append(1/NewFitness[np.argmax(NewFitness)])
	worst.append(1/NewFitness[np.argmin(NewFitness)])
	means.append(1/NewFitness.mean())
	dev.append(np.std(1/NewFitness)) #Not right

Numbrun = np.linspace(1,runs,runs)
#print(Numbrun)
#plt.plot(Numbrun, np.divide(1, means), '*')
#plt.show()

with open('best.txt','w') as f:
	for i in range(runs):
		f.write("best individual of run %d has length %f \n" % (i, best[i]))

with open('worst.txt','w') as f:
	for i in range(runs):
		f.write("worst individual of run %d has length %f \n" % (i, worst[i]))

with open('meansandstd.txt','w') as f:
	for i in range(runs):
		f.write("mean and std of an individual of run %d has value %f, %f \n" % (i, means[i], dev[i]))

endTime = time.time()
#Clock end and printing
#print('Runtime for', n, 'cities is', 't =', (endTime - startTime),'s')
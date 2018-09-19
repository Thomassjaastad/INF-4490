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

def TournamentSelection(lamb, k):  #k < lamb. lamb is a subset of population
	current_member = 1
	mating_pool = []
	while current_member <= lamb:
		individuals = random.sample(population, k)    
		EvalFitness = NormFitness(Fitness, CitiesDist, individuals)   		
		mating_pool.append(individuals[np.argmax(EvalFitness)])
		current_member += 1
	return mating_pool
	
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

def PMX_pair(ParentA, ParentB):
	half = n//2
	start = np.random.randint(0, half)
	end = np.random.randint(half, n + 1)
	return PMX(ParentA, ParentB, start, end), PMX(ParentB, ParentA, start, end) 

def Mutation(Genotype):
	half = n//2
	start = np.random.randint(0, half)
	end = np.random.randint(half, n + 1)
	if start == end:
		raise ValueError('A and B are equal') 
	Genotype[start : end] = np.flip(Genotype[start : end], 0) #Must be ints 
	return Genotype, start, end 

#Solver
means = []
dev = []
best = []
worst = []
runs = 1
generations = 10
bestMeans = []

Bestgen = np.zeros(generations)
PopulationSize = 20
subPopulation = 10
NewFitness = np.zeros(PopulationSize)
NextGenMutate = np.zeros((subPopulation, n))
NextGenMutate2 = np.zeros((subPopulation, n))

population = CitiesPermute[0 : PopulationSize, :]
#Needs to be updated!!!!
for k in range(runs):	
	population = CitiesPermute[0 : PopulationSize, :]
	population = list(population)
	candidate = TournamentSelection(subPopulation, 2)
	NewCandidates[0] = candidate
	for t in range(generations):
		for i in range(PopulationSize):
			#NewCandidates = candidate
			#print(NewCandidates[0])
			try:
				offspring1, offspring2 = PMX_pair(NewCandidates[i], NewCandidates[i + 1])
				intoffspring1 = list(map(int, offspring1))
				intoffspring2 = list(map(int, offspring2))
				NextGenMutate[i] = Mutation(intoffspring1)[0]
				NextGenMutate2[i] = Mutation(intoffspring2)[0]
			except IndexError:
				NextGenMutate[-1] = Mutation(intoffspring1)[0]
				NextGenMutate2[-1] = Mutation(intoffspring2)[0]
			New_generation = np.append(NextGenMutate, NextGenMutate2, axis = 0)
			IntNewGen = list(map(int, New_generation[i]))
			NewFitness[i] = Fitness(IntNewGen, CitiesDist)[0]
		
		NewCandidates[t + 1] = list(New_generation)	
		#Bestgen[t] = np.mean(1/np.max(NewFitness))
	best.append(1/NewFitness[np.argmax(NewFitness)])
	worst.append(1/NewFitness[np.argmin(NewFitness)])
	means.append(np.mean(1/NewFitness))
	dev.append(np.std(1/NewFitness)) 

#print (Bestgen)
genaxis = np.linspace(1, generations, generations)
#print (len(genaxis))
#plt.plot(genaxis, Bestgen , '*')
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

# CS422 - Data Analytics
# Jared McLean & Jared Schreiber
# 2/22/2017
"K-Nearest-Neighbors Classifier with test data"

import csv
import math
import operator
import copy
import sys
import os
import matplotlib.pyplot as plt

#offset values by a small amount to avoid division by 0
EPSI = sys.float_info.epsilon



#use os.urand as a better random number generator than the built in python one (specified to be a cryptographically sufficient random number generator)
def nextRand():
    #generate groups of random numbers for efficiency
    rand = os.urandom(1024)
    i = 0
    while True:
        if i == 1024:
            rand = os.urandom(1024)
            i = 0

        yield int.from_bytes(rand[i : i + 8], "big") / (2**64 - 1)
        i += 8


#functions

#main function
def main():
    "main function"
    #BUPA_NUM_ATTRIBUTES = 7
    #CAR_NUM_ATTRIBUTES = 7

    BUPA_CLASS_INDEX = 6
    CAR_CLASS_INDEX = 6
    IRIS_CLASS_INDEX = 4
    WINE_CLASS_INDEX = 0

    #MIN_K = 15
    #MAX_K = 15

    #numerical conversions for non-numerical car data
    CAR_VALUES = [
        {'vhigh' : 4.0,
         'high' : 3.0,
         'med' : 2.0,
         'low' : 1.0},
        {'vhigh' : 4.0,
         'high' : 3.0,
         'med' : 2.0,
         'low' : 1.0},
        {'5more' : 5.0},
        {'more' : 6.0},
        {'small' : 1.0,
         'med' : 2.0,
         'big' : 3.0},
        {'low' : 1.0,
         'med' : 2.0,
         'high' : 3.0},
        {'unacc' : 1.0,
         'acc' : 2.0,
         'good' : 3.0,
         'vgood' : 4.0}
        ]
    
    #load data sets

    test = loadcsv("iris.csv")
    

    test1 = loadcsv("bupa_data_testset.csv")
    test1.extend(loadcsv("bupa_data_trainset.csv"))

    test2 = loadcsv("car_data_trainset.csv")
    test2.extend(loadcsv("car_data_testset.csv"))
    test2 = convertData(test2, CAR_VALUES)

    test3 = loadcsv("wine.csv")

    #dataSets = [test, test1, test2, test3]
    #classIndex = [IRIS_CLASS_INDEX, BUPA_CLASS_INDEX, CAR_CLASS_INDEX, WINE_CLASS_INDEX]

    dataSets = [test]
    classIndex = [IRIS_CLASS_INDEX]
    #run each data set
    for s in range(len(dataSets)):
        dataSet = shiftDomain(stripClass(dataSets[s], classIndex[s]))
        #dataSet = stripClass(dataSet, 3)
        #dataSet = stripClass(dataSet, 2)
        #print(dataSet)

        k = 4
        n = 10
        setRange = []
        rand = nextRand()
        #find maxes and mins of each attribute
        for i in range(len(dataSet[0])):
            setRange.append((max(dataSet, key = lambda row: row[i])[i], min(dataSet, key = lambda row: row[i])[i]))

        
        iters = 1

        #kMeansOutcome = [0 for t in range(5)]
        #PSOOutcome = [0 for t in range(5)]
        #adaptiveOutcome = [0 for t in range(5)]
        #seededPSOOutcome = [0 for t in range(5)]
        #seededAdaptiveOutcome = [0 for t in range(5)]

        #capture individual results for standard dev calculation
        kMeansOutcome = [[0 for t in range(5)] for t in range(iters)]
        PSOOutcome = [[0 for t in range(5)] for t in range(iters)]
        adaptiveOutcome = [[0 for t in range(5)] for t in range(iters)]
        seededPSOOutcome = [[0 for t in range(5)] for t in range(iters)]
        seededAdaptiveOutcome = [[0 for t in range(5)] for t in range(iters)]

        #kMeansOutcomeMult = [[0 for t in range(4)] for t in range(iters)]
        #PSOOutcomeMult = [[0 for t in range(4)] for t in range(iters)]
        #adaptiveOutcomeMult = [[0 for t in range(4)] for t in range(iters)]
        #seededPSOOutcomeMult = [[0 for t in range(4)] for t in range(iters)]
        #seededAdaptiveOutcomeMult = [[0 for t in range(4)] for t in range(iters)]

        #initialPopulation = [initializeCentroids(setRange, k, rand) for t in range(n)]
        #runClusters(dataSet, k, n, initialPopulation, kMeansOutcome, PSOOutcome, adaptiveOutcome, seededPSOOutcome, seededAdaptiveOutcome, False, False, True)

        for i in range(iters):
            #initialize random populations of centroids
            #use same initial population for each clustering algorithm in order to provide a better comparison of performance (using first generated centroid population for k-means)
            initialPopulation = [initializeCentroids(setRange, k, rand) for t in range(n)]
            runClusters(dataSet, k, n, initialPopulation, kMeansOutcome[i], PSOOutcome[i], adaptiveOutcome[i], seededPSOOutcome[i], seededAdaptiveOutcome[i], False, True, False)
        
        
        kMMean = []
        PSOMean = []
        adaptiveMean = []
        seededPSOMean = []
        seededAdaptiveMean = []
        kMSD = []
        PSOSD = []
        adaptiveSD = []
        seededPSOSD = []
        seededAdaptiveSD = []
        #calculate mean and standard deviation of results for each clustering algorithm
        for i in range(4):
            kMMean.append(sum(kMeansOutcome[j][i] for j in range(iters)) / iters)
            kMSD.append(math.sqrt(sum((kMeansOutcome[j][i] - kMMean[i])**2 for j in range(iters)) / iters))

            PSOMean.append(sum(PSOOutcome[j][i] for j in range(iters)) / iters)
            PSOSD.append(math.sqrt(sum((PSOOutcome[j][i] - PSOMean[i])**2 for j in range(iters)) / iters))

            adaptiveMean.append(sum(adaptiveOutcome[j][i] for j in range(iters)) / iters)
            adaptiveSD.append(math.sqrt(sum((adaptiveOutcome[j][i] - adaptiveMean[i])**2 for j in range(iters)) / iters))

            seededPSOMean.append(sum(seededPSOOutcome[j][i] for j in range(iters)) / iters)
            seededPSOSD.append(math.sqrt(sum((seededPSOOutcome[j][i] - seededPSOMean[i])**2 for j in range(iters)) / iters))

            seededAdaptiveMean.append(sum(seededAdaptiveOutcome[j][i] for j in range(iters)) / iters)
            seededAdaptiveSD.append(math.sqrt(sum((seededAdaptiveOutcome[j][i] - seededAdaptiveMean[i])**2 for j in range(iters)) / iters))

        #print results
        print("K Means:")
        print("Average Intradistance: " + str(kMMean[0]) + " +/- " + str(kMSD[0]))
        print("Average Interdistance: " + str(kMMean[1]) + " +/- " + str(kMSD[1]))
        print("Average Convergence Iteration: " + str(kMMean[2]) + " +/- " + str(kMSD[2]))
        print("Average Best Fitness: " + str(kMMean[3]) + " +/- " + str(kMSD[3]))
        print()
        print("PSO:")
        print("Average Intradistance: " + str(PSOMean[0]) + " +/- " + str(PSOSD[0]))
        print("Average Interdistance: " + str(PSOMean[1]) + " +/- " + str(PSOSD[1]))
        print("Average Convergence Iteration: " + str(PSOMean[2]) + " +/- " + str(PSOSD[2]))
        print("Average Best Fitness: " + str(PSOMean[3]) + " +/- " + str(PSOSD[3]))
        print()
        print("Adaptive PSO:")
        print("Average Intradistance: " + str(adaptiveMean[0]) + " +/- " + str(adaptiveSD[0]))
        print("Average Interdistance: " + str(adaptiveMean[1]) + " +/- " + str(adaptiveSD[1]))
        print("Average Convergence Iteration: " + str(adaptiveMean[2]) + " +/- " + str(adaptiveSD[2]))
        print("Average Best Fitness: " + str(adaptiveMean[3]) + " +/- " + str(adaptiveSD[3]))
        print()
        print("Seeded PSO:")
        print("Average Intradistance: " + str(seededPSOMean[0]) + " +/- " + str(seededPSOSD[0]))
        print("Average Interdistance: " + str(seededPSOMean[1]) + " +/- " + str(seededPSOSD[1]))
        print("Average Convergence Iteration: " + str(seededPSOMean[2]) + " +/- " + str(seededPSOSD[2]))
        print("Average Best Fitness: " + str(seededPSOMean[3]) + " +/- " + str(seededPSOSD[3]))
        print()
        print("Seeded Adaptive PSO:")
        print("Average Intradistance: " + str(seededAdaptiveMean[0]) + " +/- " + str(seededAdaptiveSD[0]))
        print("Average Interdistance: " + str(seededAdaptiveMean[1]) + " +/- " + str(seededAdaptiveSD[1]))
        print("Average Convergence Iteration: " + str(seededAdaptiveMean[2]) + " +/- " + str(seededAdaptiveSD[2]))
        print("Average Best Fitness: " + str(seededAdaptiveMean[3]) + " +/- " + str(seededAdaptiveSD[3]))
        print()
        print("-----------------------------------------------------------")
        print()
    return

#run the clustering algorithms on the given data set, saving results to the specified outcome arrays
def runClusters(dataSet, k, n, initialPopulation, kMeansOutcome, PSOOutcome, adaptiveOutcome, seededPSOOutcome, seededAdaptiveOutcome, printInfo, printGraphs, showClusters):
    #set of colors for printing graphs
    colors = ['red', 'blue', 'green', 'purple']
    #run each clustering algorithm for the given dataset
    centroids = runKMeans(dataSet, k, initialPopulation[0], printGraphs, kMeansOutcome)
    #print individual iteration outcomes if printInfo is true
    if printInfo:
        print("K Means:")
        print("Intradistance: " + str(kMeansOutcome[0]))
        print("Interdistance: " + str(kMeansOutcome[1]))
        print("Convergence Iteration: " + str(kMeansOutcome[2]))
        print("Best Fitness: " + str(kMeansOutcome[3]))
        print()
    if showClusters:
        for i in range(len(dataSet)):
            graphPos = 321
            #graph clusters for each combination of at most four dimensions
            for j in range(min(len(dataSet[0]), 4)):
                for t in range(j + 1, len(dataSet[0])):
                    plt.subplot(graphPos).scatter(dataSet[i][j], dataSet[i][t], c = colors[kMeansOutcome[4][i]])
                    for cent in range(k):
                        plt.subplot(graphPos).scatter(centroids[cent][j], centroids[cent][t], marker = '*', c = colors[cent], edgecolor = 'black', s = 200)
                    graphPos += 1
        
                
        plt.show()
    centroids = runPSOCluster(dataSet, k, n, initialPopulation, printGraphs, PSOOutcome)
    if printInfo:
        print("PSO:")
        print("Intradistance: " + str(PSOOutcome[0]))
        print("Interdistance: " + str(PSOOutcome[1]))
        print("Convergence Iteration: " + str(PSOOutcome[2]))
        print("Best Fitness: " + str(PSOOutcome[3]))
        print()
    if showClusters:
        for i in range(len(dataSet)):
            graphPos = 321
            for j in range(min(len(dataSet[0]), 4)):
                for t in range(j + 1, len(dataSet[0])):
                    plt.subplot(graphPos).scatter(dataSet[i][j], dataSet[i][t], c = colors[PSOOutcome[4][i]])
                    for cent in range(k):
                        plt.subplot(graphPos).scatter(centroids[cent][j], centroids[cent][t], marker = '*', c = colors[cent], edgecolor = 'black', s = 200)
                    graphPos += 1
        
                
        plt.show()
    centroids = runAdaptivePSOCluster(dataSet, k, n, initialPopulation, printGraphs, adaptiveOutcome)
    if printInfo:
        print("Adaptive PSO:")
        print("Intradistance: " + str(adaptiveOutcome[0]))
        print("Interdistance: " + str(adaptiveOutcome[1]))
        print("Convergence Iteration: " + str(adaptiveOutcome[2]))
        print("Best Fitness: " + str(adaptiveOutcome[3]))
        print()
    if showClusters:
        for i in range(len(dataSet)):
            graphPos = 321
            for j in range(min(len(dataSet[0]), 4)):
                for t in range(j + 1, len(dataSet[0])):
                    plt.subplot(graphPos).scatter(dataSet[i][j], dataSet[i][t], c = colors[adaptiveOutcome[4][i]])
                    for cent in range(k):
                        plt.subplot(graphPos).scatter(centroids[cent][j], centroids[cent][t], marker = '*', c = colors[cent], edgecolor = 'black', s = 200)
                    graphPos += 1
        
                
        plt.show()
    centroids = runKMSeededPSOCluster(dataSet, k, n, initialPopulation, printGraphs, seededPSOOutcome)
    if printInfo:
        print("Seeded PSO:")
        print("Intradistance: " + str(seededPSOOutcome[0]))
        print("Interdistance: " + str(seededPSOOutcome[1]))
        print("Convergence Iteration: " + str(seededPSOOutcome[2]))
        print("Best Fitness: " + str(seededPSOOutcome[3]))
        print()
    if showClusters:
        for i in range(len(dataSet)):
            graphPos = 321
            for j in range(min(len(dataSet[0]), 4)):
                for t in range(j + 1, len(dataSet[0])):
                    plt.subplot(graphPos).scatter(dataSet[i][j], dataSet[i][t], c = colors[seededPSOOutcome[4][i]])
                    for cent in range(k):
                        plt.subplot(graphPos).scatter(centroids[cent][j], centroids[cent][t], marker = '*', c = colors[cent], edgecolor = 'black', s = 200)
                    graphPos += 1
        
                
        plt.show()
    centroids = runKMSeededAdaptivePSOCluster(dataSet, k, n, initialPopulation, printGraphs, seededAdaptiveOutcome)
    if printInfo:
        print("Seeded Adaptive PSO:")
        print("Intradistance: " + str(seededAdaptiveOutcome[0]))
        print("Interdistance: " + str(seededAdaptiveOutcome[1]))
        print("Convergence Iteration: " + str(seededAdaptiveOutcome[2]))
        print("Best Fitness: " + str(seededAdaptiveOutcome[3]))
    if showClusters:
        for i in range(len(dataSet)):
            graphPos = 321
            for j in range(min(len(dataSet[0]), 4)):
                for t in range(j + 1, len(dataSet[0])):
                    plt.subplot(graphPos).scatter(dataSet[i][j], dataSet[i][t], c = colors[seededAdaptiveOutcome[4][i]])
                    for cent in range(k):
                        plt.subplot(graphPos).scatter(centroids[cent][j], centroids[cent][t], marker = '*', c = colors[cent], edgecolor = 'black', s = 200)
                    graphPos += 1
        
                
        plt.show()


#convert data to numerical data based on provided value conversion dictionary
#for classification to work properly classifiers must be
def convertData(dataSet, valueIndex):
    set = copy.deepcopy(dataSet)
    numAttributes = len(dataSet[0])
    #convert data
    for element in set:
        for i in range(numAttributes):
            if element[i] in valueIndex[i]:
                element[i] = valueIndex[i][element[i]]
    return set


#seperate out classification attribute from data set
def stripClass(dataSet, classIndex):
    dset = copy.deepcopy(dataSet)
    for item in dset:
        item.pop(classIndex)
    return dset

#Euclidean Distance Function 2 vectors
def euclidianDist(vector1, vector2):
    #assumes both vectors same dimensionality
    dist = 0
    length = len(vector1)
    for i in range(length):
        dist += pow((vector1[i] - vector2[i]), 2)
    return math.sqrt(dist)

#runs a k-means clustering on the given dataset, returning the centroid positions and storing results in the outcome array if provided
def runKMeans(dataSet, k, initialPopulation, showGraph, outcome = None):
    numAttributes = len(dataSet[0])
    numRecords = len(dataSet)

    distances = [0 for t in range(numRecords)]
    clusters = [0 for t in range(numRecords)]

    centroids = copy.deepcopy(initialPopulation);

    totalFitnessesTime = []
    sumF = 0
    sumFPrev = -1
    #store last 2, sometimes gets stuck bouncing back and forth
    sumFPrevPrev = -2
    iter = 0
    #stop if no change or reached 50 iterations
    while sumF != sumFPrev and sumF != sumFPrevPrev and iter < 50:
        #determine which points are in which clusters by nearest centroid
        for i in range(numRecords):
            nearestCentroid = 0
            nearestDistance = math.inf
            for j in range(k):
                dist = euclidianDist(dataSet[i], centroids[j])
                if dist < nearestDistance:
                    nearestDistance = dist
                    nearestCentroid = j
            distances[i] = nearestDistance
            clusters[i] = nearestCentroid
        #store fitness results
        sumFPrevPrev = sumFPrev
        sumFPrev = sumF
        sumF = clusteringFitness(distances, clusters, k)
        totalFitnessesTime.append(sumF)

        
        #print(sumF)
        #update centroid positions
        centroids = updateKMCentroids(clusters, dataSet, numRecords, numAttributes, k, centroids)

        iter += 1
    #store results in the outcome array
    if outcome != None:
        #calculate average intercluster distance (using average distance between centroids)
        interdist = 0
        combinations = 0
        usedCentroids = []
        #calculate average intercluster distance by taking the average of the distance between each centroid
        #disclude unused centroids from this computation
        for i in clusters:
            if i not in usedCentroids:
                usedCentroids.append(i)
        for i in range(len(usedCentroids)):
            for j in range(i + 1, len(usedCentroids)):
                combinations += 1
                interdist += euclidianDist(centroids[usedCentroids[i]], centroids[usedCentroids[j]])
        if combinations != 0:
            interdist /= combinations
        else:
            interdist = 0
        #calculate average intracluster distance (using average distance from centroid to data points)
        intradist = 0
        for i in range(numRecords):
            intradist += euclidianDist(centroids[clusters[i]], dataSet[i])
        intradist /= numRecords
        outcome[0] += intradist
        outcome[1] += interdist
        outcome[2] += iter
        outcome[3] += sumF
        outcome[4] = clusters
    #print graph of convergence rates if specified to do so
    if showGraph:
        #print(intradist)
        #print(interdist)
        #print(iter)
        #print(sumF)
        plt.plot(totalFitnessesTime)
        plt.show()

    return centroids


# it is important to note that, for PSO based algorithms, a particle is a population of centroids, where the centroids are treated as subparticles
#updates to particles are accomplished by updating each subparticle individually, though fitnesses are determined over the particle as a whole

#runs a k-means seeded variation of the adaptive algorithm on the given dataset and stores the results in the outcome array
def runKMSeededAdaptivePSOCluster(dataSet, k, n, initialPopulation, showGraphs, outcome):
    rand = nextRand()
    numAttributes = len(dataSet[0])
    numRecords = len(dataSet)
    maxC = 5
    #initialize c values between 0 and maximum value
    c = [[(next(rand) * maxC, next(rand) * maxC) for t in range(k)] for t in range(n)]
    v = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbest = [[0 for t in range(numAttributes)] for t in range(k)]
    pbest = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbestFit = math.inf
    pbestFit = [math.inf for t in range(n)]
    distances = [[0 for t in range(numRecords)] for t in range(n)]
    clusters = [[0 for t in range(numRecords)] for t in range(n)]
    gbestClusters = [0 for t in range(numRecords)]
    

    present = copy.deepcopy(initialPopulation);
    #seed the first population using k-means
    present[0] = runKMeans(dataSet, k, initialPopulation[0], False)

    totalFitnessesTime = []
    sumF = 0
    sumFPrev = -1
    iter = 0
    threshold = .02
    gens = 5
    stop = 0
    #stop when change in fitness is less than threshold for the specified number of generations or after 300 iterations
    while iter < 300:
        if abs(sumF - sumFPrev) <= sumFPrev * threshold:
            stop += 1
            if stop >= gens:
                break
        else:
            stop = 0


        swarmFitness = []
        #compute cluster membership based on nearest centroids
        for particle in range(n):
            for i in range(numRecords):
                nearestCentroid = 0
                nearestDistance = math.inf
                for j in range(k):
                    dist = euclidianDist(dataSet[i], present[particle][j])
                    if dist < nearestDistance:
                        nearestDistance = dist
                        nearestCentroid = j
                distances[particle][i] = nearestDistance
                clusters[particle][i] = nearestCentroid
            #compute fitness of each centroid population
            swarmFitness.append(clusteringFitness(distances[particle], clusters[particle], k))

        #store previous fitness and average fitness of the clusterings
        sumFPrev = sumF
        sumF = 0
        sumF = sum(swarmFitness) / n

        totalFitnessesTime.append(sumF)
        #evaluate whether the current particle fitnesses exceed their personal best or the global best, and update values accordingly
        for i in range(n):
            if swarmFitness[i] < gbestFit:
                gbest = copy.deepcopy(present[i])
                gbestFit = swarmFitness[i]
                gbestClusters = copy.deepcopy(clusters[i])
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])
            elif swarmFitness[i] < pbestFit[i]:
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])
        #update particle positions using PSO procedures (updates the position of each centroid in each centroid population)
        updateParticles(present, pbest, gbest, v, numAttributes, k, n, c, rand)

        for i in range(n):
            c[i] = getCVals(c[i], v[i], k, rand, maxC)

        iter += 1

    #compute stats for the global best
    interdist = 0
    combinations = 0
    usedCentroids = []
    for i in gbestClusters:
        if i not in usedCentroids:
            usedCentroids.append(i)
    for i in range(len(usedCentroids)):
        for j in range(i + 1, len(usedCentroids)):
            combinations += 1
            interdist += euclidianDist(gbest[usedCentroids[i]], gbest[usedCentroids[j]])
    if combinations != 0:
        interdist /= combinations
    else:
        interdist = 0
    #calculate average intracluster distance (using average distance from centroid to data points)
    intradist = 0
    for i in range(numRecords):
        intradist += euclidianDist(gbest[gbestClusters[i]], dataSet[i])
    #return global best results
    intradist /= numRecords
    outcome[0] += intradist
    outcome[1] += interdist
    outcome[2] += iter
    outcome[3] += gbestFit
    outcome[4] = gbestClusters

    #print graph of convergence rates if specified to do so
    #print(iter)
    #print(sumF)
    if showGraphs:
        plt.plot(totalFitnessesTime)
        plt.show()

    return gbest

#runs a k-means seeded variation of the PSO clustering algorithm on the given dataset and stores the results in the outcome array
#code documentation similar to corresponding code sections in runKMSeededAdaptivePSOCluster procedure
def runKMSeededPSOCluster(dataSet, k, n, initialPopulation, showGraphs, outcome):
    rand = nextRand()
    numAttributes = len(dataSet[0])
    numRecords = len(dataSet)
    #use c1=c2=1.49 for c values
    c = [[(1.49, 1.49) for t in range(k)] for t in range(n)]
    v = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbest = [[0 for t in range(numAttributes)] for t in range(k)]
    pbest = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbestFit = math.inf
    pbestFit = [math.inf for t in range(n)]
    distances = [[0 for t in range(numRecords)] for t in range(n)]
    clusters = [[0 for t in range(numRecords)] for t in range(n)]
    gbestClusters = [0 for t in range(numRecords)]
    
    present = copy.deepcopy(initialPopulation);

    present[0] = runKMeans(dataSet, k, initialPopulation[0], False)

    totalFitnessesTime = []
    sumF = 0
    sumFPrev = -1
    iter = 0
    threshold = .02
    gens = 5
    stop = 0
    #stop when change in fitness is less than threshold for the specified number of generations or after 300 iterations
    while iter < 300:
        if abs(sumF - sumFPrev) <= sumFPrev * threshold:
            stop += 1
            if stop >= gens:
                break
        else:
            stop = 0

        swarmFitness = []

        for particle in range(n):
            for i in range(numRecords):
                nearestCentroid = 0
                nearestDistance = math.inf
                for j in range(k):
                    dist = euclidianDist(dataSet[i], present[particle][j])
                    if dist < nearestDistance:
                        nearestDistance = dist
                        nearestCentroid = j
                distances[particle][i] = nearestDistance
                clusters[particle][i] = nearestCentroid

            swarmFitness.append(clusteringFitness(distances[particle], clusters[particle], k))

        
        sumFPrev = sumF
        sumF = 0
        sumF = sum(swarmFitness) / n

        totalFitnessesTime.append(sumF)

        for i in range(n):
            if swarmFitness[i] < gbestFit:
                gbest = copy.deepcopy(present[i])
                gbestFit = swarmFitness[i]
                gbestClusters = copy.deepcopy(clusters[i])
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])
            elif swarmFitness[i] < pbestFit[i]:
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])

        updateParticles(present, pbest, gbest, v, numAttributes, k, n, c, rand)

        iter += 1

    interdist = 0
    combinations = 0
    usedCentroids = []
    for i in gbestClusters:
        if i not in usedCentroids:
            usedCentroids.append(i)
    for i in range(len(usedCentroids)):
        for j in range(i + 1, len(usedCentroids)):
            combinations += 1
            interdist += euclidianDist(gbest[usedCentroids[i]], gbest[usedCentroids[j]])
    if combinations != 0:
        interdist /= combinations
    else:
        interdist = 0
    #calculate average intracluster distance (using average distance from centroid to data points)
    intradist = 0
    for i in range(numRecords):
        intradist += euclidianDist(gbest[gbestClusters[i]], dataSet[i])
    intradist /= numRecords
    outcome[0] += intradist
    outcome[1] += interdist
    outcome[2] += iter
    outcome[3] += gbestFit
    outcome[4] = gbestClusters
    
    if showGraphs:
        plt.plot(totalFitnessesTime)
        plt.show()

    return gbest

#update the centroids for a k-means clustering
def updateKMCentroids(clusters, dataSet, numRecords, numAttributes, k, currentCentroids):
    clusterCenter = [[0 for t in range(numAttributes)] for t in range(k)]
    numInCluster = [0 for t in range(k)]
    #determine the total magnitude of each vector dimension for each cluster and the number of points in the clusters
    for i in range(numRecords):
        for j in range(numAttributes):
            clusterCenter[clusters[i]][j] += dataSet[i][j]
        numInCluster[clusters[i]] += 1
    #determine clusters centerpoint by averaging the vector magnitudes based on the total magnitudes and number of points in the clusters
    for i in range(k):
        for j in range(numAttributes):
            if numInCluster[i] == 0:
                #if no items in cluster leave at current location
                clusterCenter[i][j] = currentCentroids[i][j]
            else:
                #print(numInCluster)
                clusterCenter[i][j] /= numInCluster[i]
    #return the new set of centroids
    return clusterCenter
    
#get new c values using genetic algorithms
def getCVals(cVals, v, k, rand, maxC):
    #one percent chance of mutation
    mutationChance = .01
    fitness = [0 for t in range(k)]
    newCs = []
    for i in range(k):
        fitness[i] = getGAFitness(v[i])
    parents = selectParents(fitness, rand, k)
    #if there is an odd number of parents, allow most fit individual in set of parents to move on without crossover (may be mutated)
    if k % 2 == 1:
        bestFit = math.inf
        fitParentIndex = 0
        for i in range(k):
            if fitness[parents[i]] < bestFit:
                bestFit = fitness[parents[i]]
                fitParent = i
        #add best parent to set of new c values and replace by the last element (which will be skipped during crossover due to odd number of parents)
        newCs.append((cVals[parents[fitParent]][0], cVals[parents[fitParent]][1]))
        parents[fitParent] = parents[k - 1]

    for i in range(0, k - 1, 2):
        cross = int(round(next(rand) * 3))

        #print(len(parents))
        #print(i)

        #perform crossover on first c value
        if cross == 0:
            newCs.append((crossover(cVals[parents[i]][0], cVals[parents[i + 1]][0], rand), cVals[parents[i]][1]))
            newCs.append((crossover(cVals[parents[i]][0], cVals[parents[i + 1]][0], rand), cVals[parents[i + 1]][1]))
        #perform crossover on second c value
        elif cross == 1:
            newCs.append((cVals[parents[i]][0], crossover(cVals[parents[i]][1], cVals[parents[i + 1]][1], rand)))
            newCs.append((cVals[parents[i + 1]][0], crossover(cVals[parents[i]][1], cVals[parents[i + 1]][1], rand)))
        #perform crossover on both c values
        elif cross == 2:
            newCs.append((crossover(cVals[parents[i]][0], cVals[parents[i + 1]][0], rand), crossover(cVals[parents[i]][1], cVals[parents[i + 1]][1], rand)))
            newCs.append((crossover(cVals[parents[i]][0], cVals[parents[i + 1]][0], rand), crossover(cVals[parents[i]][1], cVals[parents[i + 1]][1], rand)))
        #if 3 perform no crossover
        else:
            newCs.append((cVals[parents[i]][0], cVals[parents[i]][1]))
            newCs.append((cVals[parents[i + 1]][0], cVals[parents[i + 1]][1]))

    for i in range(k):
        #check if mutate first c value
        if next(rand) < mutationChance:
            #print("mutate")
            #mutate by assigning new random c value in valid range
            newCs[i] = (next(rand) * maxC, newCs[i][1])
        #check if mutate second c value
        if next(rand) < mutationChance:
            #print("mutate")
            #mutate by assigning new random c value in valid range
            newCs[i] = (newCs[i][0], next(rand) * maxC)

    return newCs


#for crossover take a random point between the two parents c values
def crossover(c1, c2, rand):
    if c1 < c2:
        return next(rand) * (c2 - c1) + c1
    else:
        return next(rand) * (c1 - c2) + c2



#select parents using tournament selection
def selectParents(fitness, rand, k):
    select = 3
    #tourney = [0 for t in range(select)]
    parents = []
    for i in range(k):
        bestFit = math.inf
        winner = 0
        for j in range(select):
            #generate random number between 0 and k and round to nearest int
            selected = int(round(next(rand) * (k - 1)))
            if fitness[selected] < bestFit:
                bestFit = fitness[selected]
                winner = selected
        parents.append(winner)
    return parents




#magnitude of velocity vector, attempting to minimize
def getGAFitness(v):
    mag = 0
    for item in v:
        mag += pow(item, 2)
    return math.sqrt(mag)



#runs a adaptive variation of the PSO clustering algorithm on the given dataset and stores the results in the outcome array
#code documentation similar to corresponding code sections in runKMSeededAdaptivePSOCluster procedure
def runAdaptivePSOCluster(dataSet, k, n, initialPopulation, showGraphs, outcome):
    rand = nextRand()
    numAttributes = len(dataSet[0])
    numRecords = len(dataSet)
    maxC = 5
    #initialize c values between 2 and maximum value
    c = [[(next(rand) * maxC, next(rand) * maxC) for t in range(k)] for t in range(n)]
    v = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbest = [[0 for t in range(numAttributes)] for t in range(k)]
    pbest = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbestFit = math.inf
    pbestFit = [math.inf for t in range(n)]
    distances = [[0 for t in range(numRecords)] for t in range(n)]
    clusters = [[0 for t in range(numRecords)] for t in range(n)]
    gbestClusters = [0 for t in range(numRecords)]
    

    present = copy.deepcopy(initialPopulation);
    

    totalFitnessesTime = []
    sumF = 0
    sumFPrev = -1
    iter = 0
    threshold = .02
    gens = 5
    stop = 0
    #stop when change in fitness is less than threshold for the specified number of generations or after 300 iterations
    while iter < 300:
        if abs(sumF - sumFPrev) <= sumFPrev * threshold:
            stop += 1
            if stop >= gens:
                break
        else:
            stop = 0

        swarmFitness = []

        for particle in range(n):
            for i in range(numRecords):
                nearestCentroid = 0
                nearestDistance = math.inf
                for j in range(k):
                    dist = euclidianDist(dataSet[i], present[particle][j])
                    if dist < nearestDistance:
                        nearestDistance = dist
                        nearestCentroid = j
                distances[particle][i] = nearestDistance
                clusters[particle][i] = nearestCentroid

            swarmFitness.append(clusteringFitness(distances[particle], clusters[particle], k))

        
        sumFPrev = sumF
        sumF = 0

        sumF = sum(swarmFitness) / n

        totalFitnessesTime.append(sumF)

        for i in range(n):
            if swarmFitness[i] < gbestFit:
                gbest = copy.deepcopy(present[i])
                gbestFit = swarmFitness[i]
                gbestClusters = copy.deepcopy(clusters[i])
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])
            elif swarmFitness[i] < pbestFit[i]:
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])

        updateParticles(present, pbest, gbest, v, numAttributes, k, n, c, rand)

        for i in range(n):
            c[i] = getCVals(c[i], v[i], k, rand, maxC)

        iter += 1

    interdist = 0
    combinations = 0
    usedCentroids = []
    for i in gbestClusters:
        if i not in usedCentroids:
            usedCentroids.append(i)
    for i in range(len(usedCentroids)):
        for j in range(i + 1, len(usedCentroids)):
            combinations += 1
            interdist += euclidianDist(gbest[usedCentroids[i]], gbest[usedCentroids[j]])
    if combinations != 0:
        interdist /= combinations
    else:
        interdist = 0
    #calculate average intracluster distance (using average distance from centroid to data points)
    intradist = 0
    for i in range(numRecords):
        intradist += euclidianDist(gbest[gbestClusters[i]], dataSet[i])
    intradist /= numRecords
    outcome[0] += intradist
    outcome[1] += interdist
    outcome[2] += iter
    outcome[3] += gbestFit
    outcome[4] = gbestClusters

    #print(iter)
    #print(sumF)
    if showGraphs:
        plt.plot(totalFitnessesTime)
        plt.show()

    return gbest

#update particles using particle swarm update equations
def updateParticles(present, pbest, gbest, v, numAttributes, k, n, c, rand):
    #update each particle
    for particle in range(n):
        #update each centroid in the particle
        for i in range(k):
            #if c1 + c2 value greater than or equal to 4, use constriction coefficient
            if c[particle][i][0] + c[particle][i][1] >= 4:
                x = getConstrictionCoef(c[particle][i][0], c[particle][i][1])
                w = 1
            #otherwise use an inertial weight defined as half of the average of the two c values
            #note that w = x = 1 where c1 + c2 = 4
            else:
                x = 1
                w = (c[particle][i][0] + c[particle][i][1]) / 4
            #run PSO update equations
            for j in range(numAttributes):
                v[particle][i][j] = x * (w * v[particle][i][j] + c[particle][i][0] * next(rand) * (pbest[particle][i][j] - present[particle][i][j]) + c[particle][i][1] * next(rand) * (gbest[i][j] - present[particle][i][j]))
                present[particle][i][j] += v[particle][i][j]
    return


#runs a PSO clustering algorithm on the given dataset and stores the results in the outcome array
#code documentation similar to corresponding code sections in runKMSeededAdaptivePSOCluster procedure
def runPSOCluster(dataSet, k, n, initialPopulation, showGraphs, outcome):
    rand = nextRand()
    numAttributes = len(dataSet[0])
    numRecords = len(dataSet)
    #use c1 = c2 = 1.49
    c = [[(1.49, 1.49) for t in range(k)] for t in range(n)]
    v = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbest = [[0 for t in range(numAttributes)] for t in range(k)]
    pbest = [[[0 for t in range(numAttributes)] for t in range(k)] for t in range(n)]
    gbestFit = math.inf
    pbestFit = [math.inf for t in range(n)]
    distances = [[0 for t in range(numRecords)] for t in range(n)]
    clusters = [[0 for t in range(numRecords)] for t in range(n)]
    gbestClusters = [0 for t in range(numRecords)]
    
    

    present = copy.deepcopy(initialPopulation);
    

    totalFitnessesTime = []
    sumF = 0
    sumFPrev = -1
    iter = 0
    threshold = .02
    gens = 5
    stop = 0
    #stop when change in fitness is less than threshold for the specified number of generations or after 300 iterations
    while iter < 300:
        if abs(sumF - sumFPrev) <= sumFPrev * threshold:
            stop += 1
            if stop >= gens:
                break
        else:
            stop = 0



        swarmFitness = []

        for particle in range(n):
            for i in range(numRecords):
                nearestCentroid = 0
                nearestDistance = math.inf
                for j in range(k):
                    dist = euclidianDist(dataSet[i], present[particle][j])
                    if dist < nearestDistance:
                        nearestDistance = dist
                        nearestCentroid = j
                distances[particle][i] = nearestDistance
                clusters[particle][i] = nearestCentroid

            swarmFitness.append(clusteringFitness(distances[particle], clusters[particle], k))

        
        sumFPrev = sumF
        sumF = 0
        sumF = sum(swarmFitness) / n

        totalFitnessesTime.append(sumF)
        #print(gbest)

        for i in range(n):
            if swarmFitness[i] < gbestFit:
                gbest = copy.deepcopy(present[i])
                gbestFit = swarmFitness[i]
                gbestClusters = copy.deepcopy(clusters[i])
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])
            elif swarmFitness[i] < pbestFit[i]:
                pbestFit[i] = swarmFitness[i]
                pbest[i] = copy.deepcopy(present[i])

        updateParticles(present, pbest, gbest, v, numAttributes, k, n, c, rand)

        iter += 1

    interdist = 0
    combinations = 0
    usedCentroids = []
    for i in gbestClusters:
        if i not in usedCentroids:
            usedCentroids.append(i)
    for i in range(len(usedCentroids)):
        for j in range(i + 1, len(usedCentroids)):
            combinations += 1
            interdist += euclidianDist(gbest[usedCentroids[i]], gbest[usedCentroids[j]])
    if combinations != 0:
        interdist /= combinations
    else:
        interdist = 0
    #calculate average intracluster distance (using average distance from centroid to data points)
    intradist = 0
    for i in range(numRecords):
        intradist += euclidianDist(gbest[gbestClusters[i]], dataSet[i])
    intradist /= numRecords
    outcome[0] += intradist
    outcome[1] += interdist
    outcome[2] += iter
    outcome[3] += gbestFit
    outcome[4] = gbestClusters

    #print(iter)
    #print(sumF)
    
    if showGraphs:
        plt.plot(totalFitnessesTime)
        plt.show()

    return gbest


#using the average of the fitness of each centroid for overall clustering fitness
#where the fitness of each centroid is determined as the average sum of square distances to the points in the centroids cluster
#determine fitness of the given clustering
def clusteringFitness(distances, clusters, k):
    fitness = [0 for t in range(k)]
    numInCluster = [0 for t in range(k)]
    usedCentroids = k
    for i in range(len(distances)):
        #add square distance from its centroid to the fitness for the centroid whos cluster the current distance belongs to
        fitness[clusters[i]] += distances[i]**2
        numInCluster[clusters[i]] += 1
    #average the SSD for each centroid to determine fitness
    for i in range(k):
        #if centroid has no elements remove from computation (0 fitness and not counted in number of centroids)
        if numInCluster[i] == 0:
            fitness[i] = 0
            usedCentroids -= 1
        else:
            fitness[i] /= numInCluster[i]
    #return the average fitness of each centroid
    return sum(fitness) / usedCentroids

#initialize k centroids at random
def initializeCentroids(ranges, k, rand):
    initialPop = []
    for i in range(k):
        particle = []
        for r in ranges:
            particle.append(next(rand) * (r[0] - r[1]) + r[1])
        initialPop.append(particle)
    return initialPop


#determin constriction coefficient based on the constriction coefficient equation
def getConstrictionCoef(c1, c2):
    c = c1 + c2
    return 2.0 / (c - 2 + math.sqrt(c**2 - 4 * c))



#shift data set domain between -1 and 1
def shiftDomain(set):
    numAttributes = len(set[0])
    numRows = len(set)
    setRange = []
    norm = [[] for _ in range(numRows)]
    #find maxes and mins of each attribute
    for i in range(numAttributes):
        setRange.append((max(set, key = lambda row: row[i])[i], min(set, key = lambda row: row[i])[i]))
        for j in range(numRows):
            #shift to set range
            norm[j].append(2 * ((set[j][i] - setRange[i][1]) / (setRange[i][0] - setRange[i][1])) - 1)
    return norm



#load a CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    test = 0
    #open file for reading
    with open(fname, 'r') as fvar:
        reader = csv.reader(fvar)
        #read each row into array and attempt to convert attributes to numbers if numerical values
        for row in reader:
            row[:] = [(float(item)) if testNum(item) else item for item in row]
            rlist.append(row)
    return rlist

#test if given value is a number
def testNum(f):
    try:
        float(f)
        return True
    except ValueError:
        return False


#main block
main()

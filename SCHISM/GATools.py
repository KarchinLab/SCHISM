#! /usr/bin/env python
'''
prototype implementation of basic GA functionality
'''
import sys
import time

import numpy as np
from collections import Counter
from itertools import combinations

from Tree import Node

class GA(object):
    #------------------------------------------------------------------#
    def __init__(self, opts):
        
        self.objects = []
        self.generationSize = opts['generationSize']
        self.generationCount = opts['generationCount']
   
        # the fraction of objects in each generation that are
        # randomly generated from scratch and are not descendants
        # of objects so far.
        self.randomObjectFraction = opts['randomObjectFraction']
        
        # mutation probability
        self.Pm = opts['Pm']
        
        # crossover probability
        self.Pc = opts['Pc']

        self.currentGeneration = 0
        self.populationSize = 0

        # running lists corresponding to generation max and median fitness
        # and population max and median fitness after each generation
        self.generationHighestFitness = []
        self.generationMedianFitness = []
        self.populationHighestFitness = []
        self.populationMedianFitness = []

        # a random object generator method
        # it is expected that each object will 
        # have the following properties and methods:
        # properties: fitness
        # methods: mutate, cross

        # a handle to a function that generates random object
        # it accepts current generation index (0-based) as input
        # also accepts optional additional arguments
        self.randomObject = opts['randomObjectGenerator']
        
        # this is a dictionary of keyword arguments
        self.treeOptions = opts['treeOptions']

        # verbose mode
        self.verbose = opts['verbose']
        
    #------------------------------------------------------------------#
    def run_first_generation(self):
        ts = time.time()

        generationObjects = [self.randomObject(**self.treeOptions) \
                             for index in range(self.generationSize)]
        generationFitness = [x.fitness for x in generationObjects]
                
        # add current generation index to objects from this generation
        dummy = [obj.set_generation_index(self.currentGeneration) \
                 for obj in generationObjects]

        # add to and sort object by most fit
        self.objects.extend(generationObjects)
        self.objects = sorted(self.objects, key = lambda x: -x.fitness)

        #update fitness metrics
        generationFitness = sorted(generationFitness, key = lambda x: -x)
        populationFitness = generationFitness

        self.generationHighestFitness.append(generationFitness[0])
        self.generationMedianFitness.append(np.median(generationFitness))
        self.populationHighestFitness.append(populationFitness[0])
        self.populationMedianFitness.append(np.median(populationFitness))
        
        duration = time.time() - ts
        
        if self.verbose:
            print >>sys.stderr, "Running GA in verbose mode:"
            print >>sys.stderr, "Time to run generation %d: %0.4f s"%(\
                                     self.currentGeneration+1, duration)
            print >>sys.stderr, "Maximum fitness sampled so far is %0.4f"%(\
                                self.populationHighestFitness[-1])
            topTrees = filter(lambda x: x.fitness == self.populationHighestFitness[-1],\
                              self.objects)
            topTreesNewick = list(set([tree.get_newick()[1] for tree in topTrees]))
            print >>sys.stderr, 'topologies sharing the highest fitness: %s'%(\
                                    '/'.join(topTreesNewick))
            print >>sys.stderr, '--------------------------------------------------'
        self.populationSize += self.generationSize
        self.currentGeneration += 1    
    #------------------------------------------------------------------#
    def add_generation(self):
                    
        ts = time.time()

        # a fraction of objects in each generations of random fresh objects
        
        freshObjects = [self.randomObject(**self.treeOptions) \
                        for index in range(int(self.randomObjectFraction *\
                                               self.generationSize))]
        #------------------------------------#
        # and for a fraction of them, ancestors are selected from
        # the current population
        ancestors = self.select(int((1-self.randomObjectFraction) * \
                                    self.generationSize))
        #------------------------------------#
        # for those with ancestors from previous generations, one needs to 
        # perform crossover and mutation with some prescribed frequencies (probs)
        
        # pair up ancestors
        ancestorPairs = pairwise(ancestors)
        pushIndices, crossIndices = get_cross_indices(ancestorPairs, self.Pc)
    
        intermediates_crossed = melt([self.objects[pair[0]].cross(\
                                      self.objects[pair[1]]) \
                                     for pair in crossIndices])
        
        intermediates_copied = [self.objects[index].copy() \
                                for index in pushIndices]
        parents = intermediates_crossed + intermediates_copied
        #------------------------------------#
        # perform mutation with prescribed probability
        pushIndices, mutIndices = get_mut_indices(parents, self.Pm)
        
        progeny = [parents[index].mutate() for index in mutIndices] + \
                      [parents[index].copy() for index in pushIndices]
        #------------------------------------#
        # generation objects ready!
        generationObjects = progeny + freshObjects
        
        # generation Index assignment
        dummy = [obj.set_generation_index(self.currentGeneration) \
                     for obj in generationObjects]
        # add this generation to the population
        self.objects.extend(generationObjects)

        # sort objects from most to least fit
        self.objects = sorted(self.objects, key = lambda x: -x.fitness)

        #------------------------------------#
        # fitness metric updates
        generationFitness = [x.fitness for x in generationObjects]
        populationFitness = [x.fitness for x in self.objects]
        generationFitness = sorted(generationFitness, key = lambda x: -x)
        populationFitness = sorted(populationFitness, key = lambda x: -x)
        
        self.generationHighestFitness.append(generationFitness[0])
        self.generationMedianFitness.append(np.median(generationFitness))
        self.populationHighestFitness.append(populationFitness[0])
        self.populationMedianFitness.append(np.median(populationFitness))
        #------------------------------------#
        
        duration = time.time() - ts

        if self.verbose:
            print >>sys.stdout, "Time to run generation %d: %0.4f s"%(\
                                self.currentGeneration+1,duration)
            print >>sys.stderr, "Maximum fitness sampled so far is %0.4f"%(\
                                self.populationHighestFitness[-1])
            topTrees = filter(lambda x: x.fitness == self.populationHighestFitness[-1],\
                              self.objects)
            topTreesNewick = list(set([tree.get_newick()[1] for tree in topTrees]))
            print >>sys.stderr, 'topologies sharing the highest fitness: %s'%(\
                                    '/'.join(topTreesNewick))
            print >>sys.stderr, '--------------------------------------------------'
                
        self.populationSize += self.generationSize
        self.currentGeneration += 1

    #------------------------------------------------------------------#
    def select(self,nObjects):
        # an implementation of fitness proportioa
        # assumes that the entities in self.object are sorted in 
        # descending order of fitness (most to least fit)
        currentFitness = [x.fitness for x in self.objects]
        rawSlots = list(running_sum(currentFitness))
        slots = [x/float(rawSlots[-1]) for x in rawSlots]
        picks = np.random.uniform(size = nObjects)
        indices = [map(lambda x: x > p, slots).index(True) for p in picks] 
        return indices
    #------------------------------------------------------------------#
    def store_metrics(self, path):
        f_w = file(path, 'w')
        f_w.write('# generation count = %d \n#\n'%self.currentGeneration )
        
        # store fitness metrics 
        metrics = map(lambda x: map(str,x), \
                      [range(1, self.currentGeneration+1), \
                       self.generationHighestFitness,\
                       self.generationMedianFitness,\
                       self.populationHighestFitness, \
                       self.populationMedianFitness])

        print >>f_w, '\t'.join(['generation.index',\
                             'generationHighestFitness',\
                             'generationMedianFitness',\
                             'populationHighestFitness',\
                             'populationMedianFitness'])

        print >>f_w, '\n'.join(map('\t'.join, zip(*metrics)))
        
        # store objects
        print >>f_w, '############################################################'
        print >>f_w, 'Topology\tMassCost\tTopologyCost\tCost\tFitness\tappearances(generationID:count)'
        summary = self.object_summary()
        print >>f_w, summary

        f_w.close()

    #------------------------------------------------------------------#
    def object_summary(self):
        '''
        return a list of all unique objects (as defined by object.string_identifier)
        , the count string representing how many times it has appeared in each
        generation, its cost and fitness parameters
        '''
        data = zip(self.objects,\
                   map(lambda x:x.string_identifier(), self.objects),\
                   map(lambda x:x.generationIndex, self.objects))
        historyDict = {}
        infoDict = {}
        
        for obj,string,index in data:
            if string not in historyDict:
                historyDict[string] = []
                infoDict[string] = [obj.massCost, obj.topologyCost, \
                                    obj.cost, obj.fitness]
            historyDict[string].append(index)
        
        for string in historyDict.keys():
            # make generation IDs 1-indexed
            generations = [x+1 for x in historyDict[string]]
            countString = ','.join(map(lambda x: ':'.join(map(str, x)),\
                                       sorted(dict(Counter(generations)).items(),\
                                              key = lambda x: x[0])))
            infoDict[string] = [string] + infoDict[string] + [countString]
        
        summary = infoDict.values()
        summary = sorted(summary, key = lambda x: -x[4])
        return '\n'.join(map(lambda x: '\t'.join(map(str,x)), summary))

######################################################################
class GAOptions(object):
    def __init__(self,generationSize, generationCount, randomObjectFraction,\
                 Pm, Pc, randomObjectGenerator, randomObjectGeneratorArgs,\
                 verbose):                 

        self.generationSize = generationSize
        self.generationCount = generationCount
        self.randomObjectFraction =  randomObjectFraction
        self.Pm = Pm
        self.Pc = Pc
        # it is expected that the randomObject method
        # can be called with a single input corresponding to current generation
        # index, and returns an object which has a fitness field, and a genration 
        # numeric field
        self.randomObjectGenerator = randomObjectGenerator
        self.randomObjectGeneratorArgs = randomObjectGeneratorArgs
        self.verbose = verbose
        return

######################################################################
class MassRules(object):
    def __init__(self, path):
        
        clusterIDs, sampleIDs, cellularity = read_cellularity_table(path)
        self.clusterIDs = clusterIDs
        self.sampleIDs = sampleIDs
        self.sampleCount = len(sampleIDs)
        self.cellularity = {}
        
        for clusterID in clusterIDs:
            index = clusterIDs.index(clusterID)
            self.cellularity[clusterID] = cellularity[index,:]

        return
    #----------------------------------------------------------#
    def __repr__(self):
        return 'object capturing cellularity of %d mutation clusters in %d samples'%\
               (len(self.clusterIDs), self.sampleCount)
    #----------------------------------------------------------#
    def residual_mass(self, parentID, childIDs):
        return self.cellularity[parentID] - \
               sum([self.cellularity[childID] for childID in childIDs])
    #----------------------------------------------------------#
    def mass_cost(self, parentID, childIDs):
        # l2 norm cost formulation
        return np.linalg.norm(np.minimum(0, \
                              self.residual_mass(parentID, childIDs)))
#============================================================================#
class TopologyRules(object):
    usage = '''object capturing the costs associated with topological '''
    usage = '''\n configurations resulting from hypothesis test results '''
    __slots__ = ['costDict', 'clusterIDs']

    def __init__(self, path = None):
        self.costDict = {}
        if not path: return

        # filter out file header
        f_h = file(path, 'r')        
        line = f_h.readline()
        while line.startswith('#') or line.startswith('parent'):
            line = f_h.readline()
        
        # first line here
        toks = line.strip().split('\t')
        self.costDict[(toks[0],toks[1])] = float(toks[2])

        # loop thru the remaining lines
        for line in f_h:
            if line.strip() == '': break
            if line.startswith('#'): continue
            toks = line.strip().split('\t')
            self.costDict[(toks[0],toks[1])] = float(toks[2])
        f_h.close()

        self.clusterIDs = list(set(zip(*self.costDict.keys())[0] + \
                          zip(*self.costDict.keys())[1]))

        return
    #----------------------------------------------------------#
    def topology_cost(self, pairs):
        return sum([self.costDict[pair] if pair in self.costDict else 0 \
                    for pair in pairs])
#============================================================================#

#-------------------------helper methods------------------------------#
#---------------------------------------------------------------------#
def running_sum(myList):
    total = 0
    for item in myList:
        total += item
        yield total
#----------------------------------------------------------------------#
def pairwise(iterable):
    if len(iterable) % 2 != 0:
        # write an error for odd sized list
        # drop the last element, and continue
        print >> sys.stderr, "cannot create pairs from an odd-sized list"
        print >> sys.stderr, "dropping the last element: %s"%str(iterable[-1])
        myList = iterable[:-1]
    else:
        myList = iterable
    return zip(myList[0::2], myList[1::2])
#----------------------------------------------------------------------#
def melt(myList):
    return [subitem for item in myList for subitem in item]
#----------------------------------------------------------------------#
def get_cross_indices(ancestorPairs, Pc):
    '''
    return cross pair and copy indices for random crossing operation with prescribed
    probability
    '''
    n = len(ancestorPairs)
    crossoverDraw = np.random.uniform(size = n)
    
    # for pairs to be crossed, return pairs of object indices w.r.t. GA.objects
    crossIndices = [(ancestorPairs[ind][0],ancestorPairs[ind][1]) \
                    for ind in range(n) if crossoverDraw[ind] <= Pc]
    # for pairs to be copied over, return list of singular object indices w.r.t. ... 
    pushIndices = melt([(ancestorPairs[ind][0], ancestorPairs[ind][1]) \
                        for ind in range(n) if crossoverDraw[ind] > Pc])
    return pushIndices, crossIndices
#----------------------------------------------------------------------#
def get_mut_indices(parents, Pm):
    '''
    return mut and copy indices for random mutation operation with prescribed 
    probability
    '''
    n = len(parents)
    mutDraw = np.random.uniform(size = n)
    
    # index list w.r.t parents of elements to be mutated
    mutIndices = [ind for ind in range(n) if mutDraw[ind] <= Pm]
    # index list w.r.t parents of elements to be copied over
    pushIndices = [ind for ind in range(n) if mutDraw[ind] > Pm]
    return pushIndices, mutIndices
#----------------------------------------------------------------------#
def read_cellularity_table(path):
    # read in mutation cluster cellularity values 
    # across samples
    # this module typically reads in the output of 
    # HT.average_cellularity
    f_h = file(path, 'r')
    content = f_h.read().split('\n')
    f_h.close()
    content = map(lambda x: x.split('\t'),\
                  filter(lambda y: (y != '') and\
                         (not y.startswith('#')) and\
                           (not y.startswith('sample')),\
                         content))
    sampleIDs = sorted(list(set(zip(*content)[0])))
    clusterIDs = sorted(list(set(zip(*content)[1])))
    cellularity = np.zeros((len(clusterIDs), len(sampleIDs)))
    
    for element in content:
        cellularity[clusterIDs.index(element[1]),\
                    sampleIDs.index(element[0])] = float(element[2])
    return clusterIDs, sampleIDs, cellularity
#----------------------------------------------------------------------#
def read_instance(path):
    trees = []
    f_h = file(path,'r')
    
    line = f_h.readline()
    while not line.startswith('#####'):
        line = f_h.readline()
        
    header = f_h.readline()
    for line in f_h:
        toks = line.strip().split('\t')
        trees += [tuple([toks[0]] + [map(float, toks[1:5])])]
    f_h.close()
    return trees
#----------------------------------------------------------------------#
def get_consensus_edges(nwkFittestTrees):
    N = len(nwkFittestTrees)
    Edges = {} # track edge appearance across fittest trees
    cEdges = [] # edges labeled with fraction of times they appear
    for nwkTree in nwkFittestTrees:
        tree = Node.from_newick(nwkTree)
        tree.update_descendants()
        edges = tree.get_pairs()
        for edge in edges:
            if edge not in Edges.keys():
                Edges[edge] = 0
            Edges[edge]+=1

    # sort edges by value if they are numeric
    clusterLabels = list(set(zip(*Edges.keys())[0] + \
                             zip(*Edges.keys())[1]))
    # if cluster IDs are numeric, sort
    if sum([x.isdigit() for x in clusterLabels]) == len(clusterLabels):
        edgeKeys = sorted(Edges.keys(), \
                          key = lambda x: (int(x[0]),int(x[1])))
    else:
        edgeKeys = sorted(Edges.keys())

    # prepare to return edges labeled with frequency information
    for edge in edgeKeys:
         cEdges.append('\t'.join(map(str, edge) +\
                                 [str(float(Edges[edge])/N),\
                                 str(Edges[edge]) + '/' + str(N)]))

    return cEdges
#----------------------------------------------------------------------#


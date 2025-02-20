#! /usr/bin/env python
'''
Prototype implementation of basic GA functionality
'''
import sys
import time
from typing import Any, List, Tuple, Dict, Iterator

import numpy as np
from collections import Counter
from itertools import combinations

from SCHISM.Tree import Node


class GA:
    #------------------------------------------------------------------#
    def __init__(self, opts: Dict[str, Any]) -> None:
        self.objects: List[Any] = []
        self.generationSize: int = opts['generationSize']
        self.generationCount: int = opts['generationCount']
   
        # the fraction of objects in each generation that are
        # randomly generated from scratch and are not descendants
        # of objects so far.
        self.randomObjectFraction: float = opts['randomObjectFraction']
        
        # mutation probability
        self.Pm: float = opts['Pm']
        
        # crossover probability
        self.Pc: float = opts['Pc']

        self.currentGeneration: int = 0
        self.populationSize: int = 0

        # running lists corresponding to generation max and median fitness
        # and population max and median fitness after each generation
        self.generationHighestFitness: List[float] = []
        self.generationMedianFitness: List[float] = []
        self.populationHighestFitness: List[float] = []
        self.populationMedianFitness: List[float] = []

        # a random object generator method
        # it is expected that each object will 
        # have the following properties and methods:
        # properties: fitness
        # methods: mutate, cross

        # a handle to a function that generates random object.
        # It accepts current generation index (0-based) as input
        # and accepts optional additional arguments.
        self.randomObject = opts['randomObjectGenerator']
        
        # this is a dictionary of keyword arguments
        self.treeOptions: Dict[str, Any] = opts['treeOptions']

        # verbose mode
        self.verbose: bool = opts['verbose']
        
    #------------------------------------------------------------------#
    def run_first_generation(self) -> None:
        ts = time.time()

        generationObjects = [self.randomObject(**self.treeOptions)
                             for _ in range(self.generationSize)]
        generationFitness = [x.fitness for x in generationObjects]
                
        # add current generation index to objects from this generation
        for obj in generationObjects:
            obj.set_generation_index(self.currentGeneration)

        # add to and sort objects by most fit
        self.objects.extend(generationObjects)
        self.objects.sort(key=lambda x: -x.fitness)

        # update fitness metrics
        generationFitness_sorted = sorted(generationFitness, reverse=True)
        populationFitness = generationFitness_sorted

        self.generationHighestFitness.append(generationFitness_sorted[0])
        self.generationMedianFitness.append(np.median(generationFitness_sorted))
        self.populationHighestFitness.append(populationFitness[0])
        self.populationMedianFitness.append(np.median(populationFitness))
        
        duration = time.time() - ts
        
        if self.verbose:
            print("Running GA in verbose mode:", file=sys.stderr)
            print(f"Time to run generation {self.currentGeneration+1}: {duration:0.4f} s", file=sys.stderr)
            print(f"Maximum fitness sampled so far is {self.populationHighestFitness[-1]:0.4f}", file=sys.stderr)
            topTrees = list(filter(lambda x: x.fitness == self.populationHighestFitness[-1],
                                   self.objects))
            topTreesNewick = list(set([tree.get_newick()[1] for tree in topTrees]))
            print(f"Topologies sharing the highest fitness: {'/'.join(topTreesNewick)}", file=sys.stderr)
            print('--------------------------------------------------', file=sys.stderr)
        self.populationSize += self.generationSize
        self.currentGeneration += 1    
    #------------------------------------------------------------------#
    def add_generation(self) -> None:
        ts = time.time()

        # a fraction of objects in each generation of random fresh objects
        freshObjects = [self.randomObject(**self.treeOptions)
                        for _ in range(int(self.randomObjectFraction * self.generationSize))]
        #------------------------------------#
        # for the remaining objects, select ancestors from the current population
        ancestors = self.select(int((1 - self.randomObjectFraction) * self.generationSize))
        #------------------------------------#
        # perform crossover and mutation with prescribed probabilities
        ancestorPairs = pairwise(ancestors)
        pushIndices, crossIndices = get_cross_indices(ancestorPairs, self.Pc)
    
        intermediates_crossed = melt([self.objects[pair[0]].cross(self.objects[pair[1]])
                                      for pair in crossIndices])
        intermediates_copied = [self.objects[index].copy() for index in pushIndices]
        parents = intermediates_crossed + intermediates_copied
        #------------------------------------#
        # perform mutation with prescribed probability
        pushIndices, mutIndices = get_mut_indices(parents, self.Pm)
        
        progeny = [parents[index].mutate() for index in mutIndices] + \
                  [parents[index].copy() for index in pushIndices]
        #------------------------------------#
        # generation objects ready!
        generationObjects = progeny + freshObjects
        
        # assign generation index
        for obj in generationObjects:
            obj.set_generation_index(self.currentGeneration)
        # add this generation to the population
        self.objects.extend(generationObjects)

        # sort objects from most to least fit
        self.objects.sort(key=lambda x: -x.fitness)

        #------------------------------------#
        # fitness metric updates
        generationFitness = [x.fitness for x in generationObjects]
        populationFitness = [x.fitness for x in self.objects]
        generationFitness_sorted = sorted(generationFitness, reverse=True)
        populationFitness_sorted = sorted(populationFitness, reverse=True)
        
        self.generationHighestFitness.append(generationFitness_sorted[0])
        self.generationMedianFitness.append(np.median(generationFitness_sorted))
        self.populationHighestFitness.append(populationFitness_sorted[0])
        self.populationMedianFitness.append(np.median(populationFitness_sorted))
        #------------------------------------#
        
        duration = time.time() - ts

        if self.verbose:
            print(f"Time to run generation {self.currentGeneration+1}: {duration:0.4f} s", file=sys.stdout)
            print(f"Maximum fitness sampled so far is {self.populationHighestFitness[-1]:0.4f}", file=sys.stderr)
            topTrees = list(filter(lambda x: x.fitness == self.populationHighestFitness[-1],
                                   self.objects))
            topTreesNewick = list(set([tree.get_newick()[1] for tree in topTrees]))
            print(f"Topologies sharing the highest fitness: {'/'.join(topTreesNewick)}", file=sys.stderr)
            print('--------------------------------------------------', file=sys.stderr)
                
        self.populationSize += self.generationSize
        self.currentGeneration += 1

    #------------------------------------------------------------------#
    def select(self, nObjects: int) -> List[int]:
        # Implementation of fitness proportional selection.
        # Assumes that self.objects is sorted in descending order of fitness.
        currentFitness = [x.fitness for x in self.objects]
        rawSlots = list(running_sum(currentFitness))
        slots = [x / float(rawSlots[-1]) for x in rawSlots]
        picks = np.random.uniform(size=nObjects)
        indices = [list(map(lambda x: x > p, slots)).index(True) for p in picks]
        return indices

    #------------------------------------------------------------------#
    def store_metrics(self, path: str) -> None:
        with open(path, 'w') as f_w:
            f_w.write(f"# generation count = {self.currentGeneration} \n#\n")
            
            # store fitness metrics 
            metrics_data = list(map(list, [
                list(range(1, self.currentGeneration + 1)),
                self.generationHighestFitness,
                self.generationMedianFitness,
                self.populationHighestFitness,
                self.populationMedianFitness
            ]))
            
            print('\t'.join(['generation.index',
                             'generationHighestFitness',
                             'generationMedianFitness',
                             'populationHighestFitness',
                             'populationMedianFitness']), file=f_w)
            
            #print('\n'.join(map('\t'.join, zip(*metrics_data))), file=f_w)
            print('\n'.join(map(lambda row: '\t'.join(map(str, row)), zip(*metrics_data))), file=f_w)

            # store objects
            print('############################################################', file=f_w)
            print('Topology\tMassCost\tTopologyCost\tCost\tFitness\tappearances(generationID:count)', file=f_w)
            summary = self.object_summary()
            print(summary, file=f_w)

    #------------------------------------------------------------------#
    def object_summary(self) -> str:
        '''
        Return a summary of all unique objects (as defined by object.string_identifier),
        the count string representing how many times it has appeared in each generation,
        and its cost and fitness parameters.
        '''
        data = list(zip(self.objects,
                        list(map(lambda x: x.string_identifier(), self.objects)),
                        list(map(lambda x: x.generationIndex, self.objects))))
        historyDict: Dict[str, List[int]] = {}
        infoDict: Dict[str, List[Any]] = {}
        
        for obj, string, index in data:
            if string not in historyDict:
                historyDict[string] = []
                infoDict[string] = [obj.massCost, obj.topologyCost, obj.cost, obj.fitness]
            historyDict[string].append(index)
        
        for string in historyDict.keys():
            # make generation IDs 1-indexed
            generations = [x + 1 for x in historyDict[string]]
            countString = ','.join(map(lambda x: ':'.join(map(str, x)),
                                       sorted(dict(Counter(generations)).items(), key=lambda x: x[0])))
            infoDict[string] = [string] + infoDict[string] + [countString]
        
        summary = list(infoDict.values())
        summary.sort(key=lambda x: -x[4])
        return '\n'.join(map(lambda x: '\t'.join(map(str, x)), summary))


######################################################################
class GAOptions:
    def __init__(self, generationSize: int, generationCount: int, randomObjectFraction: float,
                 Pm: float, Pc: float, randomObjectGenerator: Any, randomObjectGeneratorArgs: Any,
                 verbose: bool) -> None:
        self.generationSize = generationSize
        self.generationCount = generationCount
        self.randomObjectFraction = randomObjectFraction
        self.Pm = Pm
        self.Pc = Pc
        # It is expected that the randomObjectGenerator method
        # can be called with a single input corresponding to the current generation index
        # and returns an object which has a fitness field and a generation numeric field.
        self.randomObjectGenerator = randomObjectGenerator
        self.randomObjectGeneratorArgs = randomObjectGeneratorArgs
        self.verbose = verbose


######################################################################
class MassRules:
    def __init__(self, path: str) -> None:
        clusterIDs, sampleIDs, cellularity = read_cellularity_table(path)
        self.clusterIDs = clusterIDs
        self.sampleIDs = sampleIDs
        self.sampleCount = len(sampleIDs)
        self.cellularity: Dict[str, np.ndarray] = {}
        
        for clusterID in clusterIDs:
            index = clusterIDs.index(clusterID)
            self.cellularity[clusterID] = cellularity[index, :]

    #----------------------------------------------------------#
    def __repr__(self) -> str:
        return ('object capturing cellularity of %d mutation clusters in %d samples' %
                (len(self.clusterIDs), self.sampleCount))
    #----------------------------------------------------------#
    def residual_mass(self, parentID: str, childIDs: List[str]) -> np.ndarray:
        return self.cellularity[parentID] - sum([self.cellularity[childID] for childID in childIDs])
    #----------------------------------------------------------#
    def mass_cost(self, parentID: str, childIDs: List[str]) -> float:
        # L2 norm cost formulation.
        return np.linalg.norm(np.minimum(0, self.residual_mass(parentID, childIDs)))


#============================================================================#
class TopologyRules:
    usage = '''Object capturing the costs associated with topological configurations
    resulting from hypothesis test results.'''
    __slots__ = ['costDict', 'clusterIDs']

    def __init__(self, path: str = None) -> None:
        self.costDict: Dict[Tuple[str, str], float] = {}
        if not path:
            return

        # filter out file header
        with open(path, 'r') as f_h:
            line = f_h.readline()
            while line.startswith('#') or line.startswith('parent'):
                line = f_h.readline()
            
            # first line here
            toks = line.strip().split('\t')
            self.costDict[(toks[0], toks[1])] = float(toks[2])

            # loop through the remaining lines
            for line in f_h:
                if line.strip() == '':
                    break
                if line.startswith('#'):
                    continue
                toks = line.strip().split('\t')
                self.costDict[(toks[0], toks[1])] = float(toks[2])

        keys = list(self.costDict.keys())
        zipped_keys = list(zip(*keys))
        self.clusterIDs = list(set(list(zipped_keys[0]) + list(zipped_keys[1])))

        #self.clusterIDs = list(set(list(zip(*list(self.costDict.keys()))[0]) +
        #                            list(zip(*list(self.costDict.keys()))[1])))

    #----------------------------------------------------------#
    def topology_cost(self, pairs: List[Tuple[str, str]]) -> float:
        return sum([self.costDict[pair] if pair in self.costDict else 0 for pair in pairs])


#============================================================================#
#-------------------------Helper Methods------------------------------#
#---------------------------------------------------------------------#
def running_sum(myList: List[float]) -> Iterator[float]:
    total = 0.0
    for item in myList:
        total += item
        yield total

#----------------------------------------------------------------------#
def pairwise(iterable: List[int]) -> List[Tuple[int, int]]:
    if len(iterable) % 2 != 0:
        # Write an error for odd-sized list: drop the last element and continue.
        print(f"cannot create pairs from an odd-sized list", file=sys.stderr)
        print(f"dropping the last element: {iterable[-1]}", file=sys.stderr)
        myList = iterable[:-1]
    else:
        myList = iterable
    return list(zip(myList[0::2], myList[1::2]))

#----------------------------------------------------------------------#
def melt(myList: List[List[Any]]) -> List[Any]:
    return [subitem for item in myList for subitem in item]

#----------------------------------------------------------------------#
def get_cross_indices(ancestorPairs: List[Tuple[int, int]], Pc: float) -> Tuple[List[int], List[Tuple[int, int]]]:
    '''
    Return copy indices and cross indices for a random crossing operation with prescribed probability.
    '''
    n = len(ancestorPairs)
    crossoverDraw = np.random.uniform(size=n)
    
    # For pairs to be crossed, return pairs of object indices (w.r.t. GA.objects).
    crossIndices = [ancestorPairs[ind] for ind in range(n) if crossoverDraw[ind] <= Pc]
    # For pairs to be copied over, return a list of singular object indices.
    pushIndices = melt([list(ancestorPairs[ind]) for ind in range(n) if crossoverDraw[ind] > Pc])
    return pushIndices, crossIndices

#----------------------------------------------------------------------#
def get_mut_indices(parents: List[Any], Pm: float) -> Tuple[List[int], List[int]]:
    '''
    Return indices for copy and mutation operations for a random mutation operation with prescribed probability.
    '''
    n = len(parents)
    mutDraw = np.random.uniform(size=n)
    
    mutIndices = [ind for ind in range(n) if mutDraw[ind] <= Pm]
    pushIndices = [ind for ind in range(n) if mutDraw[ind] > Pm]
    return pushIndices, mutIndices

#----------------------------------------------------------------------#
def read_cellularity_table(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    '''
    Read in mutation cluster cellularity values across samples.
    This module typically reads the output of HT.average_cellularity.
    '''
    with open(path, 'r') as f_h:
        content = f_h.read().split('\n')
    content = list(filter(lambda y: y != '' and not y.startswith('#') and not y.startswith('sample'), content))
    content = list(map(lambda x: x.split('\t'), content))
    sampleIDs = sorted(list(set(list(zip(*content))[0])))
    clusterIDs = sorted(list(set(list(zip(*content))[1])))
    cellularity = np.zeros((len(clusterIDs), len(sampleIDs)))
    
    for element in content:
        cellularity[clusterIDs.index(element[1]), sampleIDs.index(element[0])] = float(element[2])
    return clusterIDs, sampleIDs, cellularity

#----------------------------------------------------------------------#
def read_instance(path: str) -> List[Tuple[str, List[float]]]:
    trees = []
    with open(path, 'r') as f_h:
        line = f_h.readline()
        while not line.startswith('#####'):
            line = f_h.readline()
        
        header = f_h.readline()
        for line in f_h:
            toks = line.strip().split('\t')
            # Wrap map with list so that it produces a list of floats.
            trees.append((toks[0], list(map(float, toks[1:5]))))
    return trees

#----------------------------------------------------------------------#
def get_consensus_edges(nwkFittestTrees: List[str]) -> List[str]:
    N = len(nwkFittestTrees)
    Edges: Dict[Tuple[str, str], int] = {}  # track edge appearance across fittest trees
    cEdges: List[str] = []  # edges labeled with fraction of times they appear
    for nwkTree in nwkFittestTrees:
        tree = Node.from_newick(nwkTree)
        tree.update_descendants()
        edges = tree.get_pairs()
        for edge in edges:
            if edge not in Edges:
                Edges[edge] = 0
            Edges[edge] += 1

    # sort edges by value if they are numeric
    clusterLabels = list(set(list(zip(*list(Edges.keys()))[0]) + list(zip(*list(Edges.keys()))[1])))
    # if cluster IDs are numeric, sort accordingly
    if sum([x.isdigit() for x in clusterLabels]) == len(clusterLabels):
        edgeKeys = sorted(Edges.keys(), key=lambda x: (int(x[0]), int(x[1])))
    else:
        edgeKeys = sorted(Edges.keys())

    # prepare to return edges labeled with frequency information
    for edge in edgeKeys:
        cEdges.append('\t'.join(map(str, edge) +
                                  [str(float(Edges[edge]) / N),
                                   f"{Edges[edge]}/{N}"]))
    return cEdges

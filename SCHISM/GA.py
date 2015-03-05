import os 
from collections import Counter

from Tree import Node
from utils import Config

from visualize import plot_ga_fitness_trace
from visualize import plot_ga_top_tree_count_trace
from visualize import plot_consensus_tree

from GATools import *

#----------------------------------------------------------------------#
def run_ga(args):
    config = Config(args.config_file)
    
    CPOVPath = os.path.join(config.working_dir,\
                            config.output_prefix + '.HT.cpov')
    
    cellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')

    topologyRules = TopologyRules(CPOVPath)
    massRules = MassRules(cellularityPath)

    clusterIDs = topologyRules.clusterIDs
    
    treeOptions = {'fitnessCoefficient': config.genetic_algorithm['fitness_coefficient'],\
                 'clusterIDs': clusterIDs,\
                 'topologyRules': topologyRules,\
                 'massRules': massRules}

    # gather other parameters for running ga
    # setup GA workflow
    gaOptions = {'generationCount': config.genetic_algorithm['generation_count'],\
                 'generationSize': config.genetic_algorithm['generation_size'],\
                 'randomObjectFraction': config.genetic_algorithm['random_object_fraction'],\
                 'Pm': config.genetic_algorithm['mutation_probability'],\
                 'Pc': config.genetic_algorithm['crossover_probability'],\
                 'randomObjectGenerator': Node.random_topology,\
                 'treeOptions': treeOptions,\
                 'verbose': config.genetic_algorithm['verbose']}
    
    if args.mode == 'serial':
        # if run in serial mode, perform series of independent 
        # ga runs, and index them by 1 to instanceCount
        instanceCount = config.genetic_algorithm['instance_count']
    elif args.mode == 'parallel':
        # if run in parallel mode, perform a single GA run
        # and name it using args.runID
        instanceCount = 1
    else:
        print >>sys.stderr, 'Unrecognized mode for GA run. please select '+\
            ' serial or parallel as mode.'
        sys.exit()

    for instance in range(instanceCount):
        gaRun = GA(gaOptions)
        gaRun.run_first_generation()
        for index in range(config.genetic_algorithm['generation_count'] - 1):
            gaRun.add_generation()
        if args.mode == 'serial':
            path = os.path.join(config.working_dir,\
                         config.output_prefix + '.GA.r' + str(instance+1)+ '.trace')
        else:
            path = os.path.join(config.working_dir,\
                                config.output_prefix + '.GA.r' + \
                                str(args.runID)+ '.trace')
        gaRun.store_metrics(path)
#----------------------------------------------------------------------#
def ga_summary(args):
    merge_ga_traces(args)
    plot_ga_fitness_trace(args)
    plot_ga_top_tree_count_trace(args)
#----------------------------------------------------------------------#
def merge_ga_traces(args):
    config = Config(args.config_file)
    Range = range(1, 1 + config.genetic_algorithm['instance_count'])
    allTrees = []

    # gather the set of trees explored by the ensemble of
    # independent GA runs
    for index in Range:
        path = os.path.join(config.working_dir,\
                            config.output_prefix + '.GA.r%d.trace'%index)
        runTrees = read_instance(path)
        allTrees += runTrees
    
    counts = dict(Counter(zip(*allTrees)[0]))
    # sort trees by fitness value
    allTrees = sorted(dict(allTrees).items(),\
                      key = lambda x: -x[1][-1])
    
    outputPath = os.path.join(config.working_dir,\
                            config.output_prefix + '.GA.trace')
    
    # print the sorted (by fitness) list of trees to GA.trace file 
    f_w = file(outputPath, 'w')
    print >>f_w, 'Topology\tMassCost\tTopologyCost\tCost\tFitness\t#(Runs)'

    for tree, metrics in allTrees:
        print >>f_w, '\t'.join([tree] + \
                               map(lambda x:'%0.45'%x,metrics) +\
                               [str(counts[tree])])
    f_w.close()
#----------------------------------------------------------------------#
def generate_consensus_tree(args):
    # construct consensus tree
    config = Config(args.config_file)
    inputPath = os.path.join(config.working_dir,\
                             config.output_prefix + '.GA.trace')
    outputPath = os.path.join(config.working_dir,\
                             config.output_prefix + '.GA.consensusTree')
    f_h = file(inputPath)
    line = f_h.readline()
    while not line.startswith('Topology'):
        line = f_h.readline()
    content = map(lambda x: x.split('\t'), \
                  filter(lambda y: y!= '', f_h.read().split('\n')))
    f_h.close()

    # narrow down the set of all topologies having maximum fitness sampled
    highestFitness = max(map(float,zip(*content)[4]))
    fittestTrees = filter(lambda x: float(x[4]) == highestFitness,\
                          content)
    
    nwkFittestTrees = zip(*fittestTrees)[0]
    consensusEdges = get_consensus_edges(nwkFittestTrees)
    
    f_w = file(outputPath, 'w')
    print >>f_w, 'parent\tchild\tfrequency\tlabel'
    print >>f_w, '\n'.join(consensusEdges)
    f_w.close()

    # plot consensus tree
    plot_consensus_tree(args)
#----------------------------------------------------------------------#

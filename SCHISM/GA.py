import os
import sys
from collections import Counter
from typing import Any, List

from SCHISM.Tree import Node
from SCHISM.utils import Config
from SCHISM.visualize import plot_ga_fitness_trace, plot_ga_top_tree_count_trace, plot_consensus_tree
from SCHISM.GATools import *  # assuming this imports GA, TopologyRules, MassRules, read_instance, get_consensus_edges

#----------------------------------------------------------------------#
def run_ga(args: Any) -> None:
    config = Config(args.config_file)

    CPOVPath = os.path.join(config.working_dir,
                            config.output_prefix + '.HT.cpov')

    cellularityPath = os.path.join(config.working_dir,
                                   config.output_prefix + '.cluster.cellularity')

    topologyRules = TopologyRules(CPOVPath)
    massRules = MassRules(cellularityPath)

    clusterIDs = topologyRules.clusterIDs

    treeOptions = {
        'fitnessCoefficient': config.genetic_algorithm['fitness_coefficient'],
        'clusterIDs': clusterIDs,
        'topologyRules': topologyRules,
        'massRules': massRules
    }

    # gather other parameters for running GA
    # setup GA workflow
    gaOptions = {
        'generationCount': config.genetic_algorithm['generation_count'],
        'generationSize': config.genetic_algorithm['generation_size'],
        'randomObjectFraction': config.genetic_algorithm['random_object_fraction'],
        'Pm': config.genetic_algorithm['mutation_probability'],
        'Pc': config.genetic_algorithm['crossover_probability'],
        'randomObjectGenerator': Node.random_topology,
        'treeOptions': treeOptions,
        'verbose': config.genetic_algorithm['verbose']
    }

    if args.mode == 'serial':
        # if run in serial mode, perform series of independent 
        # GA runs, and index them by 1 to instanceCount
        instanceCount = config.genetic_algorithm['instance_count']
    elif args.mode == 'parallel':
        # if run in parallel mode, perform a single GA run
        # and name it using args.runID
        instanceCount = 1
    else:
        print('Unrecognized mode for GA run. Please select serial or parallel as mode.', file=sys.stderr)
        sys.exit(1)

    for instance in range(instanceCount):
        gaRun = GA(gaOptions)
        gaRun.run_first_generation()
        for _ in range(config.genetic_algorithm['generation_count'] - 1):
            gaRun.add_generation()
        if args.mode == 'serial':
            path = os.path.join(config.working_dir,
                                f"{config.output_prefix}.GA.r{instance+1}.trace")
        else:
            path = os.path.join(config.working_dir,
                                f"{config.output_prefix}.GA.r{args.runID}.trace")
        gaRun.store_metrics(path)


#----------------------------------------------------------------------#
def ga_summary(args: Any) -> None:
    merge_ga_traces(args)
    plot_ga_fitness_trace(args)
    plot_ga_top_tree_count_trace(args)


#----------------------------------------------------------------------#
def merge_ga_traces(args: Any) -> None:
    config = Config(args.config_file)
    instance_range = range(1, 1 + config.genetic_algorithm['instance_count'])
    allTrees: List[Any] = []

    # gather the set of trees explored by the ensemble of independent GA runs
    for index in instance_range:
        path = os.path.join(config.working_dir,
                            f"{config.output_prefix}.GA.r{index}.trace")
        runTrees = read_instance(path)
        allTrees += runTrees

    # Assuming that each element of allTrees is a tuple (tree, metrics)
    counts = dict(Counter(list(zip(*allTrees))[0]))
    # sort trees by fitness value (assuming last metric is fitness)
    allTrees_sorted = sorted(dict(allTrees).items(), key=lambda x: -x[1][-1])

    outputPath = os.path.join(config.working_dir,
                              f"{config.output_prefix}.GA.trace")

    with open(outputPath, 'w') as f_w:
        print('Topology\tMassCost\tTopologyCost\tCost\tFitness\t#(Runs)', file=f_w)
        for tree, metrics in allTrees_sorted:
            # Convert each metric to string with 4 decimal places
            metric_strs = list(map(lambda x: f'{x:0.4f}', metrics))
            print('\t'.join([tree] + metric_strs + [str(counts[tree])]), file=f_w)


#----------------------------------------------------------------------#
def generate_consensus_tree(args: Any) -> None:
    # construct consensus tree
    config = Config(args.config_file)
    inputPath = os.path.join(config.working_dir,
                             f"{config.output_prefix}.GA.trace")
    outputPath = os.path.join(config.working_dir,
                              f"{config.output_prefix}.GA.consensusTree")

    with open(inputPath, 'r') as f_h:
        line = f_h.readline()
        while not line.startswith('Topology'):
            line = f_h.readline()
        # read the rest of the file, splitting on newlines and filtering out empty lines
        content = list(map(lambda x: x.split('\t'),
                           filter(lambda y: y != '', f_h.read().split('\n'))))

    # narrow down the set of all topologies having maximum fitness sampled
    fitness_values = list(map(float, list(zip(*content))[4]))
    highestFitness = max(fitness_values)
    fittestTrees = list(filter(lambda x: float(x[4]) == highestFitness, content))

    nwkFittestTrees = list(zip(*fittestTrees))[0]
    consensusEdges = get_consensus_edges(nwkFittestTrees)

    with open(outputPath, 'w') as f_w:
        print('parent\tchild\tfrequency\tlabel', file=f_w)
        print('\n'.join(consensusEdges), file=f_w)

    # plot consensus tree
    plot_consensus_tree(args)

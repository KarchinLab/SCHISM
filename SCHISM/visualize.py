import os
import sys
import numpy as np

from utils import Config

from HT import Sample
from HT import read_input_samples
from HT import read_cluster_assignments

#----------------------------------------------------------------------#
def plot_cpov(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print('Plotting cpov matrix requires matplotlib \n'
              'module. Failed to import matplotlib. Please install \n'
              'matplotlib package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    cpovPath = os.path.join(config.working_dir,
                            config.output_prefix + '.HT.cpov')
    cpov, clusterIDs = read_cpov_matrix(cpovPath)
    
    N = cpov.shape[1]

    # Block out diagonal elements as they are meaningless
    for index in range(N):
        cpov[index, index] = 100

    width = min(7, 1.5 * N)
    fig, ax = plt.subplots()

    # Generate colormap resembling ggplot2 mutedBlue
    cdict1 = {'red': ((0.0, 0.226, 0.226),
                      (1.0, 1.0, 1.0)),
              'blue': ((0.0, 0.593, 0.593),
                       (1.0, 1.0, 1.0)),
              'green': ((0.0, 0.226, 0.226),
                        (1.0, 1.0, 1.0))}
    mutedBlue = LinearSegmentedColormap('mutedBlue', cdict1)
    plt.register_cmap(cmap=mutedBlue)
    cmap = mutedBlue
    cmap.set_over('0.72')

    # Generate heatmap
    heatmap = ax.pcolor(cpov, cmap=mutedBlue,
                        vmax=1.0, vmin=-0.05,
                        edgecolors=[0.4, 0.4, 0.4])
    
    ax.set_xticks(np.arange(N) + 0.5, minor=False)
    ax.set_yticks(np.arange(N) + 0.5, minor=False)
    ax.set_xticklabels(clusterIDs)
    ax.set_yticklabels(clusterIDs)

    # Add labels to heatmap
    for y in range(N):
        for x in range(N):
            if (config.hypothesis_test['test_level'] == 'mutations') and (x != y):
                plt.text(x + 0.5, y + 0.5, '%.2f' % cpov[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')
            else:
                continue

    ax.set_xlim([0, N])
    ax.set_ylim([0, N])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    fig = plt.gcf()
    fig.set_size_inches(width, width)
    
    cpovFigure = os.path.join(config.working_dir,
                              config.output_prefix + '.HT.cpov.pdf')
    fig.savefig(cpovFigure, dpi=300)

#----------------------------------------------------------------------#
def plot_pov(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print('Plotting pov matrix requires matplotlib \n'
              'module. Failed to import matplotlib. Please install \n'
              'matplotlib package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    povPath = os.path.join(config.working_dir,
                           config.output_prefix + '.HT.pov')
    pov, mutIDs = read_cpov_matrix(povPath)
    
    pov, mutIDs = reorder_pov_by_cluster(pov, mutIDs, config)
    N = pov.shape[1]

    # Block out diagonal elements as they are meaningless
    for index in range(N):
        pov[index, index] = 100
    
    width = min(7, 1.5 * N)
    fig, ax = plt.subplots()

    cdict1 = {'red': ((0.0, 0.226, 0.226),
                      (1.0, 1.0, 1.0)),
              'blue': ((0.0, 0.593, 0.593),
                       (1.0, 1.0, 1.0)),
              'green': ((0.0, 0.226, 0.226),
                        (1.0, 1.0, 1.0))}
    mutedBlue = LinearSegmentedColormap('mutedBlue', cdict1)
    plt.register_cmap(cmap=mutedBlue)
    cmap = mutedBlue
    cmap.set_over('0.72')
    
    heatmap = ax.pcolor(pov, cmap=mutedBlue,
                        vmax=1.0, vmin=-0.05,
                        edgecolors=[0.4, 0.4, 0.4])
    
    ax.set_xticks(np.arange(N) + 0.5, minor=False)
    ax.set_yticks(np.arange(N) + 0.5, minor=False)
    ax.set_xticklabels(mutIDs)
    ax.set_yticklabels(mutIDs)

    ax.set_xlim([0, N])
    ax.set_ylim([0, N])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    fig = plt.gcf()
    fig.set_size_inches(width, width)
    
    povFigure = os.path.join(config.working_dir,
                             config.output_prefix + '.HT.pov.pdf')
    fig.savefig(povFigure, dpi=300)

#----------------------------------------------------------------------#
def plot_ga_fitness_trace(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plotting GA fitness trace requires matplotlib \n'
              'module. Failed to import matplotlib. Please install \n'
              'matplotlib package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    run_range = range(1, 1 + config.genetic_algorithm['instance_count'])
    
    nrow = int(np.ceil(len(run_range) / 4.0))
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    
    if nrow == 1:
        axs = [axs]
    axs_flat = [element for row in axs for element in row]

    for index, runID in enumerate(run_range):
        ax = axs_flat[index]
        inputTrace = os.path.join(config.working_dir,
                                  config.output_prefix + '.GA.r' + str(runID) + '.trace')
        generation, popHighestFitness = get_population_highest_fitness(inputTrace)
        ax.plot(generation, popHighestFitness, linestyle=':', color='#377eb8')
        ax.scatter(generation, popHighestFitness, marker='8', color='#377eb8')
        ax.set_title('runID = %d' % runID, fontsize=12)
        ax.set_xlim(0.5, 0.5 + config.genetic_algorithm['generation_count'])
        ax.set_ylim(-0.05, 1.05)

    for element in axs_flat[index+1:]:
        element.axis('off')

    fig.suptitle('Population Highest Fitness after Each Generation', fontsize=14)
    fig.set_size_inches(10, nrow * 2.5)

    outputPath = os.path.join(config.working_dir,
                              config.output_prefix + '.GA.fitnessTrace.pdf')
    plt.savefig(outputPath)

#----------------------------------------------------------------------#
def plot_ga_top_tree_count_trace(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plotting GA top tree count trace requires matplotlib \n'
              'module. Failed to import matplotlib. Please install \n'
              'matplotlib package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    run_range = range(1, 1 + config.genetic_algorithm['instance_count'])
    
    generationCol = 0
    popHighestFitnessCol = 3

    nrow = int(np.ceil(len(run_range) / 4.0))
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    if nrow == 1:
        axs = [axs]
    axs_flat = [element for row in axs for element in row]

    for index, runID in enumerate(run_range):
        ax = axs_flat[index]
        inputTrace = os.path.join(config.working_dir,
                                  config.output_prefix + '.GA.r' + str(runID) + '.trace')
        generations, popHighestFitness = get_population_highest_fitness(inputTrace)
        treeStats = get_tree_stats(inputTrace)
        counts = []
        for gIndex in range(len(generations)):
            fitness = popHighestFitness[gIndex]
            trees = list(filter(lambda x: (x[1] >= fitness) and (x[2] <= generations[gIndex]), treeStats))
            counts.append(len(set([tree[0] for tree in trees])))
        ax.plot(generations, counts, linestyle=':', color='#4daf4a')
        ax.scatter(generations, counts, marker='8', color='#4daf4a')
        ax.set_title('runID = %d' % runID, fontsize=12)
        ax.set_xlim(0.5, 0.5 + config.genetic_algorithm['generation_count'])
        ax.set_ylim(-0.05, max(counts) + 0.05)

    for element in axs_flat[index+1:]:
        element.axis('off')

    fig.suptitle('Number of Tree Topologies with Maximum Fitness sampled after Each Generation', fontsize=14)
    fig.set_size_inches(10, nrow * 2.5)
    
    outputPath = os.path.join(config.working_dir,
                              config.output_prefix + '.GA.topTreeCount.pdf')
    plt.savefig(outputPath)

#----------------------------------------------------------------------#
def plot_consensus_tree(args):
    try:
        import igraph
    except ImportError:
        print('Plotting consensus tree requires python-igraph \n'
              'module. Failed to import igraph. Please install \n'
              'python-igraph package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    cTreePath = os.path.join(config.working_dir,
                             config.output_prefix + '.GA.consensusTree')
    cTreeGraphPath = os.path.join(config.working_dir,
                                  config.output_prefix + '.GA.consensusTree.pdf')

    edges, weights, labels = read_consensus_tree(cTreePath)
    vertices = list(set(list(zip(*edges))[0] + list(zip(*edges))[1]))
    root = list(set(vertices) - set(list(zip(*edges))[1]))[0]
        
    edges.append(('GL', root))
    weights.append(1.0)
    vertices.append('GL')
    labels.append('1/1')

    edgeClusterLabels = list(zip(*edges))[1]
    edgeLabels = list(map(lambda x: ':'.join(x), zip(edgeClusterLabels, labels)))

    g = igraph.Graph(directed=True)
    g.add_vertices(vertices)
    g.add_edges(edges)
    g.es['label'] = edgeLabels
    g.es['arrow_size'] = [0.75] * len(edges)
    g.es['color'] = 'darkgray'
    g.es['width'] = [2 * item for item in weights]

    g.vs['label'] = [''] * (len(vertices)-1) + ['GL']
    g.vs['color'] = 'cornflowerblue'

    layout = g.layout('rt', root=[len(vertices)-1])
        
    visual_style = {"bbox": (500, 500),
                    "margin": 75,
                    "edge_labels": labels,
                    "vertex_size": 30,
                    "layout": layout}

    try:
        igraph.plot(g, cTreeGraphPath, **visual_style)
    except TypeError:
        print('Plotting consensus tree requires Cairo library \n'
              'and Pycairo to support python-igraph. \n'
              'Dependencies missing. Please install \n'
              'Cairo and PyCairo package and try again\n'
              '(guide available: https://gist.github.com/Niknafs/6b50d9df9d5396a2e92e)\n',
              file=sys.stderr)
        return

#----------------------------------------------------------------------#
def read_cpov_matrix(path):
    with open(path, 'r') as f_h:
        lines = f_h.read().split('\n')
    content = list(map(lambda x: x.split('\t'),
                       list(filter(lambda y: y != '' and (not y.startswith('parent')) and (not y.startswith('#')), lines)))
                      )
    uniqueIDs = list(set(list(zip(*content))[0]))
    N = len(uniqueIDs)
    if sum([x.isdigit() for x in uniqueIDs]) == N:
        clusterIDs = sorted(uniqueIDs, key=lambda x: int(x))
    else:
        clusterIDs = sorted(uniqueIDs)
    
    cpov = np.zeros((N, N))
    for item in content:
        cpov[clusterIDs.index(item[0])][clusterIDs.index(item[1])] = float(item[2])
    return cpov, clusterIDs

#----------------------------------------------------------------------#
def read_consensus_tree(path):
    with open(path, 'r') as f_h:
        elements = list(map(lambda x: x.split('\t'),
                            list(filter(lambda y: y != '' and (not y.startswith('#')) and (not y.startswith('parent')), 
                                        f_h.read().split('\n')))))
    edges = list(zip(list(zip(*elements))[0], list(zip(*elements))[1]))
    weights = list(map(float, list(zip(*elements))[2]))
    labels = list(list(zip(*elements))[3])
    return edges, weights, labels

#----------------------------------------------------------------------#
def get_population_highest_fitness(path):
    generationCol = 0
    popHighestFitnessCol = 3

    with open(path, 'r') as f_h:
        content = f_h.read().split('############')[0].split('\n')
    content = list(filter(lambda x: (not x.startswith('#')) and (not x.startswith('generation.index')), content))
    block = list(map(lambda x: x.split('\t'),
                     list(filter(lambda y: y != '', content))))
    generation = list(map(int, list(zip(*block))[generationCol]))
    popHighestFitness = list(map(float, list(zip(*block))[popHighestFitnessCol]))
    return generation, popHighestFitness

#----------------------------------------------------------------------#
def get_tree_stats(path):
    TopologyCol = 0
    FitnessCol = -2
    appearanceCol = -1

    with open(path, 'r') as f_h:
        blockLine = '############################################################'
        content = f_h.read().split(blockLine)[1].split('\n')
    content = list(filter(lambda x: (x != '') and (not x.startswith('Topology')), content))
    content = list(map(lambda x: x.split('\t'), content))
    treeStats = list(map(lambda x: (x[TopologyCol], float(x[FitnessCol]),
                             min(list(map(lambda y: int(y.split(':')[0]), x[appearanceCol].split(','))))),
                     content))
    return treeStats

#----------------------------------------------------------------------#
def bound_fix(value):
    return min(max(value, 0.0), 1.0)

#----------------------------------------------------------------------#
def plot_mut_clust_cellularity(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plotting mutation/cluster cellularities requires matplotlib \n'
              'module. Failed to import matplotlib. Please install \n'
              'matplotlib package and try again', file=sys.stderr)
        return

    config = Config(args.config_file)
    if config.cellularity_estimation == 'schism':
        mutCellularityPath = os.path.join(config.working_dir,
                                          config.output_prefix + '.mutation.cellularity')
    else:
        mutCellularityPath = os.path.join(config.working_dir,
                                          config.mutation_cellularity_input)
    
    clustCellularityPath = os.path.join(config.working_dir,
                                        config.output_prefix + '.cluster.cellularity')
    clusterPath = os.path.join(config.working_dir, config.mutation_to_cluster_assignment)
    cluster2mut = read_cluster_assignments(clusterPath)

    cl2index, samples = read_input_samples(clustCellularityPath)
    clBounds = {}
    samples = sorted(samples, key=lambda x: x.name)
    x = list(range(len(samples)))
    for cl in cl2index.keys():
        lowerBound = ['NA'] * len(samples)
        upperBound = ['NA'] * len(samples)
        for id, sample in enumerate(samples):
            if sample.mutCellularity[cl2index[cl]] != -1:
                lowerBound[id] = sample.mutCellularity[cl2index[cl]] - sample.mutSigma[cl2index[cl]]
                upperBound[id] = sample.mutCellularity[cl2index[cl]] + sample.mutSigma[cl2index[cl]]
        clBounds[cl] = [x, lowerBound, upperBound]
    
    colorPalette = ['#323f7b', '#cb3245', '#638e4d', '#9a336d', '#e2a86a',
                    '#246c8f', '#7d303d', '#734d85', '#077783', '#9c7688',
                    '#b48b73', '#7da1bf', '#4b6b6c', '#7b7282', '#263246']

    if len(cl2index) > len(colorPalette):
        print('Cellularity plot not supported for more than 15 clusters.', file=sys.stderr)
        sys.exit()
    plt.grid(True)
    fig, ax = plt.subplots(1, 1)
    for cl, index in cl2index.items():
        trace = np.mean(np.array([clBounds[cl][1], clBounds[cl][2]]), 0)
        trace = list(map(bound_fix, trace))
        clBounds[cl][1] = list(map(bound_fix, clBounds[cl][1]))
        clBounds[cl][2] = list(map(bound_fix, clBounds[cl][2]))
        ax.fill_between(clBounds[cl][0],
                        clBounds[cl][1],
                        clBounds[cl][2], color=colorPalette[index],
                        alpha=0.5)
        ax.plot(clBounds[cl][0], list(trace),
                color=colorPalette[index], label=cl,
                linestyle="dotted", marker="o", linewidth=1)
    mut2index, samples = read_input_samples(mutCellularityPath)
    samples = sorted(samples, key=lambda x: x.name)
    for cl in cluster2mut:
        coords = []
        for mut in cluster2mut[cl]:
            xm = list(range(len(samples)))
            ym = [sample.mutCellularity[mut2index[mut]] for sample in samples]
            coords.extend(list(zip(xm, ym)))
        coords = list(filter(lambda x: x[1] != -1, coords))
        if coords:
            x_vals, y_vals = list(zip(*coords))
            ax.scatter(x_vals, y_vals, color=colorPalette[cl2index[cl]],
                       marker='+', s=20.0)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    ax.legend(handles, labels)
    
    plt.xticks(list(range(len(samples))))
    plt.yticks([x * 0.1 for x in range(11)])
    plt.xlim((-0.5, 2.5))
    plt.ylim((-0.1, 1.1))
    xtext = [sample.name for sample in samples]
    ax.set_xticklabels(xtext)
    plt.gca().yaxis.grid(True)
    outputPath = os.path.join(config.working_dir,
                              config.output_prefix + '.cellularity.png')
    plt.savefig(outputPath)
    return

#----------------------------------------------------------------------#
def reorder_pov_by_cluster(pov, mutIDs, config):
    clusterPath = os.path.join(config.working_dir,
                               config.mutation_to_cluster_assignment)
    if not os.path.exists(clusterPath):
        return pov, mutIDs
    
    cluster2mut = read_cluster_assignments(clusterPath)
    pairs = []
    for cl in cluster2mut:
        pairs.extend(list(zip(cluster2mut[cl], [cl] * len(cluster2mut[cl])))
                     )
    print(pairs)
    pairs = sorted(pairs, key=lambda x: x[1])
    print(pairs)
    sortedmutIDs = list(zip(*pairs))[0]
    print(sortedmutIDs)

    sortedPov = np.zeros(pov.shape)
    for pnID in sortedmutIDs:
        for cnID in sortedmutIDs:
            sortedPov[sortedmutIDs.index(pnID), sortedmutIDs.index(cnID)] = pov[mutIDs.index(pnID), mutIDs.index(cnID)]
    return sortedPov, sortedmutIDs

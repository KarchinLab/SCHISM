import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import Config
#----------------------------------------------------------------------#
def plot_cpov(args):
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print >>sys.stderr, \
            'Plotting consensus tree requires matplotlib \n' +\
            'module. Failed to import matplotlib. Please install \n'+\
            'matplotlib package and try again'
        sys.exit()

    config = Config(args.config_file)
    cpovPath = os.path.join(config.working_dir,\
                            config.output_prefix +'.HT.cpov')
    cpov, clusterIDs = read_cpov_matrix(cpovPath)
    
    N = cpov.shape[1]

    # block out diagonal elements as they are meaningless
    # when it comes to cluster pair topology costs
    for index in range(N): cpov[index, index] = 100
    
    width = min(7 , 1.5 * N)
    
    fig, ax = plt.subplots()
    
    # generate colormap that resembles ggplot2 mutedBlue
    # colormap (scale_fill_gradient2 colors)
    cdict1 = {'red':((0.0, 0.226, 0.226),
                     (1.0, 1.0,1.0)),
              'blue': ((0.0, 0.593, 0.593),
                       (1.0, 1.0, 1.0)),
               'green': ((0.0,0.226, 0.226),
                         (1.0, 1.0, 1.0))}

    mutedBlue = LinearSegmentedColormap('mutedBlue', cdict1)
    plt.register_cmap(cmap=mutedBlue)
    cmap = mutedBlue
    cmap.set_over('0.72')

    # generate heatmap
    heatmap = ax.pcolor(cpov, cmap = mutedBlue ,\
                        vmax = 1.0, vmin = -0.05,\
                        edgecolors = [0.4,0.4,0.4])
    
    ax.set_xticks(np.arange(N) + 0.5, minor = False)
    ax.set_yticks(np.arange(N) + 0.5, minor = False)

    ax.set_xticklabels(clusterIDs)
    ax.set_yticklabels(clusterIDs)

    
    # add labels to heatmap
    for y in range(N):
        for x in range(N):
            if (config.hypothesis_test['test_level'] == 'mutations') and \
               (x != y):
                # elements in cpov will be votes and thus fractional
                plt.text(x + 0.5, y + 0.5, '%.2f' % cpov[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                 )
            else:
                # elements in cpov will be binary, no label necessary
                continue

    ax.set_xlim([0,N])
    ax.set_ylim([0,N])

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    fig = plt.gcf()
    fig.set_size_inches(width, width)
    
    cpovFigure = os.path.join(config.working_dir,\
                              config.output_prefix + '.HT.cpov.pdf')
    fig.savefig(cpovFigure, dpi = 300)
#----------------------------------------------------------------------#
def plot_ga_fitness_trace(args):
    # plot the GA trace corresponding to the run(s) specified
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print >>sys.stderr, \
            'Plotting consensus tree requires matplotlib \n' +\
            'module. Failed to import matplotlib. Please install \n'+\
            'matplotlib package and try again'
        sys.exit()
    
    config = Config(args.config_file)
    Range = range(1, 1 + config.genetic_algorithm['instance_count'])
    
    fig = plt.figure()

    nrow = int(np.ceil(len(Range)/ 4.0))
    ncol = 4
    
    fig, axs = plt.subplots(nrow, ncol)
    axs_ = [element for row in axs for element in row]

    for index, runID in enumerate(Range):

        ax = axs_[index]

        inputTrace = os.path.join(config.working_dir,\
                                  config.output_prefix + '.GA.r' + str(runID) + '.trace')
        generation, popHighestFitness = get_population_highest_fitness(inputTrace)

        lineplot = ax.plot(generation, popHighestFitness, \
                           linestyle= ':', color = '#377eb8')
        scatter = ax.scatter(generation, popHighestFitness, marker = '8',\
                             color = '#377eb8') # scalarMap.to_rgba(runID))
        ax.set_title('runID = %d'%runID, fontsize = 12)
        ax.set_xlim(0.5, 0.5 + config.genetic_algorithm['generation_count'])
        ax.set_ylim(-0.05, 1.05)

    # remove axes in empty subplots
    blanks =  axs_[index+1:]
    for element in blanks:
        element.axis('off')

    # plot title
    fig.suptitle('Population Highest Fitness after Each Generation', fontsize =14)
    fig.set_size_inches(10, nrow * 2.5)

    outputPath = os.path.join(config.working_dir,\
                                  config.output_prefix + '.GA.fitnessTrace.pdf')
    plt.savefig(outputPath)
#----------------------------------------------------------------------#
def plot_ga_top_tree_count_trace(args):
    # plot the number of trees with fitness equal to the highest 
    # population fitness after each generation

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print >>sys.stderr, \
            'Plotting consensus tree requires matplotlib \n' +\
            'module. Failed to import matplotlib. Please install \n'+\
            'matplotlib package and try again'
        sys.exit()

    config = Config(args.config_file)
    Range = range(1, 1 + config.genetic_algorithm['instance_count'])
    
    generationCol = 0
    popHighestFitnessCol = 3

    fig = plt.figure()

    nrow = int(np.ceil(len(Range)/ 4.0))
    ncol = 4
    
    fig, axs = plt.subplots(nrow, ncol)
    axs_ = [element for row in axs for element in row]

    for index, runID in enumerate(Range):

        ax = axs_[index]

        inputTrace = os.path.join(config.working_dir,\
                                  config.output_prefix + '.GA.r' + str(runID) + '.trace')
        generations, popHighestFitness = get_population_highest_fitness(inputTrace)
        treeStats = get_tree_stats(inputTrace)

        counts = []
        for gIndex in range(len(generations)):
            fitness = popHighestFitness[gIndex]
            trees = filter(lambda x: (x[1] >= fitness) and\
                           (x[2] <= generations[gIndex]), treeStats)
            counts.append(len(set([tree[0] for tree in trees])))
        
        lineplot = ax.plot(generations, counts, \
                           linestyle= ':', color = '#4daf4a')
        scatter = ax.scatter(generations, counts, marker = '8',\
                                 color = '#4daf4a') # scalarMap.to_rgba(runID))
    
        ax.set_title('runID = %d'%runID, fontsize = 12)
        ax.set_xlim(0.5, 0.5 + config.genetic_algorithm['generation_count'])
        ax.set_ylim(-0.05, max(counts) + 0.05)

    # remove axes in empty subplots
    blanks =  axs_[index+1:]
    for element in blanks:
        element.axis('off')

    # plot title
    fig.suptitle('Number of Tree Topologies with Maximum Fitness sampled after Each Generation', fontsize =14)
    fig.set_size_inches(10, nrow * 2.5)
    
    outputPath = os.path.join(config.working_dir,\
                                  config.output_prefix + '.GA.topTreeCount.pdf')
    plt.savefig(outputPath)

#----------------------------------------------------------------------#
def plot_consensus_tree(args):
    # this module uses python igraph package to
    # draw consensus tree topology

    try:
        import igraph
    except ImportError:
        print >>sys.stderr, \
            'Plotting consensus tree requires python igraph \n' +\
            'module. Failed to import igraph. Please install \n'+\
            'python-igraph package and try again'
        sys.exit

    config = Config(args.config_file)
    cTreePath = os.path.join(config.working_dir,\
                             config.output_prefix + '.GA.consensusTree')
    cTreeGraphPath = os.path.join(config.working_dir,\
                             config.output_prefix + '.GA.consensusTree.pdf')

    edges, weights, labels = read_consensus_tree(cTreePath)
    vertices = list(set(zip(*edges)[0] + zip(*edges)[1]))
    root = list(set(vertices) - set(zip(*edges)[1]))[0]
        
    edges += [('GL', root)]
    weights += [1.0]
    vertices += ['GL']
    labels += ['1/1']

    edgeClusterLabels = zip(*edges)[1]

    edgeLabels = map(lambda x: ':'.join(x), \
                     zip(edgeClusterLabels, labels))

    g = igraph.Graph(directed = True)
    g.add_vertices(vertices)
    g.add_edges(edges)
    # layout = g.layout_reingold_tilford(root=['GL'])
    g.es['label'] = edgeLabels
    g.es['arrow_size'] = [0.75] * len(edges)
    g.es['color'] = 'darkgray'
    g.es['width'] = [2 * item for item in weights]

    g.vs['label'] = [''] * (len(vertices)-1) + ['GL']
    g.vs['color'] = 'cornflowerblue'

    layout = g.layout('rt',root=[len(vertices)-1])
        
    visual_style = {}
    visual_style["bbox"] = (500, 500)
    visual_style["margin"] = 75
    visual_style["edge_labels"] = labels
    visual_style["vertex_size"] = 30
    visual_style['layout'] = layout
        
    igraph.plot(g, cTreeGraphPath, **visual_style)
#----------------------------------------------------------------------#
def read_cpov_matrix(path):
    # return cpov np.array and clusterIDs list
    
    f_h = file(path, 'r')
    lines = f_h.read().split('\n')
    f_h.close()
    
    content = map(lambda x: x.split('\t'), \
                  filter(lambda y: y != '' and (not y.startswith('parent')) \
                         and (not y.startswith('#')), lines))

    # extract sorted list of clusterIDs 
    uniqueIDs = list(set(zip(*content)[0]))
    N = len(uniqueIDs)
    if sum([x.isdigit() for x in uniqueIDs]) == N:
        clusterIDs = sorted(uniqueIDs, key = lambda x: int(x))
    else:
        clusterIDs = sorted(uniqueIDs)
    
    # potentially add another argument for the order 
    # user wants the items to apper
    cpov = np.zeros((N, N))
    for item in content:
        cpov[clusterIDs.index(item[0])][clusterIDs.index(item[1])] = float(item[2])
    return cpov, clusterIDs
#----------------------------------------------------------------------#
def read_consensus_tree(path):
    f_h = file(path,'r')
    elements = map(lambda x: x.split('\t'),\
                   filter(lambda y: y!= '' and (not y.startswith('#')) \
                          and (not y.startswith('parent')), \
                          f_h.read().split('\n')))
    edges = zip(zip(*elements)[0], zip(*elements)[1])
    weights = map(float, zip(*elements)[2])
    labels = list(zip(*elements)[3])
    
    return edges, weights, labels
#----------------------------------------------------------------------#
def get_population_highest_fitness(path):
    
    generationCol = 0
    popHighestFitnessCol = 3

    f_h = file(path)
    # extract the first block of file reporting population
    # and generation fitness parameters
    content = f_h.read().split('############')[0].split('\n')
    f_h.close()

    content = filter(lambda x: (not x.startswith('#')) and \
                               (not x.startswith('generation.index')), content)

    block = map(lambda x: x.split('\t'), \
                filter(lambda y: y!= '', content))

    generation = map(int,zip(*block)[generationCol])
    popHighestFitness = map(float, zip(*block)[popHighestFitnessCol])

    return generation, popHighestFitness

#----------------------------------------------------------------------#
def get_tree_stats(path):
    TopologyCol = 0
    FitnessCol = -2
    appearanceCol = -1

    f_h = file(path)
    # extract the first block of file reporting population
    # and generation fitness parameters
    blockLine = '############################################################'
    content = f_h.read().split(blockLine)[1].split('\n')
    f_h.close()
    content = filter(lambda x: (x!='') and \
                               (not x.startswith('Topology')), content)
    content = map(lambda x: x.split('\t'), content)
    # for each tree, record:
    # 1) topology in newick format
    # 2) fitness value
    # 3) earliest generation it appeared (1-based)
    treeStats = map(lambda x: (x[TopologyCol], float(x[FitnessCol]),\
                             min(map(lambda y: int(y.split(':')[0]),\
                                     x[appearanceCol].split(',')))),\
                  content)
    
    return treeStats

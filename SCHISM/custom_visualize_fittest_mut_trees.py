import argparse
import sys
import os
import numpy as np

from GATools import get_consensus_edges

sys.path = [p for p in sys.path if 'conda' in p]
#print sys.path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest = 'GA_trace_path', required=True)
    parser.add_argument('-c', dest = 'case_id', required=True)
    parser.add_argument('-o', dest= 'out_dir_path', required=True)
    parser.add_argument('-m', dest= 'mutid_to_mut_path', required=True)
    args = parser.parse_args()
    return args

def read_mutid_to_mut(mutid_to_mut_path):
    mutid_to_mut = {}
    with open(mutid_to_mut_path, 'r') as mip:
        lines = mip.readlines()
    mutid_to_mut = {line.split(':')[0].strip():line.split(':')[1].strip()
                    for line in lines if ':' in line}
    mutid_to_mut['0'] = 'Normal'
    return mutid_to_mut

def get_fittest_trees(GA_trace):
    f_h = open(GA_trace)
    line = f_h.readline()
    while not line.startswith('Topology'):
        line = f_h.readline()
    content = [x.split('\t') for x in [y for y in f_h.read().split('\n') if y!= '']]
    f_h.close()

    # narrow down the set of all topologies having maximum fitness sampled
    highestFitness = max(list(map(float,list(zip(*content))[4])))
    
    # throw error if max tree fitness is 0
    try:
        if highestFitness == 0:
    	    raise ValueError
    except ValueError:
        print >> sys.stderr, \
            'All trees have 0 fitness. Not plotting individual trees.'
        sys.exit()

    fittestTrees = [x for x in content if float(x[4]) == highestFitness]

    return fittestTrees

def translate_topology(mutid_to_mut, topology):
	topology_transl = ''
        for c in topology:
        	if c in mutid_to_mut.keys():
        		topology_transl += mutid_to_mut[c]
        	else:
        		topology_transl += c
	return topology_transl

def write_edgelist(pairs, path):
    with open(path, 'w') as op:
    	op.write('parent\tchild\n')
        op.write('\n'.join(['\t'.join(pair) for pair in pairs]))

def get_vertex_color(vertex_label, substrings):
    if 'normal' in vertex_label.lower():
        return 'gray75'
    for ss in substrings:
        if ss in vertex_label:
            return 'gray15'
    return 'gray50'

def plot_mut_tree(vertices, edges, vertex_labels, vertex_colors, cTreeGraphPath):
    try:
        import igraph
    except ImportError:
        print >> sys.stderr, \
            'Plotting consensus tree requires python-igraph \n' +\
            'module. Failed to import igraph. Please install \n'+\
            'python-igraph package and try again'
        return

    g = igraph.Graph(directed = True)
    g.add_vertices([v for v in vertices])
    g.add_edges(edges)
    
    #g.es['label'] = edgeLabels
    #print weights
    #g.es['width'] = [15*w for w in weights]
    g.es['width'] = [5 for item in edges]
    g.es['arrow_size'] = [2.5 for item in edges]
    g.es['color'] = 'gray50'
    #g.es['label_color'] = 'gray80' 
    
    g.vs['label'] = vertex_labels 
    #print g.vs['label'], g.vs['name']
    g.vs['color'] = vertex_colors #['gray60'] + ['gray20' for i in range(len(vertices)-1)]
    g.vs['frame_color'] = vertex_colors #['gray60'] + ['gray20' for i in range(len(vertices)-1)]
    g.vs['label_color'] = 'white'
    
    #layout = g.layout_reingold_tilford(mode="out", root=[0])
    #layout = g.layout('rt', root = [0])
    layout = g.layout_sugiyama()
    visual_style = {}
    #box_width=len(vertex_labels)*75 + len(labels)*50
    #box_height=len(vertex_labels)*75 + len(labels)*50
    #visual_style["bbox"] = (box_width, box_height)
    visual_style["bbox"] = (800, 800)
    visual_style["margin"] = 60
    #visual_style["edge_labels"] = labels
    visual_style["vertex_labels"] = vertex_labels
    visual_style["vertex_size"] = 105
    visual_style['layout'] = layout

    try:
        igraph.plot(g, cTreeGraphPath, **visual_style)
    except TypeError:
        print >>sys.stderr, \
            'Plotting consensus tree requires Cairo library \n' +\
            'and Pycairo to support python-graph. \n' + \
            'Dependencies missing. Please install \n'+\
            'Cairo and PyCairo package and try again\n' +\
            '(guide available: https://gist.github.com/Niknafs/6b50d9df9d5396a2e92e)\n'
        return

def plot_fittest_mut_trees(args):
    # this module uses python igraph package to
    # draw topologies of the fittest tree(s)
    
    
    # plots of fittest trees will be in outdir/fittest_trees
    outdir = args.out_dir_path
    outpath = outdir + '/fittest_trees'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ## identify fittest trees from .GA.trace
    GA_trace = args.GA_trace_path
    fittestTrees = get_fittest_trees(GA_trace)

    mutid_to_mut = read_mutid_to_mut(args.mutid_to_mut_path)

    ## for each tree plot topology and include metrics
    for i in range(len(fittestTrees)):
    	topology, massCost, topologyCost, cost, fitness, numRuns = fittestTrees[i]

        # output tree file
        treename = args.case_id + '_tree' + str(i+1) + '_fitness' + str(round(float(fitness), 3))
        cTreeGraphPath = os.path.join(outpath, treename + '.pdf')
        print "Generating plot for fittest tree " + str(i) + " : " + cTreeGraphPath

        ## translate topology to reflect mutation names
        topology_transl = translate_topology(mutid_to_mut, topology)
        parent_child = [t.split('\t')[0:2] for t in get_consensus_edges([topology_transl])]
        
        ## write edgelist to file 
        tree_tsv_out = os.path.join(outpath, treename + '_edges.tsv')
        write_edgelist(parent_child, tree_tsv_out)

        parents = [pair[0] for pair in parent_child]
        children = [pair[1] for pair in parent_child]
        
        vertices = list(set(parents) | set(children))
        vertex_labels = vertices
        vertex_colors = [get_vertex_color(vl, ['KRAS','GNAS']) for vl in vertex_labels]

        edges = parent_child
        
        plot_mut_tree(vertices, edges, vertex_labels, vertex_colors, cTreeGraphPath)


if __name__ == "__main__":
    config = parse_args()
    plot_fittest_mut_trees(config)
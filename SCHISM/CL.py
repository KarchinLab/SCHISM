import os
import sys

import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn import metrics

from utils import Config

from HT import aggregate_votes
from HT import average_cellularity
from HT import read_schism_decisions

#----------------------------------------------------------------------#
def cluster_muts(args):
    config = Config(args.config_file)
    HT_decisions= os.path.join(config.working_dir,\
                               config.output_prefix + '.HT.pov')
    povList = read_schism_decisions(HT_decisions)
    mut2index, aVoteMatrix = augmented_vote_matrix(povList) 

    if config.clustering_method['algorithm'] == 'AP':
        labels, silhouetteCoeff = ap_cluster_avm(config, aVoteMatrix)
        print >>sys.stderr, 'AP clustering --> cluster_count: %d, silhouette_coefficient: %0.4f'%\
                            (len(set(labels)), silhouetteCoeff)
    elif config.clustering_method['algorithm'] == 'DBSCAN':
        labels, silhouetteCoeff = dbscan_cluster_avm(config, aVoteMatrix)
        print >>sys.stderr, 'DBSCAN clustering --> cluster_count: %d, silhouette_coefficient: %0.4f'%\
                            (len(set(labels)), silhouetteCoeff)
    elif config.clustering_method['algorithm'] == 'KMeans':
        labels, silhouetteCoeff = kmeans_cluster_avm(config, aVoteMatrix)
        print >>sys.stderr, 'KMeans clustering --> cluster_count: %d, silhouette_coefficient: %0.4f'%\
                            (len(set(labels)), silhouetteCoeff)

    # in cases were mutIDs are numeric, make clusterIDs ascending when possible
    labels = remap_labels(mut2index, labels)
    
    mutation_to_cluster_file = os.path.join(config.working_dir, config.mutation_to_cluster_assignment)
    write_cluster_definitions(mut2index, labels, mutation_to_cluster_file)


#----------------------------------------------------------------------#
def integrate_cluster_metrics(args):
    config = Config(args.config_file)
    clusterCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')
    # average cellularity values over confirmed clusters
    average_cellularity(config,clusterCellularityPath)
    # aggregate HT votes over cluster definitions
    aggregate_votes(config)

#----------------------------------------------------------------------#
def augmented_vote_matrix(pov):
    mutIDs = list(set(zip(*pov)[0]))
    mut2index = dict(zip(mutIDs, range(len(mutIDs))))
    N = len(mutIDs)
    # augmented votes matrix, each row equals to the corresponding row in pov
    # followed by corresponding column in pov matrix
    aVoteMatrix = np.zeros((N, 2 * N))
    for parent, child, rejection in pov:
        parentid = mut2index[parent]
        childid = mut2index[child]
        vote = float(rejection)
        aVoteMatrix[parentid, childid] = vote
        aVoteMatrix[childid, parentid + N] = vote
    return mut2index, aVoteMatrix

#----------------------------------------------------------------------#
def write_cluster_definitions(mut2index, labels, path):
    f_w = file(path, 'w')
    print >>f_w, '\t'.join(['mutationID', 'clusterID'])
    
    index2mut = dict(zip( mut2index.values(), mut2index.keys()))

    # make cluster ids 0-based
    mid = min(labels)
    labels = labels - mid

    for index in sorted(index2mut.keys()):
        mutID = index2mut[index]
        clusterID = labels[index]
        print >>f_w, '\t'.join(map(str, [mutID, clusterID]))
    f_w.close()
#----------------------------------------------------------------------#
def ap_cluster_avm(config, aVoteMatrix):
    
    prefMin = config.clustering_method['min_preference']
    prefMax = config.clustering_method['max_preference']
    inc = config.clustering_method['preference_increments']
    minClusterCount = config.clustering_method['min_cluster_count']
    maxClusterCount = config.clustering_method['max_cluster_count'] + 1

    if config.clustering_method['verbose'] == True:
        print >>sys.stderr, 'Clustering mutations by application of affinity propagation to POV matrix'

    N = aVoteMatrix.shape[0]

    # potentially improve this section
    prefs = range(prefMin, prefMax, inc)
    scores = []
    
    for value in prefs:
        
        af = AffinityPropagation(preference=value).fit(aVoteMatrix)
        cluster_centers_indices = af.cluster_centers_indices_
        
        if cluster_centers_indices == None:
            print >>sys.stderr , 'invalid preference %f'% value
            continue

        labels = af.labels_
        n_clusters = len(cluster_centers_indices)
        if n_clusters > 1 and n_clusters < N:
            silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')
            scores.append([value, silhouetteCoeff, n_clusters])
            if config.clustering_method['verbose'] == True:
                print >>sys.stderr, 'preference: %0.2f, n_clusters: %d, silhouette_coeff: %0.6f'%(value, n_clusters, silhouetteCoeff)
        else:
            scores.append([value, 'NA',n_clusters])

    # -------------picking a reasonable solution-------------#
    try:
        possible_solutions = filter(lambda x: x[1] != 'NA', scores)
        possible_solutions = filter(lambda x: x[2] <= config.clustering_method['max_cluster_count'] and \
                                          x[2] >= config.clustering_method['min_cluster_count'], possible_solutions)
        max_s_coeff = max(zip(*possible_solutions)[1])
        # find the maximum silhouette coefficient 
        top_solutions = filter(lambda x: x[1] == max_s_coeff, possible_solutions)
        # the solution with maximum silhouette coefficicent, and least number of clusters will be picked
        
        top_s_cluster_count = list(set(zip(*top_solutions)[2]))
        if len(top_s_cluster_count) != 1 and config.clustering_method['verbose'] == True:
            print >>sys.stderr, 'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: %s'%(','.join(map(str, top_s_cluster_count)))
            print >>sys.stderr, 'Picking the solution with smallest cluster count'

        min_cluster_count = min(zip(*top_solutions)[2])
        pref_pick = filter(lambda x: x[2] == min_cluster_count, top_solutions)[0][0]
        
        if config.clustering_method['verbose'] == True:
                print >>sys.stderr, 'best solution contains %d clusters.'%min_cluster_count


        af = AffinityPropagation(preference=pref_pick).fit(aVoteMatrix)
        labels = af.labels_
        silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')

        return labels, silhouetteCoeff

    except:
        print >>sys.stderr, 'No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values'
        sys.exit()
#----------------------------------------------------------------------#
def dbscan_cluster_avm(config, aVoteMatrix):
    minClusterCount = config.clustering_method['min_cluster_count']
    maxClusterCount = config.clustering_method['max_cluster_count'] + 1
    
    min_minPts = config.clustering_method['min_minPts']
    max_minPts = config.clustering_method['max_minPts']
    inc_minPts = config.clustering_method['minPts_increments']
    
    min_eps = config.clustering_method['min_eps']
    max_eps = config.clustering_method['max_eps']
    inc_eps = config.clustering_method['eps_increments']
    
    if config.clustering_method['verbose'] == True:
        print >>sys.stderr, 'Clustering mutations by application of DBSCAN to POV matrix'

    N = aVoteMatrix.shape[0]

    # grid search over possible values for minPts, and eps to find 
    # a solution with highest silhouette coefficient
    minPtsRange = range(min_minPts, max_minPts, inc_minPts)
    epsRange = np.arange(min_eps, max_eps, inc_eps)

    scores = []
    for pts in minPtsRange:
        for eps in epsRange:
            db =  DBSCAN(eps = eps, min_samples = pts).fit(aVoteMatrix)
            try:
                labels = db.labels_
                n_clusters = len(set(labels))
                #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            except:
                print >>sys.stderr , 'invalid (minPts,eps) value (%d, %0.2f)'% (pts, eps)
                continue
            if n_clusters > 1 and n_clusters < N:
                silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')
                scores.append([pts,eps,  silhouetteCoeff, n_clusters])
                if config.clustering_method['verbose'] == True:
                    print >>sys.stderr, 'minPts: %d, eps: %0.2f, n_clusters: %d, silhouette_coeff: %0.6f'%(pts, eps, n_clusters, silhouetteCoeff)
            else:
                scores.append([pts, eps, 'NA',n_clusters])
    

    # -------------picking a reasonable solution-------------#
    try:
        possible_solutions = filter(lambda x: x[2] != 'NA', scores)
        possible_solutions = filter(lambda x: x[3] <= config.clustering_method['max_cluster_count'] and \
                                          x[3] >= config.clustering_method['min_cluster_count'], possible_solutions)

        max_s_coeff = max(zip(*possible_solutions)[2])
        # find the maximum silhouette coefficient 
        top_solutions = filter(lambda x: x[2] == max_s_coeff, possible_solutions)
        # the solution with maximum silhouette coefficicent, and least number of clusters will be picked
        
        top_s_cluster_count = list(set(zip(*top_solutions)[3]))
        if len(top_s_cluster_count) != 1 and config.clustering_method['verbose'] == True:
            print >>sys.stderr, 'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: %s'%(','.join(map(str, top_s_cluster_count)))
            print >>sys.stderr, 'Picking the solution with smallest cluster count'

        min_cluster_count = min(zip(*top_solutions)[3])
        pick = filter(lambda x: x[3] == min_cluster_count, top_solutions)[0]
        minPts, eps, silhoetteCoeff, n_clusters = pick

        db = DBSCAN(eps = eps, min_samples = minPts).fit(aVoteMatrix)
        labels = db.labels_
        silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')
        n_clusters = len(set(labels))

        if config.clustering_method['verbose'] == True:
                print >>sys.stderr, 'best solution contains %d clusters.'%n_clusters

        return labels, silhouetteCoeff

    except:
        print >>sys.stderr, 'No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values'
        sys.exit()

#----------------------------------------------------------------------#
def kmeans_cluster_avm(config, aVoteMatrix):
    
    n_init = config.clustering_method['n_init']
    minClusterCount = config.clustering_method['min_cluster_count']
    maxClusterCount = config.clustering_method['max_cluster_count']

    if config.clustering_method['verbose'] == True:
        print >>sys.stderr, 'Clustering mutations by application of KMeans method to POV matrix'

    N = aVoteMatrix.shape[0]

    # potentially improve this section
    cl_count = range(minClusterCount, maxClusterCount+1)
    scores = []
    
    for count in cl_count:
        km = KMeans(init = 'k-means++', n_clusters = count, n_init = n_init ).fit(aVoteMatrix)
        
        try:
            #cluster_centers_indices = km.cluster_centers_indices_
            labels = km.labels_
            n_clusters = len(set(labels))
        except:
            print >>sys.stderr , 'invalid cluster_count value %d'% (count)
            continue

        if n_clusters > 1 and n_clusters < N:
            silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')
            scores.append([n_clusters, silhouetteCoeff])
            if config.clustering_method['verbose'] == True:
                print >>sys.stderr, 'n_clusters: %d, silhouette_coeff: %0.6f'%(n_clusters, silhouetteCoeff)
        else:
            scores.append([n_clusters, 'NA'])

    # -------------picking a reasonable solution-------------#
    try:
        possible_solutions = filter(lambda x: x[1] != 'NA', scores)
        possible_solutions = filter(lambda x: x[0] <= config.clustering_method['max_cluster_count'] and \
                                          x[0] >= config.clustering_method['min_cluster_count'], possible_solutions)

        max_s_coeff = max(zip(*possible_solutions)[1])
        # find the maximum silhouette coefficient 
        top_solutions = filter(lambda x: x[1] == max_s_coeff, possible_solutions)
        # the solution with maximum silhouette coefficicent, and least number of clusters will be picked
        
        top_s_cluster_count = list(set(zip(*top_solutions)[1]))
        if len(top_s_cluster_count) != 1 and config.clustering_method['verbose'] == True:
            print >>sys.stderr, 'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: %s'%(','.join(map(str, top_s_cluster_count)))
            print >>sys.stderr, 'Picking the solution with smallest cluster count'

        min_cluster_count = min(zip(*top_solutions)[0])
        count_pick = filter(lambda x: x[0] == min_cluster_count, top_solutions)[0][0]
        
        if config.clustering_method['verbose'] == True:
                print >>sys.stderr, 'best solution contains %d clusters.'%count_pick

        km = KMeans(init = 'k-means++', n_clusters = count_pick, n_init = n_init ).fit(aVoteMatrix)
        labels = km.labels_
        silhouetteCoeff = metrics.silhouette_score(aVoteMatrix, labels, metric='sqeuclidean')

        return labels, silhouetteCoeff

    except:
        print >>sys.stderr, 'No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values'
        sys.exit()
        
#----------------------------------------------------------------------#
def remap_labels(mut2index, labels):
    '''
    in cases where the mutation ideas are numeric, rename cluster labels
    such that those are ascending 
    '''
    mutIDs = mut2index.keys()
    N = len(mutIDs)

    if sum([x.isdigit() for x in mutIDs]) != N:
        # if there are non-numeric mutation ids --> skip this step 
        return labels
    
    mut2clust = {}
    for mut in mut2index:
        mut2clust[mut] = labels[mut2index[mut]]
    mut2clust = sorted(mut2clust.items(), key = lambda x: int(x[0]))

    index = 0
    clustmap = {}
    for val in zip(*mut2clust)[1]:
        if val not in clustmap:
            clustmap[val] = index
            index += 1

    newLabels = np.array([clustmap[l] for l in labels])
    return newLabels
#----------------------------------------------------------------------#

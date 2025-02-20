import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple

from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn import metrics

from SCHISM.utils import Config
from SCHISM.HT import aggregate_votes, average_cellularity, read_schism_decisions

#----------------------------------------------------------------------#
def cluster_muts(args: Any) -> None:
    config = Config(args.config_file)
    HT_decisions = os.path.join(config.working_dir,
                                config.output_prefix + '.HT.pov')
    pov_list = read_schism_decisions(HT_decisions)
    mut2index, a_vote_matrix = augmented_vote_matrix(pov_list)

    algorithm = config.clustering_method['algorithm']
    if algorithm == 'AP':
        labels, silhouette_coeff = ap_cluster_avm(config, a_vote_matrix)
        print(f'AP clustering --> cluster_count: {len(set(labels))}, silhouette_coefficient: {silhouette_coeff:0.4f}',
              file=sys.stderr)
    elif algorithm == 'DBSCAN':
        labels, silhouette_coeff = dbscan_cluster_avm(config, a_vote_matrix)
        print(f'DBSCAN clustering --> cluster_count: {len(set(labels))}, silhouette_coefficient: {silhouette_coeff:0.4f}',
              file=sys.stderr)
    elif algorithm == 'KMeans':
        labels, silhouette_coeff = kmeans_cluster_avm(config, a_vote_matrix)
        print(f'KMeans clustering --> cluster_count: {len(set(labels))}, silhouette_coefficient: {silhouette_coeff:0.4f}',
              file=sys.stderr)
    else:
        print(f"Unknown clustering algorithm: {algorithm}", file=sys.stderr)
        sys.exit(1)

    # In cases where mutIDs are numeric, make clusterIDs ascending when possible
    labels = remap_labels(mut2index, labels)

    mutation_to_cluster_file = os.path.join(config.working_dir, config.mutation_to_cluster_assignment)
    write_cluster_definitions(mut2index, labels, mutation_to_cluster_file)


#----------------------------------------------------------------------#
def integrate_cluster_metrics(args: Any) -> None:
    config = Config(args.config_file)
    cluster_cellularity_path = os.path.join(config.working_dir,
                                             config.output_prefix + '.cluster.cellularity')
    # Average cellularity values over confirmed clusters
    average_cellularity(config, cluster_cellularity_path)
    # Aggregate HT votes over cluster definitions
    aggregate_votes(config)


#----------------------------------------------------------------------#
def augmented_vote_matrix(pov: List[Tuple[str, str, Any]]) -> Tuple[Dict[str, int], np.ndarray]:
    # Extract unique mutation IDs from the first column of pov.
    columns = list(zip(*pov))
    mutIDs = list(set(columns[0]))
    mut2index = {mut: idx for idx, mut in enumerate(mutIDs)}
    N = len(mutIDs)
    # Augmented votes matrix: each row equals the corresponding row in pov
    # followed by corresponding column in the pov matrix.
    a_vote_matrix = np.zeros((N, 2 * N))
    for parent, child, rejection in pov:
        parentid = mut2index[parent]
        childid = mut2index[child]
        vote = float(rejection)
        a_vote_matrix[parentid, childid] = vote
        # Note: originally the code had a line that might be a typo.
        # The following line is kept as in your comments:
        a_vote_matrix[childid, childid] = vote  # childid+N remains as intended?
        # If the original intention was to index by childid, parentid+N, use the next line:
        a_vote_matrix[childid, parentid + N] = vote
    return mut2index, a_vote_matrix


#----------------------------------------------------------------------#
def write_cluster_definitions(mut2index: Dict[str, int],
                              labels: np.ndarray,
                              path: str) -> None:
    with open(path, 'w') as f_w:
        print('\t'.join(['mutationID', 'clusterID']), file=f_w)

        index2mut = {v: k for k, v in mut2index.items()}

        # Make cluster ids 0-based.
        mid = min(labels)
        labels = labels - mid

        for index in sorted(index2mut.keys()):
            mutID = index2mut[index]
            clusterID = labels[index]
            print(f'{mutID}\t{clusterID}', file=f_w)


#----------------------------------------------------------------------#
def ap_cluster_avm(config: Config, a_vote_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    pref_min = config.clustering_method['min_preference']
    pref_max = config.clustering_method['max_preference']
    inc = config.clustering_method['preference_increments']
    min_cluster_count = config.clustering_method['min_cluster_count']
    max_cluster_count = config.clustering_method['max_cluster_count'] + 1

    if config.clustering_method.get('verbose', False):
        print('Clustering mutations by application of affinity propagation to POV matrix', file=sys.stderr)

    N = a_vote_matrix.shape[0]
    max_cluster_count = min(N, max_cluster_count)

    # Grid search over preferences.
    prefs = range(pref_min, pref_max, inc)
    scores = []

    for value in prefs:
        af = AffinityPropagation(preference=value).fit(a_vote_matrix)
        cluster_centers_indices = af.cluster_centers_indices_

        if cluster_centers_indices is None:
            print(f'invalid preference {value:.2f}', file=sys.stderr)
            continue

        labels = af.labels_
        n_clusters = len(cluster_centers_indices)
        if 1 < n_clusters < N:
            silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')
            scores.append([value, silhouette_coeff, n_clusters])
            if config.clustering_method.get('verbose', False):
                print(f'preference: {value:0.2f}, n_clusters: {n_clusters}, silhouette_coeff: {silhouette_coeff:0.6f}', file=sys.stderr)
        else:
            scores.append([value, 'NA', n_clusters])

    # -------------Picking a reasonable solution-------------#
    try:
        possible_solutions = [x for x in scores if x[1] != 'NA']
        possible_solutions = [x for x in possible_solutions if config.clustering_method['max_cluster_count'] >= x[2] >= config.clustering_method['min_cluster_count']]
        max_s_coeff = max(zip(*possible_solutions)[1])
        # Find the maximum silhouette coefficient.
        top_solutions = [x for x in possible_solutions if x[1] == max_s_coeff]
        # If multiple solutions tie, pick the one with the smallest cluster count.
        top_s_cluster_count = set(x[2] for x in top_solutions)
        if len(top_s_cluster_count) != 1 and config.clustering_method.get('verbose', False):
            print(f'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: {",".join(map(str, top_s_cluster_count))}', file=sys.stderr)
            print('Picking the solution with smallest cluster count', file=sys.stderr)

        min_cluster_count_solution = min(x[2] for x in top_solutions)
        pref_pick = [x[0] for x in top_solutions if x[2] == min_cluster_count_solution][0]

        if config.clustering_method.get('verbose', False):
            print(f'best solution contains {min_cluster_count_solution} clusters.', file=sys.stderr)

        af = AffinityPropagation(preference=pref_pick).fit(a_vote_matrix)
        labels = af.labels_
        silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')
        return labels, silhouette_coeff

    except Exception as e:
        print('No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values', file=sys.stderr)
        sys.exit(1)


#----------------------------------------------------------------------#
def dbscan_cluster_avm(config: Config, a_vote_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    min_cluster_count = config.clustering_method['min_cluster_count']
    max_cluster_count = config.clustering_method['max_cluster_count'] + 1

    min_minPts = config.clustering_method['min_minPts']
    max_minPts = config.clustering_method['max_minPts']
    inc_minPts = config.clustering_method['minPts_increments']

    min_eps = config.clustering_method['min_eps']
    max_eps = config.clustering_method['max_eps']
    inc_eps = config.clustering_method['eps_increments']

    if config.clustering_method.get('verbose', False):
        print('Clustering mutations by application of DBSCAN to POV matrix', file=sys.stderr)

    N = a_vote_matrix.shape[0]
    max_cluster_count = min(N, max_cluster_count)

    # Grid search over possible values for minPts and eps.
    minPts_range = range(min_minPts, max_minPts, inc_minPts)
    eps_range = np.arange(min_eps, max_eps, inc_eps)

    scores = []
    for pts in minPts_range:
        for eps in eps_range:
            db = DBSCAN(eps=eps, min_samples=pts).fit(a_vote_matrix)
            try:
                labels = db.labels_
                n_clusters = len(set(labels))
            except Exception:
                print(f'invalid (minPts, eps) value ({pts}, {eps:0.2f})', file=sys.stderr)
                continue
            if 1 < n_clusters < N:
                silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')
                scores.append([pts, eps, silhouette_coeff, n_clusters])
                if config.clustering_method.get('verbose', False):
                    print(f'minPts: {pts}, eps: {eps:0.2f}, n_clusters: {n_clusters}, silhouette_coeff: {silhouette_coeff:0.6f}', file=sys.stderr)
            else:
                scores.append([pts, eps, 'NA', n_clusters])

    # -------------Picking a reasonable solution-------------#
    try:
        possible_solutions = [x for x in scores if x[2] != 'NA']
        possible_solutions = [x for x in possible_solutions if config.clustering_method['max_cluster_count'] >= x[3] >= config.clustering_method['min_cluster_count']]

        max_s_coeff = max(zip(*possible_solutions)[2])
        # Find the maximum silhouette coefficient.
        top_solutions = [x for x in possible_solutions if x[2] == max_s_coeff]
        # If multiple solutions tie, pick the one with the smallest cluster count.
        top_s_cluster_count = set(x[3] for x in top_solutions)
        if len(top_s_cluster_count) != 1 and config.clustering_method.get('verbose', False):
            print(f'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: {",".join(map(str, top_s_cluster_count))}', file=sys.stderr)
            print('Picking the solution with smallest cluster count', file=sys.stderr)

        min_cluster_count_solution = min(x[3] for x in top_solutions)
        pick = [x for x in top_solutions if x[3] == min_cluster_count_solution][0]
        minPts, eps, silhouette_coeff, n_clusters = pick

        db = DBSCAN(eps=eps, min_samples=minPts).fit(a_vote_matrix)
        labels = db.labels_
        silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')
        n_clusters = len(set(labels))

        if config.clustering_method.get('verbose', False):
            print(f'best solution contains {n_clusters} clusters.', file=sys.stderr)

        return labels, silhouette_coeff

    except Exception as e:
        print('No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values', file=sys.stderr)
        sys.exit(1)


#----------------------------------------------------------------------#
def kmeans_cluster_avm(config: Config, a_vote_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    n_init = config.clustering_method['n_init']
    min_cluster_count = config.clustering_method['min_cluster_count']
    max_cluster_count = config.clustering_method['max_cluster_count']

    if config.clustering_method.get('verbose', False):
        print('Clustering mutations by application of KMeans method to POV matrix', file=sys.stderr)

    N = a_vote_matrix.shape[0]
    max_cluster_count = min(N, max_cluster_count)

    # Grid search over possible cluster counts.
    cluster_counts = range(min_cluster_count, max_cluster_count + 1)
    scores = []

    for count in cluster_counts:
        km = KMeans(init='k-means++', n_clusters=count, n_init=n_init).fit(a_vote_matrix)
        try:
            labels = km.labels_
            n_clusters = len(set(labels))
        except Exception:
            print(f'invalid cluster_count value {count}', file=sys.stderr)
            continue

        if 1 < n_clusters < N:
            silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')
            scores.append([n_clusters, silhouette_coeff])
            if config.clustering_method.get('verbose', False):
                print(f'n_clusters: {n_clusters}, silhouette_coeff: {silhouette_coeff:0.6f}', file=sys.stderr)
        else:
            scores.append([n_clusters, 'NA'])

    # -------------Picking a reasonable solution-------------#
    try:
        possible_solutions = [x for x in scores if x[1] != 'NA']
        possible_solutions = [x for x in possible_solutions if config.clustering_method['max_cluster_count'] >= x[0] >= config.clustering_method['min_cluster_count']]

        max_s_coeff = max(zip(*possible_solutions)[1])
        # Find the maximum silhouette coefficient.
        top_solutions = [x for x in possible_solutions if x[1] == max_s_coeff]
        # If multiple solutions tie, pick the one with the smallest cluster count.
        top_s_cluster_count = set(x[0] for x in top_solutions)
        if len(top_s_cluster_count) != 1 and config.clustering_method.get('verbose', False):
            print(f'Multiple solutions with varying cluster counts tie for best performance. Range of cluster count values includes: {",".join(map(str, top_s_cluster_count))}', file=sys.stderr)
            print('Picking the solution with smallest cluster count', file=sys.stderr)

        count_pick = min(x[0] for x in top_solutions)

        if config.clustering_method.get('verbose', False):
            print(f'best solution contains {count_pick} clusters.', file=sys.stderr)

        km = KMeans(init='k-means++', n_clusters=count_pick, n_init=n_init).fit(a_vote_matrix)
        labels = km.labels_
        silhouette_coeff = metrics.silhouette_score(a_vote_matrix, labels, metric='sqeuclidean')

        return labels, silhouette_coeff

    except Exception as e:
        print('No clustering solution matching requirements found. Please consider using alternative algorithms, or ranges of values', file=sys.stderr)
        sys.exit()


#----------------------------------------------------------------------#
def remap_labels(mut2index: Dict[str, int], labels: np.ndarray) -> np.ndarray:
    """
    In cases where the mutation IDs are numeric, rename cluster labels
    such that they are ascending.
    """
    mutIDs = list(mut2index.keys())
    N = len(mutIDs)

    # If not all mutation IDs are numeric, skip remapping.
    if sum(1 for x in mutIDs if x.isdigit()) != N:
        return labels

    mut2clust = {mut: labels[mut2index[mut]] for mut in mut2index}
    # Sort based on numeric conversion of mutation IDs.
    mut2clust_sorted = sorted(mut2clust.items(), key=lambda x: int(x[0]))

    clustmap: Dict[int, int] = {}
    index = 0
    for _, cl in zip(*zip(*mut2clust_sorted)):
        if cl not in clustmap:
            clustmap[cl] = index
            index += 1

    new_labels = np.array([clustmap[l] for l in labels])
    return new_labels

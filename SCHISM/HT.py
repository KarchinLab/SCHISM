import os
import shutil
import sys
import numpy as np
from scipy.stats import norm, chi2
from scipy.special import comb as choose  # Consider using: from scipy.special import comb as choose
from typing import Tuple
from SCHISM.CE import read_mutation_counts
from SCHISM.CE import generate_cellularity_file
from SCHISM.CE import generate_cellularity_file_mult

from utils import Config

# A helper function to flatten a list of lists
melt = lambda x: [subitem for item in x for subitem in item]

#----------------------------------------------------------------------#
def ifloat(x: str) -> float:
    """
    Intelligent string-to-float conversion.
    Replaces missing/invalid values for cellularity by -1.0.
    """
    try:
        y = float(x)
        if 0.0 <= y <= 1.0:
            return y
        else:
            return -1.0
    except ValueError:
        return -1.0

#----------------------------------------------------------------------#
def prep_hypothesis_test(args: any) -> None:
    config = Config(args.config_file)
    # prepare input of hypothesis test framework
    
    # if schism is used to estimate cellularity, generate mutation.cellularity
    # regardless of the cellularity estimator, generate cluster.cellularity if cluster definitions are known

    if config.cellularity_estimation == 'schism':
        mutationReadFile = os.path.join(config.working_dir, config.mutation_raw_input)
        mutData = read_mutation_counts(mutationReadFile)
        
        # tumor sample purity: scale to [0,1] interval if percentages are provided
        purity = config.tumor_sample_purity
        for sample in purity.keys():
            if purity[sample] > 1.0:
                purity[sample] = purity[sample] / 100.0 

        if sorted(list(purity.keys())) != sorted(list(mutData.keys())):
            print('Mismatch between samples for which purity is available and those for which read count is available.',
                  file=sys.stderr)
            print('Cellularity estimation failed.', file=sys.stderr)
            sys.exit()

        mutCellularityPath = os.path.join(config.working_dir, config.output_prefix + '.mutation.cellularity')
        clusterCellularityPath = os.path.join(config.working_dir, config.output_prefix + '.cluster.cellularity')
        # no multiplicity provided
        first_entry = list(mutData.values())[0][0]
        if len(first_entry) == 4:
            generate_cellularity_file(mutData, purity, mutCellularityPath,
                                      config.cellularity_estimator['coverage_threshold'],
                                      config.cellularity_estimator['absent_mode'])
        # multiplicity provided
        elif len(first_entry) == 5:
            generate_cellularity_file_mult(mutData, purity, mutCellularityPath,
                                           config.cellularity_estimator['coverage_threshold'],
                                           config.cellularity_estimator['absent_mode'])

        if (not hasattr(config, 'cluster_analysis')) or config.cluster_analysis != 'schism':
            # assumes cluster definitions are provided by the user through the config file
            average_cellularity(config, clusterCellularityPath)
    else:
        clusterCellularityPath = os.path.join(config.working_dir, config.output_prefix + '.cluster.cellularity')
        if (not hasattr(config, 'cluster_analysis')) or config.cluster_analysis != 'schism':
            # assumes cluster definitions are provided by the user through the config file
            average_cellularity(config, clusterCellularityPath)

#----------------------------------------------------------------------#
def average_cellularity(config: any, clusterCellularityPath: str) -> None:
    # read in mutation cellularities
    if config.cellularity_estimation == 'schism':
        mutCellularityPath = os.path.join(config.working_dir, config.output_prefix + '.mutation.cellularity')
    else:
        mutCellularityPath = os.path.join(config.working_dir, config.mutation_cellularity_input)

    mut2index, samples = read_input_samples(mutCellularityPath)
    
    # read in mutation-to-cluster assignment
    clusterPath = os.path.join(config.working_dir, config.mutation_to_cluster_assignment)
    cluster2mut = read_cluster_assignments(clusterPath)

    # initiate the output file
    clusterCellularityPath = os.path.join(config.working_dir, config.output_prefix + '.cluster.cellularity')
    
    with open(clusterCellularityPath, 'w') as f_w:
        print('\t'.join(['sampleID', 'clusterID', 'cellularity', 'sd']), file=f_w)
    
        for sample in samples:
            for cluster in sorted(cluster2mut.keys()):
                # In each sample, find the mean and standard error of cluster cellularity values
                cellularities = [sample.mutCellularity[mut2index[mutID]] for mutID in cluster2mut[cluster]]
                estimatedError = [sample.mutSigma[mut2index[mutID]] for mutID in cluster2mut[cluster]]
                # Drop missing values
                cellularities = list(filter(lambda x: x != -1.0, cellularities))
                estimatedError = list(filter(lambda x: x != -1.0, estimatedError))
    
                if len(cellularities) == 0:
                    clusterMean = 'NA'
                else:
                    clusterMean = np.mean(cellularities)
                    
                if len(cellularities) == 0:
                    clusterError = 'NA'
                elif len(set(cellularities)) == 1:
                    clusterError = np.mean(estimatedError)
                else:
                    clusterError = np.sqrt(np.var(cellularities))
                        
                print('\t'.join([sample.name, cluster,
                                 f'{clusterMean:0.4f}', f'{clusterError:0.4f}']), file=f_w)

#----------------------------------------------------------------------#
def read_input_samples(path: str) -> Tuple[dict, list]:
    """
    Convert the sample mutation cellularity information from the input file
    to a list of sample objects.
    """
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    lines = list(filter(lambda x: (not x.startswith('sample')) and (x != '') and (not x.startswith('#')), lines))
    content = list(map(lambda x: x.split('\t'), lines))
    sampleIDs = sorted(list(set(list(zip(*content))[0])))
    mutIDs = sorted(list(set(list(zip(*content))[1])))
    
    # mutations will appear in this order in vectors/matrices from this point on
    mut2index = dict(zip(mutIDs, range(len(mutIDs))))
    
    samples = []
    for sampleID in sampleIDs:
        # Filter rows corresponding to this sampleID
        sample_table = list(filter(lambda x: x[0] == sampleID, content))
        samples.append(Sample(sample_table, mut2index))
    return mut2index, samples

#----------------------------------------------------------------------#
def read_cluster_assignments(path: str) -> dict:
    cluster2mut = {}
    with open(path, 'r') as f_h:
        for line in f_h:
            if line.startswith('mutationID') or line.startswith('#'):
                continue
            if line == '\n':
                break
            toks = line.strip().split('\t')
            if toks[1] not in cluster2mut:
                cluster2mut[toks[1]] = []
            cluster2mut[toks[1]].append(toks[0])
    return cluster2mut

#----------------------------------------------------------------------#
def hypothesis_test(args: any) -> None:
    config = Config(args.config_file)
    
    # Path to HT input
    if config.hypothesis_test['test_level'] == 'clusters':
        testInputPath = os.path.join(config.working_dir, config.output_prefix + '.cluster.cellularity')
    elif config.hypothesis_test['test_level'] == 'mutations':
        if config.cellularity_estimation == 'schism':
            testInputPath = os.path.join(config.working_dir, config.output_prefix + '.mutation.cellularity')
        else:
            testInputPath = os.path.join(config.working_dir, config.mutation_cellularity_input)
    
    mut2index, samples = read_input_samples(testInputPath)
    
    hypothesisTest = HT(samples)
    hypothesisTest.combine_exact()
    hypothesisTest.decide(config.hypothesis_test['significance_level'])
    
    if config.hypothesis_test['store_pvalues'] == True:
        pvaluePath = os.path.join(config.working_dir, config.output_prefix + '.HT.pvalues')
        hypothesisTest.store_pvalues(pvaluePath)
    
    if config.hypothesis_test['test_level'] == 'mutations':
        decisionPath = os.path.join(config.working_dir, config.output_prefix + '.HT.pov')
        header = 'parent\tchild\trejection'
        hypothesisTest.store_decisions(header, decisionPath)
        if (not hasattr(config, 'cluster_analysis')) or config.cluster_analysis != 'schism':
            aggregate_votes(config)
    else:
        # HT on clusters
        decisionPath = os.path.join(config.working_dir, config.output_prefix + '.HT.cpov')
        header = 'parent.cluster\tchild.cluster\ttopologyCost'
        hypothesisTest.store_decisions(header, decisionPath)

#----------------------------------------------------------------------#
def aggregate_votes(config: any) -> None:
    povPath = os.path.join(config.working_dir, config.output_prefix + '.HT.pov')
    cpovPath = os.path.join(config.working_dir, config.output_prefix + '.HT.cpov')
    
    povList = read_schism_decisions(povPath)
    
    clusterPath = os.path.join(config.working_dir, config.mutation_to_cluster_assignment)
    cluster2mut = read_cluster_assignments(clusterPath)
    
    votes = take_votes(povList, cluster2mut)
    
    store_votes(votes, cpovPath)
    return

#----------------------------------------------------------------------#
def read_schism_decisions(path: str) -> list:
    content = list(map(lambda x: x.split('\t'),
                       open(path, 'r').read().strip().split('\n')))
    content = list(filter(lambda x: (not x[0].startswith('parent')) and (not x[0].startswith('#')), content))
    return content

#----------------------------------------------------------------------#
def take_votes(povList: list, cluster2mut: dict) -> dict:
    clusterIds = sorted(cluster2mut.keys())
    
    # Each cluster votes regarding its ancestral potential toward mutations in other clusters.
    votes = {}
    for cluster1 in clusterIds:
        r1 = list(filter(lambda x: x[0] in cluster2mut[cluster1], povList))
        for cluster2 in clusterIds:
            if cluster1 == cluster2:
                votes[(cluster1, cluster2)] = (0.0, len(cluster2mut[cluster1])**2)
                continue
            rows = list(map(lambda x: float(x[2]),
                            filter(lambda x: x[1] in cluster2mut[cluster2], r1)))
            votes[(cluster1, cluster2)] = (f'{np.mean(rows):0.4f}', len(rows))
    return votes

#----------------------------------------------------------------------#
def store_votes(votes: dict, path: str) -> None:
    with open(path, 'w') as f_w:
        print('\t'.join(['parent.cluster', 'child.cluster', 'topologyCost', 'blockSize']), file=f_w)
    
        votes_list = list(map(lambda x: x[0] + x[1], votes.items()))
        votes_list = sorted(votes_list, key=lambda x: (x[0], x[1]))
        print('\n'.join(list(map(lambda x: '\t'.join(map(str, x)), votes_list))), file=f_w)

#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
class Sample:
    index = 0
    def __init__(self, table: list, mut2index: dict) -> None:
        # Stores mutation ID, the estimated cellularity values of all mutations
        # in a sample, as well as their estimated error.
        self.index = Sample.index
        Sample.index += 1
        
        # Sample ID
        self.name = table[0][0]
        
        # Number of mutations in this sample
        self.size = len(table)
        
        self.mut2index = mut2index
        self.mutCellularity = ['NA'] * self.size
        self.mutSigma = ['NA'] * self.size
        
        for entry in table:
            mutIndex = self.mut2index[entry[1]]
            self.mutCellularity[mutIndex] = entry[2]
            self.mutSigma[mutIndex] = entry[3]
        
        self.mutCellularity = list(map(ifloat, self.mutCellularity))
        self.mutSigma = list(map(ifloat, self.mutSigma))
        
        # 1-D array tracking non-missing elements
        self.isfilled = [0 if ((self.mutCellularity[i] == -1.0) or (self.mutSigma[i] == -1.0)) else 1
                         for i in range(self.size)]
        
        # Stores the pairwise differences in mutCellularity and their error.
        self.delta = []
        self.deltaSigma = []
        # 2D mask indicating non-missing elements.
        self.isFilled = []
        self.fill_delta()
        
    #---------------------------------------------------#        
    def fill_delta(self) -> None:
        # Populates a matrix of pairwise differences in cellularity between all ordered pairs of mutations.
        for i in range(self.size):
            self.delta.extend([self.mutCellularity[i] - np.array(self.mutCellularity)])
            self.deltaSigma.extend([np.sqrt(self.mutSigma[i]**2 + np.array(self.mutSigma)**2)])
        
        self.delta = np.array(self.delta)
        self.deltaSigma = np.array(self.deltaSigma)
    
        self.isFilled = 1 + np.zeros((self.size, self.size))
        for i in range(self.size):
            if self.isfilled[i] == 0:
                self.isFilled[i, :] = 0
                self.isFilled[:, i] = 0

#----------------------------------------------------------------------#
class HT:
    """
    HT = "hypothesis test"
    Assumes that the data matrix reports the delta cellularity of mutation pairs.
    Performs the hypothesis test H0(i->j) for each (i,j).
    """
    def __init__(self, samples: list) -> None:
        self.data = []
        self.sigma = []
        self.isfilled = []
        self.size = 0
        for sample in samples:
            self.data.append(sample.delta)
            self.sigma.append(sample.deltaSigma)
            self.isfilled.append(sample.isFilled)
            self.size += 1
            
        self.mut2index = samples[0].mut2index
        self.dim = self.data[0].shape  # Assumes square matrix
        
    #---------------------------------------------------#
    def combine_exact(self) -> None:
        """
        Combine information across samples and compute a likelihood ratio statistic.
        """
        statistic = np.zeros(self.dim)
    
        for idx in range(self.size):
            # For each sample, use only the negative delta values.
            mask = (self.data[idx] < 0).astype(int)
            statistic = statistic + ((self.data[idx] / self.sigma[idx])**2) * mask * self.isfilled[idx]
    
        # Sum the isfilled attribute to get the total number of samples available for each pair.
        totalSamples = sum(self.isfilled)
    
        # Get the unique counts of available samples.
        countRange = list(map(int, np.unique(totalSamples)))
    
        # Initialize pvalue array.
        self.pvalue = np.zeros(self.dim)
    
        for value in countRange:
            mask = (totalSamples == value).astype(int)
            if value == 0:
                # If no samples available, assign p-value 1.
                self.pvalue += mask
                continue
            n = value
            pvalue = np.zeros(self.dim)
            for k in range(1, 1+n):
                pvalue += (1 - chi2.cdf(statistic, k)) * choose(n, k, exact=True) / (2.0 ** n)
            self.pvalue += pvalue * mask
    
    #---------------------------------------------------#
    def decide(self, alpha: float) -> None:
        """
        Decide the outcome of the test using significance level alpha.
        """
        self.pov = np.zeros(self.dim)
        for i in range(self.dim[0]):
            for j in range(self.dim[0]):
                if self.pvalue[i, j] <= alpha:
                    self.pov[i, j] = 1  # Reject H0: i -> j is rejected
                else:
                    self.pov[i, j] = 0
    
    #---------------------------------------------------#
    def store_pvalues(self, path: str) -> None:
        """
        Store HT p-values.
        """
        with open(path, 'w') as f_w:
            print('parent\tchild\tpvalue', file=f_w)
            index2mut = {v: k for k, v in self.mut2index.items()}
            for i in range(self.dim[0]):
                for j in range(self.dim[0]):
                    print('\t'.join([index2mut[i], index2mut[j], f'{self.pvalue[i,j]:0.4f}']), file=f_w)
    
    #---------------------------------------------------#
    def store_decisions(self, header: str, path: str) -> None:
        """
        Store HT decisions.
        """
        with open(path, 'w') as f_w:
            print(header, file=f_w)
            index2mut = {v: k for k, v in self.mut2index.items()}
            for i in range(self.dim[0]):
                for j in range(self.dim[0]):
                    print('\t'.join([index2mut[i], index2mut[j], str(int(self.pov[i,j]))]), file=f_w)
    
    #---------------------------------------------------#
    def __repr__(self) -> str:
        dim_str = '*'.join(map(str, self.dim))
        return f'cohort data: cohort size {self.size}, data dimensions: {dim_str}'

#----------------------------------------------------------------------#

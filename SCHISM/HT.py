import os
import shutil
import sys
import numpy as np

from scipy.stats import norm, chi2
from scipy.misc import comb as choose

from CE import read_mutation_counts
from CE import generate_cellularity_file

from utils import Config

melt = lambda x: [subitem for item in x\
                          for subitem in item]
#----------------------------------------------------------------------#
def ifloat(x):
    # intelligent string to float conversion 
    # replace missing/invalid values for cellularity by -1.0
    try: 
        y =  float(x)
        if y >= 0.0 and y <= 1.0:
            return y
        else:
            return -1.0
    except ValueError: 
        return -1.0
#----------------------------------------------------------------------#
def prep_hypothesis_test(args):
    config = Config(args.config_file)
    # prepare input of hypothesis test framework
    
    # if schism is used to estimate cellularity, 
    # generate mutation.cellularity

    # regardless of the cellularity estimator, generate
    # cluster.cellularity if cluster definitions are known

    if config.cellularity_estimation == 'schism':
        mutationReadFile = os.path.join(config.working_dir,\
                                    config.mutation_raw_input)
        mutData = read_mutation_counts(mutationReadFile)
        
        # tumor sample purity, scale to [0,1] interval if 
        # the user reported percentages
        purity = config.tumor_sample_purity
        for sample in purity.keys():
            if purity[sample] > 1.0:
                purity[sample] = purity[sample] / 100.0 

        if sorted(purity.keys()) != sorted(mutData.keys()):
            print >>sys.stderr, 'Mismatch between samples for which purity is' + \
                'available and those for which read count is available.'
            print >>sys.stderr, 'Cellularity estimation failed.'
            sys.exit()

        mutCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.mutation.cellularity')
        clusterCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')

        generate_cellularity_file(mutData, purity, mutCellularityPath,\
                                  config.cellularity_estimator['coverage_threshold'],\
                                  config.cellularity_estimator['absent_mode'])
        
        if (not(hasattr(config, 'cluster_analysis')))  or config.cluster_analysis != 'schism':
            # assumes cluster definitions are provided by the user through config file
            average_cellularity(config,clusterCellularityPath)
    else:
        
        clusterCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')
        if (not(hasattr(config, 'cluster_analysis')))   or config.cluster_analysis != 'schism':
            # assumes cluster definitions are provided by the user through config file
            average_cellularity(config,clusterCellularityPath)
#----------------------------------------------------------------------#
def average_cellularity(config, clusterCellularityPath):
    # read in mutation cellularities
    if config.cellularity_estimation == 'schism':
        mutCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.mutation.cellularity')
    else:
        mutCellularityPath = os.path.join(config.working_dir,\
                                    config.mutation_cellularity_input)

    mut2index, samples = read_input_samples(mutCellularityPath)
    
    # read in mutation to cluster assignment
    clusterPath = os.path.join(config.working_dir, \
                               config.mutation_to_cluster_assignment)
    cluster2mut = read_cluster_assignments(clusterPath)

    # initiate the output file
    clusterCellularityPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')
    
    f_w = file(clusterCellularityPath, 'w')
    print >>f_w, '\t'.join(['sampleID', 'clusterID', 'cellularity', 'sd'])
    
    for sample in samples:
        for cluster in sorted(cluster2mut.keys()):
            # in each sample find the mean and standard error
            # of cluster cellularity values
            
            cellularities = [sample.mutCellularity[mut2index[mutID]] \
                              for mutID in cluster2mut[cluster]]
            estimatedError = [sample.mutSigma[mut2index[mutID]] \
                              for mutID in cluster2mut[cluster]]
            # drop missing values
            cellularities = filter(lambda x: x != -1.0, cellularities)
            estimatedError = filter(lambda x: x != -1.0, estimatedError)

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
                        
            print >>f_w, '\t'.join([sample.name, cluster,\
                                           '%0.4f'%clusterMean, '%0.4f'%clusterError])

    f_w.close()
#----------------------------------------------------------------------#
def read_input_samples(path):
    # convert the sample mutation cellularity information from the input file
    # table to a list of sample objects capturing data from each sample
    
    lines = file(path).read().strip().split('\n')
    lines = filter(lambda x: (not x.startswith('sample')) and \
                   (x != '') and (not x.startswith('#')) , lines)

    content = map(lambda x: x.split('\t'), lines)
    sampleIDs = sorted(list(set(zip(*content)[0])))
    mutIDs = sorted(list(set(zip(*content)[1])))

    # mutations will appear in this order in vectors/matrices from this 
    # point on
    mut2index = dict(zip(mutIDs,range(len(mutIDs))))

    samples = []
    for sampleID in sampleIDs:
        samples.append(Sample(filter(lambda x: x[0] == sampleID, content),\
                              mut2index))
    return mut2index, samples
#----------------------------------------------------------------------#
def read_cluster_assignments(path):        
    cluster2mut = {}
    f_h = file(path)

    for line in f_h:
        if line.startswith('mutationID') or line.startswith('#'):
            continue
        if line == '\n':
            break
        toks = line.strip().split('\t')
        if toks[1] not in cluster2mut:
            cluster2mut[toks[1]] = []
        cluster2mut[toks[1]].append(toks[0])
    f_h.close()

    return cluster2mut
#----------------------------------------------------------------------#
def hypothesis_test(args):
    config = Config(args.config_file)

    # path to HT input
    if config.hypothesis_test['test_level'] == 'clusters':
        testInputPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.cluster.cellularity')
    elif config.hypothesis_test['test_level'] == 'mutations':
        if config.cellularity_estimation == 'schism':
            testInputPath = os.path.join(config.working_dir,\
                                   config.output_prefix + '.mutation.cellularity')
        else:
            testInputPath = os.path.join(config.working_dir,\
                                    config.mutation_cellularity_input)

    mut2index, samples = read_input_samples(testInputPath)
    
    hypothesisTest = HT(samples)
    hypothesisTest.combine_exact()
    hypothesisTest.decide(config.hypothesis_test['significance_level'])

    if config.hypothesis_test['store_pvalues'] == True:
        pvaluePath = os.path.join(config.working_dir,\
                                  config.output_prefix + '.HT.pvalues')
        hypothesisTest.store_pvalues(pvaluePath)

    if config.hypothesis_test['test_level'] == 'mutations':
        decisionPath = os.path.join(config.working_dir,\
                                config.output_prefix + '.HT.pov')
        header = 'parent\tchild\trejection'
        hypothesisTest.store_decisions(header, decisionPath)
        if (not(hasattr(config, 'cluster_analysis')))  or config.cluster_analysis != 'schism':
            aggregate_votes(config)            
    else:
        # HT on clusters
        decisionPath = os.path.join(config.working_dir,\
                                config.output_prefix + '.HT.cpov')
        header = 'parent.cluster\tchild.cluster\ttopologyCost'
        hypothesisTest.store_decisions(header, decisionPath)
#----------------------------------------------------------------------#
def aggregate_votes(config):

    povPath = os.path.join(config.working_dir,\
                           config.output_prefix + '.HT.pov')
    cpovPath = os.path.join(config.working_dir,\
                           config.output_prefix + '.HT.cpov')
    
    povList = read_schism_decisions(povPath)

    clusterPath = os.path.join(config.working_dir, \
                               config.mutation_to_cluster_assignment)
    cluster2mut = read_cluster_assignments(clusterPath)

    votes = take_votes(povList, cluster2mut)
    
    store_votes(votes, cpovPath)
    return
#----------------------------------------------------------------------#
def read_schism_decisions(path):
    content = map(lambda x: x.split('\t'),\
                  file(path).read().strip().split('\n'))
    content = filter(lambda x: (not x[0].startswith('parent')) and \
                               (not x[0].startswith('#')) , content)
    return content
#----------------------------------------------------------------------#
def take_votes(povList, cluster2mut):
    clusterIds = sorted(cluster2mut.keys())
    
    # mutations in each cluster vote regarding its ancestral potential to
    # mutations in the other clusters
    # keep track of the votes of each cluster pairs
    votes = {}
    for cluster1 in clusterIds:
        r1 = filter(lambda x: x[0] in cluster2mut[cluster1], povList)
        for cluster2 in clusterIds:
            if cluster1 == cluster2:
                votes[(cluster1, cluster2)] = (0.0,\
                                               len(cluster2mut[cluster1])**2)
                continue
            rows = map(lambda x: float(x[2]),
                       filter(lambda x: x[1] in cluster2mut[cluster2], r1))

            votes[(cluster1, cluster2)] = ('%0.4f'%np.mean(rows),len(rows))

    return votes
#----------------------------------------------------------------------#
def store_votes(votes, path):
    f_w = file(path, 'w')

    print >>f_w,  '\t'.join(['parent.cluster',\
                                  'child.cluster',\
                                  'topologyCost',\
                                  'blockSize'])

    votes = map(lambda x: x[0] + x[1], votes.items())
    votes = sorted(votes, key = lambda x: (x[0],x[1]))
    
    print >>f_w, '\n'.join(map(lambda x: '\t'.join(map(str,x)),\
                               votes))
    f_w.close()
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
class Sample(object):
    index = 0
    def __init__(self, table, mut2index):
        # stores mutation ID, the estimated cellularity values of all mutations 
        # in a sample, as well as their estimated error 
        
        self.index = Sample.index
        Sample.index += 1
        
        # sample ID
        self.name = table[0][0]
        
        # number of mutations in this sample
        self.size = len(table)

        self.mut2index = mut2index
        self.mutCellularity = ['NA'] * self.size
        self.mutSigma = ['NA'] * self.size
        
        for entry in table:
            mutIndex = self.mut2index[entry[1]]
            self.mutCellularity[mutIndex] = entry[2]
            self.mutSigma[mutIndex] = entry[3]

        self.mutCellularity = map(ifloat, self.mutCellularity)
        self.mutSigma = map(ifloat,self.mutSigma)
        
        # 1-D array tracking non-missing elements
        self.isfilled = [0 if ((self.mutCellularity[index] == -1.0) or \
                          (self.mutSigma[index] == -1.0)) else 1 \
                         for index in range(self.size)]

        # stores the pairwise differences in mutCellularity
        self.delta = []
        self.deltaSigma = []
        # 2D mask indicating non-missing elements
        self.isFilled = []
        self.fill_delta()
    #---------------------------------------------------#        
    def fill_delta(self):
        
        # populates a matrix of pairwise differences in cellularity between 
        # all possible ordered pairs of mutations        
        for id in range(self.size):
            # compute the pair cellularity delta
            self.delta.extend([self.mutCellularity[id] - \
                               np.array(self.mutCellularity)])
            # the estimated standard deviation of delta between pair cellularities
            self.deltaSigma.extend([np.sqrt(self.mutSigma[id]**2 +\
                                            np.array(self.mutSigma)**2)])

        self.delta = np.array(self.delta)
        self.deltaSigma = np.array(self.deltaSigma)

        self.isFilled = 1 + np.zeros((self.size, self.size))
        for index in range(self.size):
            if self.isfilled[index] == 0:
                self.isFilled[index,:] = 0
                self.isFilled[:,index] = 0

#----------------------------------------------------------------------#
class HT(object):
    
    # HT="hypothesis test"
    # assumes that the data matrix reports the delta cellularity of mutation pairs
    # Performs the hypothesis test of H_{0}(i->j) for each (i,j)
    
    def __init__(self, samples):        
        self.data = []
        self.sigma = []
        self.isfilled = []
        self.size = 0
        for sample in samples:
            self.data.extend([sample.delta])
            self.sigma.extend([sample.deltaSigma])
            self.isfilled.extend([sample.isFilled])
            self.size += 1
            
        self.mut2index = samples[0].mut2index
        self.dim = len(self.data[0]), len(self.data[0])
    #---------------------------------------------------#
    def combine_exact(self):
        
        # Hypothesis Test:
        # H0(i->j): i can be an ancestor of j
        # HA(i-*>j): i cannot be an ancestor of j

        statistic = np.zeros(self.dim)

        for index in range(self.size):
            
            # H0 specifies that all sampled diff values have + means -->
            # all the observed positive values are not any different than H+ null
            # hypothesis --> they contribute a multiplicative value of 1 to the LR 
            # statistic
            # the observed negative values are where the ratio difference lies, and those
            # will be forced to be generated by mean of 0 under H+ 
 
            mask  =  (self.data[index] < 0).astype(int)
        
            # sum up the (-2ln(x) ) for x corresponding each of 
            # the terms in likelihood ratio
            # put zero where elements where the delta is missing and
            # is filled with place holder
            statistic = statistic + \
                     ((self.data[index]/ self.sigma[index])**2)* \
                     mask * self.isfilled[index]
            
        # sum over isfilled attribute to get the total number
        # of samples available for comparison of a pair of mutations
        totalSamples = sum(self.isfilled)

        # get the list of existing number of available samples across
        # all pairs of mutations
        countRange = map(int, list(np.unique(totalSamples)))
        
        # start by a blank input for pvalue
        self.pvalue = np.zeros(self.dim)

        for value in countRange:
            # for each value in existing set of available samples for a pair
            # assign the p-value for such pairs using exact test formulation
            # for that total sample count
            mask = (totalSamples == value).astype(int)
            if value == 0:
                # if no samples available for a pair of mutations
                # use pvalue of 1 to indicate lack of rejection
                # (insufficient information)
                self.pvalue += mask
                continue
            n = value
            pvalue = np.zeros(self.dim)
            for k in range(1, 1+n):
                pvalue += (1-chi2.cdf(statistic,k)) * \
                          choose(n, k, exact = True) / (2.0 ** n)
            self.pvalue += pvalue * mask

    #---------------------------------------------------#
    def decide(self, alpha):
        
        # decide th outcome of the test using
        # the significance level alpha.

        self.pov = np.zeros(self.dim)
        
        pairs = [(i,j) for i in range(self.dim[0]) for j in range(self.dim[0])]
        for pair in pairs:
            ij = self.pvalue[pair]
            if ij <= alpha:
                # 1 signals a rejection event for null of i -> j
                self.pov[pair] = 1
            else:
                self.pov[pair] = 0
    #---------------------------------------------------#
    def store_pvalues(self, path):
        # store HT p-values
        f_w = file(path, 'w')
        print >>f_w, 'parent\tchild\tpvalue'

        index2mut = dict(zip(self.mut2index.values(),\
                         self.mut2index.keys()))

        for i in range(self.dim[0]):
            for j in range(self.dim[0]):

                print >>f_w, '\t'.join([index2mut[i], index2mut[j],\
                                        '%0.4f'%self.pvalue[i,j]])
        f_w.close()
    #---------------------------------------------------#
    def store_decisions(self, header, path):
        # store HT decisions
        f_w = file(path, 'w')
        print >>f_w, header

        index2mut = dict(zip(self.mut2index.values(),\
                         self.mut2index.keys()))

        for i in range(self.dim[0]):
            for j in range(self.dim[0]):

                print >>f_w, '\t'.join([index2mut[i], index2mut[j],\
                                        str(int(self.pov[i,j]))])
        f_w.close()
    #---------------------------------------------------#
    def __repr__(self):
        dim = '*'.join(map(str, self.dim))
        return 'cohort data: cohort size %d, data dimensions: %s'%(self.size, dim)
    #---------------------------------------------------#

#----------------------------------------------------------------------#




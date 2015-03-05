import os
import sys

import numpy as np

from utils import Config

#----------------------------------------------------------------------#
def read_mutation_counts(path):
    f_h = file(path, 'r')
    
    expectedColumns = ['sampleID', 'mutationID', 'referenceReads', \
                       'variantReads', 'copyNumber']

    # check the header line is present as expected
    if f_h.readline().strip().split('\t') != expectedColumns:
        print >>sys.stderr, 'Invalid column Headers in mutation read count input.' +\
                            'Cellularity estimation failed'
        print >>sys.stderr, 'Expected columns are:\n %s'%\
            '\t'.join(expectedColumns)
        sys.exit()
    
    content = map(lambda x: x.split('\t'),\
                  filter(lambda y: (y!= '') and (not y.startswith('#')) ,\
                                                  f_h.read().split('\n')))

    # confirm that the correct number of entries present in all lines
    if list(set(map(len,content))) != [5]:
        print >>sys.stderr, 'Invalid number of columns present in '+\
            ' mutation read count input. Cellularity estimation failed'
        sys.exit()


    content = map(lambda x: [x[0],x[1]] + map(floatField, x[2:]), content)
    samples = sorted(list(set(zip(*content)[0])))
    
    sampleMuts = {}
    for sample in samples:
        data = map(lambda y: y[1:], filter(lambda x: x[0] == sample, content))
        sampleMuts[sample] = data

    mutCount = len(sampleMuts[samples[0]])
    for sample in samples:
        if len(sampleMuts[sample])!= mutCount:
            print >>sys.stderr, 'Mismatch in mutation count between:' +\
                samples[0] + ' and ' + sample
            print >>sys.stderr, 'Cellularity estimation failed'
            sys.exit()
    
    return sampleMuts
#----------------------------------------------------------------------#
def generate_cellularity_file(mutData, purity, cellularityPath, coverageThreshold,\
                              absentMode):
    f_w = file(cellularityPath, 'w')
    
    print >>f_w, '\t'.join(['sampleID','mutationID','cellularity','sd'])

    for sample in mutData.keys():
        p = purity[sample]

        for element in mutData[sample]:
            # mutations outside diploid or hemizygous regions
            if element[3] not in [1,2]:
                print >>f_w, '\t'.join([sample, element[0], 'NA', 'NA'])
                continue
            # mutation with insufficient coverage
            if (element[1] + element[2]) < coverageThreshold:
                print >>f_w, '\t'.join([sample, element[0], 'NA', 'NA'])
                continue
            
            # handling zero read counts for alternate allele
            if element[2] == 0.0:
                if absentMode == 1:
                    # assign cellularity of 0.0 to mutation
                    # with standard error of 0.05
                    print >>f_w, '\t'.join([sample, element[0], '%0.4f'%0.0,\
                                            '%0.4f'%0.05])
                    continue
                else:
                    # assign a pseudo count of 1 to reference and alternate reads 
                    element[1] += 1.0
                    element[2] += 1.0

            # handling zero read counts for reference allele
            if element[1] == 0.0:
                element[1] += 1.0
                element[2] += 1.0
            
            # diploid region
            if element[3] == 2:
                vaf = element[2]/(element[1] + element[2])
                cellularity = min(1, 2 * vaf / p)
                noiseSd = np.sqrt(((2/p)**2)*vaf*(1-vaf)/(element[1] + element[2]))
                print >>f_w, '\t'.join([sample, element[0], '%0.4f'%cellularity,\
                                        '%0.4f'%noiseSd])
            # hemizygous region
            # this formulation assumes that the region is diploid in normal
            # genome, and has copy number 1 in tumor genome
            elif element[3] == 1:
                vaf = element[2]/(element[1] + element[2])
                cellularity = min(1, (2-p) * vaf / p)
                noiseSd = np.sqrt((((2-p)/p)**2) *vaf*(1-vaf)/(element[1] + element[2]))
                print >>f_w, '\t'.join([sample, element[0], '%0.4f'%cellularity,\
                                        '%0.4f'%noiseSd])
            else:
                print >>sys.stderr, 'Cellularity estimation failed for.' +\
                                    'sample:%s, mut:%s'%(sample,\
                                                         '\t'.join(map(str, element)))
                sys.exit()

    f_w.close()
#----------------------------------------------------------------------#
def floatField(x):
    # intelligent string to float conversion 
    # replace missing/invalid values of reads/cna with -1.0
    try: 
        y =  float(x)
        if y >= 0.0 :
            return y
        else:
            return 0.0
    except ValueError: 
        return 0.0

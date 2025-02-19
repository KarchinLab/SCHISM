import os
import sys
import numpy as np
from typing import List, Dict, Any

from utils import Config

#----------------------------------------------------------------------#
def floatField(x: str) -> float:
    """Intelligently converts a string to a float.
    
    Replaces missing/invalid values of reads/CNA with 0.0.
    """
    try:
        y = float(x)
        return y if y >= 0.0 else 0.0
    except ValueError:
        return 0.0

#----------------------------------------------------------------------#
def read_mutation_counts(path: str) -> Dict[str, List[List[Any]]]:
    """
    Reads a mutation counts file and returns a dictionary mapping each sampleID 
    to a list of mutation entries. Each entry is a list starting with mutationID 
    followed by numeric values.
    """
    with open(path, 'r') as f_h:
        expected_columns = ['sampleID', 'mutationID', 'referenceReads',
                            'variantReads', 'copyNumber']
        # Check header line
        header = f_h.readline().strip().split('\t')[:5]
        if header != expected_columns:
            print(f'Invalid column Headers in mutation read count input. '
                  f'Cellularity estimation failed', file=sys.stderr)
            expected_str = "\t".join(expected_columns)
            print(f'Expected columns are:\n{expected_str}', file=sys.stderr)
            #print(f'Expected columns are:\n{"\t".join(expected_columns)}', file=sys.stderr)
            sys.exit(1)

        # Read the remaining content
        content_lines = f_h.read().split('\n')

    # Filter out empty lines and comments; split each line by tab
    content = [
        line.split('\t')
        for line in content_lines
        if line and not line.startswith('#')
    ]

    # Confirm that each line has 5 or 6 entries.
    lengths = list(set(map(len, content)))
    if lengths not in [[5], [6]]:
        print('Invalid number of columns present in mutation read count input. '
              'Cellularity estimation failed', file=sys.stderr)
        sys.exit(1)

    # Convert numeric fields using floatField.
    # First two columns are strings; the rest should be converted.
    content = [
        row[:2] + list(map(floatField, row[2:]))
        for row in content
    ]

    # Extract unique samples from the first column.
    samples = sorted({row[0] for row in content})

    sample_muts: Dict[str, List[List[Any]]] = {}
    for sample in samples:
        data = [row for row in content if row[0] == sample]
        # Remove the sample name from each mutation entry
        sample_muts[sample] = [row[1:] for row in data]

    mut_count = len(sample_muts[samples[0]])
    for sample in samples:
        if len(sample_muts[sample]) != mut_count:
            print(f'Mismatch in mutation count between: {samples[0]} and {sample}',
                  file=sys.stderr)
            print('Cellularity estimation failed', file=sys.stderr)
            sys.exit(1)

    return sample_muts

#----------------------------------------------------------------------#
def generate_cellularity_file(mut_data: Dict[str, List[List[Any]]],
                              purity: Dict[str, float],
                              cellularity_path: str,
                              coverage_threshold: float,
                              absent_mode: int) -> None:
    """
    Generate a file with cellularity estimates.
    """
    with open(cellularity_path, 'w') as f_w:
        print('\t'.join(['sampleID', 'mutationID', 'cellularity', 'sd']), file=f_w)

        for sample in mut_data.keys():
            p = purity[sample]

            for element in mut_data[sample]:
                # element: [mutationID, referenceReads, variantReads, copyNumber, ...]
                # mutations outside diploid or hemizygous regions
                if element[3] not in [1, 2]:
                    print('\t'.join([sample, element[0], 'NA', 'NA']), file=f_w)
                    continue

                # mutation with insufficient coverage
                if (element[1] + element[2]) < coverage_threshold:
                    print('\t'.join([sample, element[0], 'NA', 'NA']), file=f_w)
                    continue

                # handling zero read counts for alternate allele
                if element[2] == 0.0:
                    if absent_mode == 1:
                        # assign cellularity of 0.0 with a standard error of 0.05
                        print('\t'.join([sample, element[0], f'{0.0:0.4f}', f'{0.05:0.4f}']), file=f_w)
                        continue
                    if absent_mode == 2:
                        cellularity = 0.0
                        # Only add pseudo counts in estimating SD, not VAF.
                        # Handle gain SD estimation.
                        vaf = (element[2] + 1) / (element[1] + element[2] + 2)
                        if element[3] >= 2:
                            noise_sd = np.sqrt(((2 / p) ** 2) * vaf * (1 - vaf) /
                                               (element[1] + element[2] + 2))
                        else:
                            noise_sd = np.sqrt((((2 - p) / p) ** 2) * vaf * (1 - vaf) /
                                               (element[1] + element[2] + 2))
                        
                        print('\t'.join([sample, element[0], f'{cellularity:0.4f}', f'{noise_sd:0.4f}']), file=f_w)
                        continue
                    if absent_mode == 0:
                        # assign a pseudo count of 1 to both reference and alternate reads 
                        element[1] += 1.0
                        element[2] += 1.0

                # handling zero read counts for reference allele
                if element[1] == 0.0:
                    element[1] += 1.0
                    element[2] += 1.0

                # diploid region
                if element[3] == 2:
                    vaf = element[2] / (element[1] + element[2])
                    cellularity = min(1, 2 * vaf / p)
                    noise_sd = np.sqrt(((2 / p) ** 2) * vaf * (1 - vaf) / (element[1] + element[2]))
                    print('\t'.join([sample, element[0], f'{cellularity:0.4f}', f'{noise_sd:0.4f}']), file=f_w)
                # hemizygous region
                elif element[3] == 1:
                    vaf = element[2] / (element[1] + element[2])
                    cellularity = min(1, (2 - p) * vaf / p)
                    noise_sd = np.sqrt((((2 - p) / p) ** 2) * vaf * (1 - vaf) / (element[1] + element[2]))
                    print('\t'.join([sample, element[0], f'{cellularity:0.4f}', f'{noise_sd:0.4f}']), file=f_w)
                else:
                    mut_str = "\t".join(map(str, element))
                    print(f'Cellularity estimation failed for sample:{sample}, mut:{mut_str}', file=sys.stderr)

                    #print(f'Cellularity estimation failed for sample:{sample}, mut:{"\t".join(map(str, element))}',
                    #      file=sys.stderr)
                    sys.exit(1)

#----------------------------------------------------------------------#
def generate_cellularity_file_mult(mut_data: Dict[str, List[List[Any]]],
                                   purity: Dict[str, float],
                                   cellularity_path: str,
                                   coverage_threshold: float,
                                   absent_mode: int) -> None:
    """
    Generate a file with cellularity estimates using a modified calculation that takes
    into account a user-provided multiplicity level.
    """
    with open(cellularity_path, 'w') as f_w:
        print('\t'.join(['sampleID', 'mutationID', 'cellularity', 'sd']), file=f_w)

        for sample in mut_data.keys():
            p = purity[sample]

            for element in mut_data[sample]:
                # mutation with insufficient coverage
                if (element[1] + element[2]) < coverage_threshold:
                    print('\t'.join([sample, element[0], 'NA', 'NA']), file=f_w)
                    continue

                # handling zero read counts for alternate allele
                if element[2] == 0.0:
                    if absent_mode == 1:
                        print('\t'.join([sample, element[0], f'{0.0:0.4f}', f'{0.05:0.4f}']), file=f_w)
                        continue
                    if absent_mode == 2:
                        cellularity = 0.0
                        vaf = (element[2] + 1) / (element[1] + element[2] + 2)
                        if element[3] >= 2:
                            noise_sd = np.sqrt(((2 / p) ** 2) * vaf * (1 - vaf) /
                                               (element[1] + element[2] + 2))
                        else:
                            noise_sd = np.sqrt((((2 - p) / p) ** 2) * vaf * (1 - vaf) /
                                               (element[1] + element[2] + 2))
                        
                        print('\t'.join([sample, element[0], f'{cellularity:0.4f}', f'{noise_sd:0.4f}']), file=f_w)
                        continue
                    if absent_mode == 0:
                        element[1] += 1.0
                        element[2] += 1.0

                # handling zero read counts for reference allele
                if element[1] == 0.0:
                    element[1] += 1.0
                    element[2] += 1.0

                # modified cellularity calculation for a user-provided multiplicity level
                multiplicity = element[4]
                total_cn = p * element[3] + (1 - p) * 2
                coverage = element[1] + element[2]
                vaf = element[2] / coverage
                cellularity = min(1, (total_cn * vaf) / (p * multiplicity))
                noise_sd = total_cn / (multiplicity * p) * np.sqrt(vaf * (1 - vaf) / coverage)
                print('\t'.join([sample, element[0], f'{cellularity:0.4f}', f'{noise_sd:0.4f}']), file=f_w)

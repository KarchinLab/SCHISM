===========
Schism
===========

SCHISM is a computational tool designed to infer subclonal hierarchy and
the tumor evolution from somatic mutations. The inference process involves
computational assessment of two fundamental properties of tumor evolution:
lineage precedence rule and lineage divergence rule.

First, SCHISM combines information about somatic mutation cellularity (aka
mutation cancer cell fraction) across all tumor sample(s) available from a
patient in a hypothesis testing framework to identify the statistical support
for the lineage relationship between each pair of mutations or mutation clusters.
The results of the hypothesis test are represented as Cluster Order Precedence 
Violation (CPOV) matrix which informs the subsequent step in SCHISM and ensures 
compliance of candidate tree topologies with lineage precedence rule.

Next, an implementation of genetic algorithm (GA) based on a fitness function that
 incorporates considerations for both lineage precedence (CPOV) rule and lineage 
divergence rule explores the space of tree topologies and returns a prioritized list
 of candidate subclonal phylogenetic trees, most compatible with observed cellularity
 data. 


Links
===========

Please consult the SCHISM wiki page for installation guide, software tutorial, and 
usage examples.

https://github.com/Niknafs/SCHISM/wiki

Releases
===========
    SCHISM-1.0.0    2015-03-05    Initial release, concurrent with manuscript publication
    SCHISM-1.0.1    2016-01-01    Minor visualization update. Citation updated. 

Citation
===========

If you use SCHISM in your research, please cite:

    Niknafs et al. SubClonal Hierarchy Inference from Somatic Mutations: automatic 
    reconstruction of cancer evolutionary trees from multi-region next generation 
    sequencing http://dx.plos.org/10.1371/journal.pcbi.1004416


Availability
===========

Stable releases of SCHISM are available from KarchinLab website. 
http://karchinlab.org/apps/appSchism
and also
https://github.com/Niknafs/SCHISM/releases

Platform
===========

Basic functionality of SCHISM scripts developed in python should be available across
 all platforms. 

The plotting functionality of SCHISM relies on matplotlib, igraph c core and 
python-igraph. Matplotlib should run across all platforms. igraph dependencies (c core, 
and python-igraph), however, have been tested on linux distributions Fedora and Ubuntu 
only. An installation guide for igraph dependencies is available on the project wiki
page. 


Support
===========

Please contact the package developer Noushin Niknafs (niknafs at jhu dot edu) 
with suggestions, questions or bug reports.


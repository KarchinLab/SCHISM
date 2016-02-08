import sys
import yaml

class Config():
    # generate run configuration object from user-provided config file in yaml format
    def __init__(self, path):
        f_h = file(path,'r')
        config = yaml.load(f_h)
        f_h.close()
        self.__dict__.update(config)
        # check parameter set for validity
        if hasattr(self, 'cluster_analysis') and self.cluster_analysis == 'schism':
            if self.hypothesis_test['test_level'] == 'clusters':
                print >>sys.stderr, 'Cluster analysis by SCHISM requires setting the "test_level" parameter to "mutations".'
                print >>sys.stderr, 'Please update the configuration file before proceeding.'
                sys.exit()
        return

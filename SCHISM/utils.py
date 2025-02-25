import sys
import yaml

class Config:
    # Generate run configuration object from a user-provided YAML config file.
    def __init__(self, path: str) -> None:
        with open(path, 'r') as f_h:
            config = yaml.safe_load(f_h)
        self.__dict__.update(config)
        # Check parameter set for validity.
        if hasattr(self, 'cluster_analysis') and self.cluster_analysis == 'schism':
            if self.hypothesis_test['test_level'] == 'clusters':
                print('Cluster analysis by SCHISM requires setting the "test_level" parameter to "mutations".', file=sys.stderr)
                print('Please update the configuration file before proceeding.', file=sys.stderr)
                sys.exit()

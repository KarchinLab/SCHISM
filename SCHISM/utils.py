import yaml

class Config():
    # generate run configuration object from user-provided config file in yaml format
    def __init__(self, path):
        f_h = file(path,'r')
        config = yaml.load(f_h)
        f_h.close()
        self.__dict__.update(config)
        return

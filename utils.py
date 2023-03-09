import yaml, os


def read_yaml(fpath):
    with open(fpath, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def extract_suffix(fpath):
    basepath = os.path.basename(fpath)
    assert basepath.endswith('.yaml')
    
    suffix = basepath.rstrip('.yaml')
    return suffix
import yaml
import os
from pathlib import Path
from utils.utils import check_suffix, check_file

def load_yaml(file='yolov8n.yaml'):
    """@brief load yaml file
       @param file yaml file path
       @return yaml file content as dictionary if file exists else None
       """
    # check file is empty or not
    if check_yaml(file, suffix='.yaml'):
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    else:
        print('file must exists')
        return None

def check_yaml(file, suffix='.yaml'):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)

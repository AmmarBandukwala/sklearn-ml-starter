import os
import shutil
import yaml
import pickle

from urllib import request

ROOT_PATH = './outputs'

def save_pickle(data, file_path: str):
    """Save a python object as a pickle file.

    Args:
        data (object): Python object for serialization.
        file_path (str): Target file path for output.
    """
    full_path = os.path.join(ROOT_PATH, file_path)
    with open(full_path, 'wb') as stream:
        pickle.dump(data, stream)

def load_pickle(full_path: str):
    """Deserialize a pickle file into a python object.

    Args:
        full_path (str): Source file path of pickle file.

    Returns:
        object: Python object from pickle file.
    """
    with open(full_path, 'rb') as stream:
        ret_obj = pickle.load(stream)
        return ret_obj
    
def open_yaml_file(filePath: str) -> dict:
    """Open YAML file from a local file path.

    Args:
        filePath (str): Absolute File Path

    Returns:
        dict: Python dictionary object of YAML key/value pairs.
    """
    with open(filePath, "r") as stream:
        try:
            dic_values = yaml.safe_load(stream)
            return dic_values
        except yaml.YAMLError as exc:
            print(exc)
            return None

def open_yaml_url(url: str) -> dict:
    """Open YAML file from a https URI locator.

    Args:
        filePath (str): Absolute File Path

    Returns:
        dict: Python dictionary object of YAML key/value pairs.
    """
    with request.urlopen(url) as stream:
        try:
            dic_values = yaml.safe_load(stream)
            return dic_values
        except yaml.YAMLError as exc:
            print(exc)
            return None
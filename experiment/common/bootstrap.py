import argparse
import mlflow

from common.filemanager import open_yaml_file, open_yaml_url

def setup(debug_args: argparse.Namespace = None) -> dict:
    """Bootstrap entry method to configure all parameters to invoke the model training.

    Args:
        debug_args (argparse.Namespace, optional): Override default behavior for debugging. Defaults to None.

    Returns:
        dict: YAML file converted to a python object.
    """

    # mlflow enable
    mlflow.autolog()
    
    # get arguments
    args = parse_args()
    
    # override from code if nessary 
    if debug_args is not None:
        args = debug_args
    
    # determine if local or url path, load yaml file into dictionary
    if args.parameter_file_local:
        parameters = open_yaml_file(args.parameter_file_local)
        return parameters
    if args.parameter_file_url:
        parameters = open_yaml_url(args.parameter_file_url)
        return parameters
    
    return None
    
def parse_args() -> argparse.Namespace:
    """Set the parameters for argparser to enable invocation of the python script with names parameter inputs.

    Returns:
        argparse.Namespace: Return argument namespace object with parameter configuration.
    """
    
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--parameter_file_local", type=str)
    parser.add_argument("--parameter_file_url", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args
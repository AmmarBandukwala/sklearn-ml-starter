import pytest
import os

from experiment.common.filemanager import open_yaml_file, open_yaml_url

## COMMANDS TO RUN
# Show Output of Pass Test - pytest -rP tests/test_filemanager.py
# Show Output of Failed TEst - pytest -rx tests/test_filemanager.py

def test_open_yaml_url():        
    yaml_dic = open_yaml_url("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/assets/environment/conda-yamls/pydata.yml")
    assert yaml_dic
    print(yaml_dic)

def test_open_yaml_file():        
    yaml_dic = open_yaml_file(os.path.join("experiment", "parameters", "workflow_parameters.yml"))
    assert yaml_dic
    print(yaml_dic)
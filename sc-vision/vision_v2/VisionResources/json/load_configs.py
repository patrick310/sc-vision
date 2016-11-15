import json

def load_configs(filepath):
    if filepath is None:
        filepath = 'configs.json'
    with open(filepath, "r") as file:
        configs = json.load(file)
    return configs

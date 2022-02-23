import json


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_config(key):
    return config_dict[key]


config_dict = read_json("config.json")

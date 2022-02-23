import json


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path, dict_obj):
    with open(file_path, "w") as f:
        return json.dump(dict_obj, f)


def get_config(key):
    return config_dict[key]


config_dict = read_json("config.json")

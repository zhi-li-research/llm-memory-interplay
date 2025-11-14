import json


def find_all_substring_indices(string, substring, start=0, end=None):
    """
    Find all indices of a substring in a string
    """
    indices = []
    while True:
        index = string.find(substring, start, end)
        if index == -1:
            break
        indices.append(index)
        start = index + len(substring)
    return indices

def read_json_file(filename, jsonl=False):
    """Reads a JSON file and returns the data."""
    
    if jsonl:
        data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data
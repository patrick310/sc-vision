import json

def _byteify(input):
    if isinstance(input, dict):
        return {_byteify(key): _byteify(value)
            for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [_byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def load_dictionary(filepath):
    with open(filepath, "r") as f:
        dictionary = json.load(f)
    dictionary = _byteify(dictionary)
    return dictionary

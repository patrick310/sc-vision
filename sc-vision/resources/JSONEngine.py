import json
#json loader

class JSONEngine:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dictionary = None

    def _byteify(self, input):
        if isinstance(input, dict):
            return {self._byteify(key): self._byteify(value)
                for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [self._byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    def load(self):
        with open(self.filepath, "r") as f:
            self.dictionary = json.load(f)
        self.dictionary = self._byteify(self.dictionary)
        return self.dictionary

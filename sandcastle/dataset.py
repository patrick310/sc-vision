import os

class ImageRecord:
    """Records the history of predictions on an Image"""

    def __init__(self, pilImage):

        self.original_image = pilImage
        self.ground_truth = None
        self.filename = None


class Dataset:

    def __init__(self, name = None):

        self.name = name
        self.record_set = []
        self.allowed_classes = []

    def number_of_classes(self):

        return len(self.allowed_classes)

    @staticmethod
    def create_folder(label):
        try:
            os.mkdir(label)
        except FileExistsError:
            print(label + " already exists, continuing...")

    def create_folder_structure(self):

        assert self.name is not None
        self.create_folder(self.name)
        self.create_folder(self.name + "/" + "train")
        self.create_folder(self.name + "/" + "validate")

        for label in self.allowed_classes:
            self.create_folder(self.name + "/" + "train" + "/" + label)
            self.create_folder(self.name + "/" + "validate" + "/" + label)



if __name__ == '__main__':
    a = Dataset("test_set")
    a.allowed_classes = ["apple", "cat", "dog", "pear"]
    a.number_of_classes()
    a.create_folder_structure()
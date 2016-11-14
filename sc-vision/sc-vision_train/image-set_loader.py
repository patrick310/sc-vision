try:
    import cPickle as pickle
except:
    import pickle
import cv2, os, glob
import numpy as np
import sys


class Merger(object):
    def __init__(self, loaders):
        self.loaders = loaders

    def merge(self):
        self.images = list()
        for loader in self.loaders:
            for image in loader.images:
                self.images.append(image)
        self.images = np.array(self.images)

        for loader in self.loaders:
            del loader

        loader = CV2ImageLoader()
        loader.images = self.images
        return loader

class CV2ImageLoader(object):
    def __init__(self, path=None, img_size=(None, None), gray=False, count=None):
        if path is None: self.path = os.getcwd()
        else: self.path = path

        if img_size[0] is None and img_size[1] is None: self.img_size = 'default'
        else: self.img_size = img_size

        if not gray: self.gray = False
        else: self.gray = True

        if count is None: self.count = 'default'
        else: self.count = count

    def load_images_from_path(self):
        cwd = os.getcwd()
        os.chdir(self.path)

        img_names = list(glob.glob('*.jpg*'))
        if self.count != 'default':
            img_names = img_names[:self.count]

        images = list()
        for name in img_names:
	    print "loading " + name
	    img = cv2.imread(name)
            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.img_size != 'default':
                img = cv2.resize(img, self.img_size)
            images.append(img)

        self.images = np.array(images)

        os.chdir(cwd)
        return self.images

    def save_images_to_file(self, filepath):
        data_string = pickle.dumps(self.images)
        f = open(filepath, "w")
        f.write(data_string)
        f.flush()
        f.close()

    def load_images_from_file(self, filepath):
        f = open(filepath, "r")
        data_string = f.read()
        f.close()
        self.images = pickle.loads(data_string)
        return self.images

    def display_images(self):
        for img in self.images:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_images(self):
        return self.images

#EOF

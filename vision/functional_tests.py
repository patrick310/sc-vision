import unittest
from .deploy.camera import TestCamera
import imghdr


class ImagePreparationTest(unittest.TestCase):

    def test_can_generate_test_images(self):
        camera = TestCamera()
        camera.capture()

    def can_prepare_images_h5py(self):
        print("Need to finish tests")
        #Prepares and checks image sets

if __name__ == '__main__':
    unittest.main(warnings='ignore')
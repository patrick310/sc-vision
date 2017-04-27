import unittest
from deploy.camera import TestCamera
#from train.train_network import *
import imghdr
import numpy as np


class ImagePreparationTest(unittest.TestCase):

    def test_can_generate_test_images(self):
        camera = TestCamera()
        test_image_list = [camera.capture() for count in range(0,4)]
        self.assertIn('PIL.Image', str(test_image_list))
		
    def test_can_prepare_images_h5py(self):
        #Prepares and checks image sets]
        None
        
    #Greg notices a cool new intranet-based applicaion called VisionNet. He sees visionnet in the browser title and
    #html header.
    
    #Greg is immediately invited to upload images
    
    #The images show up in a table under 'class 1' row.
    
    #Greg clicks add class button and names it class 2
    
    #Finish the tests
        

if __name__ == '__main__':
    unittest.main(warnings='ignore')

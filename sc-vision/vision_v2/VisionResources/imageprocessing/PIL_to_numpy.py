from PIL import Image
import numpy as np

def PIL_to_numpy(image):
    return np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0], 3)
    

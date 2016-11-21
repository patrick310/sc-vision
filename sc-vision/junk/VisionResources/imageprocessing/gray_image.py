import PIL
from PIL import Image
from PIL import ImageOps

def gray_image(image):
    return ImageOps.grayscale(image)

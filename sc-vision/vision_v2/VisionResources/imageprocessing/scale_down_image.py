import PIL
from PIL import Image
from PIL.Image import LANCZOS

def scale_down_image(image, size):
    return image.resize(size, resample = PIL.Image.LANCZOS)

import PIL
from PIL import Image
from PIL.Image import LANCZOS

# LANCZOS has the best scale-down image quality of the available
# filters.
def scale_down_image(image, size):
    return image.resize(size, resample = PIL.Image.LANCZOS)

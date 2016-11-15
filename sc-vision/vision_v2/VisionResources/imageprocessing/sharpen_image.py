import PIL
from PIL import Image
from PIL import ImageFilter

def sharpen_image(image, radius, percent, threshold):
    filter = ImageFilter.UnsharpMask(
                    radius = radius,
                    percent = percent,
                    threshold = threshold)
    return image.filter(filter)

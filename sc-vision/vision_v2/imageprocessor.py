import PIL
from PIL import Image
from PIL.Image import LANCZOS
from PIL import ImageFilter
from PIL import ImageOps
import io

def open_image_from_stream(stream):
    # Make sure to rewind stream to beginning of data.
    stream.seek(0)
    return Image.open(stream)

def open_image_from_file(name):
    return Image.open(name)

def scale_down_image(image, size):
    return image.resize(size, resample = PIL.Image.LANCZOS)

def gray_image(image):
    return ImageOps.grayscale(image)

def sharpen_image(image, radius, percent, threshold):
    filter = ImageFilter.UnsharpMask(
                    radius = radius,
                    percent = percent,
                    threshold = threshold)
    return image.filter(filter)

im = open_image_from_file('childphoto.jpg')
gray_then_scale = scale_down_image(gray_image(im), (256, 256))
scale_then_gray = gray_image(scale_down_image(im, (256, 256)))
gray_then_scale.save("gray_then_scale.jpg")
scale_then_gray.save("scale_then_gray.jpg")

sharped = sharpen_image(gray_then_scale, radius = 3, percent = 150, threshold = 4)
sharped.save("sharped.jpg")

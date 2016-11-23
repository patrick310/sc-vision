import picamera
import numpy as np
from PIL import Image, ImageOps
import time
import io
from keras.models import load_model

print "Loading models"
model = load_model("MY_MODEL.hdf5")
print "loaded"

def capture_image(
        resolution = (3280, 2464),
        exposure_mode = 'auto',
        warmup_delay = 0.2,
        size = (156, 156),
        gray = True,
        format = 'jpeg'
        ):
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode
        time.sleep(warmup_delay)
        cam.capture(stream, format = format)
        cam.close()
    stream.seek(0)
    image = Image.open(stream)
    return image

def grayscale_copy(image):
    return ImageOps.grayscale(image)

def resize_copy(image, size):
    return image.resize(size)

def image_to_numpy(image):
    return np.array(image)

def flatten_images_for_network(images, width, height):
    flattened = list()
    for image in images:
        flattened.append(image_to_numpy(image).reshape(1, width, height))
    flattened = np.array(flattened)
    return flattened

image = capture_image()
image = grayscale_copy(image)
image = resize_copy(image, (156, 156))
images = flatten_images_for_network(image, 156, 156)
print model.predict_proba(images, batch_size = 1, verbose = 0)

"""
def capture(
    resolution = (3280, 2464),
    exposure_mode = 'auto',
    warmup_delay = 0.2,
    size = (156, 156),
    gray = True,
    format = 'jpeg'
    ):
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode
        time.sleep(warmup_delay)
        cam.capture(stream, format = format)
        cam.close()
    stream.seek(0)
    original_image = Image.open(stream)
    if gray:
        formatted_image = ImageOps.grayscale(original_image)
    formatted_image = formatted_image.resize(size)
    array = np.array(formatted_image)
    print array.shape
    return original_image, array

def put_through_network(arr):
    global model
    temp = list()
    arr = arr.reshape(1, 156, 156)
    temp.append(arr)
    arr = np.array(temp)
    print model.predict_proba(arr, batch_size = 1, verbose = 0)[0]

print "taking pic"
original, arr = capture()
test = Image.fromarray(arr)
test.save("test.jpg")
print arr
print arr.shape
print "putting into network"
put_through_network(arr)
print "done"
"""

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

def gray_image(image):
    return ImageOps.grayscale(image)

def resize_image(image, size):
    return image.resize(size)

def format_image_for_network(image, size):
    arr = np.array(image)
    arr = arr.reshape(1, size[0], size[1])
    arr = arr.astype('float32')
    arr /= 255.0
    temp = list()
    temp.append(arr)
    arr = np.array(temp)
    return arr

def get_image_prediction(arr):
    return model.predict_proba(arr, batch_size = 1, verbose = 0)[0]

image = capture_image()
image = gray_image(image)
image = resize_image(image, (156, 156))
arr = format_image_for_network(arr, (156, 156))
print get_image_prediction(arr)

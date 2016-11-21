import picamera
import numpy as np
from PIL import Image, ImageOps
import io
import time
from keras.models import load_model

print "Loading models"
model = load_model("MY_MODEL.hdf5")
print "loaded"

def capture(
    resolution = (3280, 2464),
    exposure_mode = 'auto',
    delay = 0.2,
    size = (156, 156),
    gray = True,
    format = 'jpeg'
    ):
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode
        time.sleep(delay)
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

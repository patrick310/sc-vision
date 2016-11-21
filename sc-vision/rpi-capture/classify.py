import picamera
import numpy as np
from PIL import Image, ImageOps
import io

def capture():
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = (3280, 2464)
        cam.exposure_mode = 'auto'
        cam.capture(stream, 'jpeg')
        cam.close()
    stream.seek(0)
    original = Image.open(stream)
    image = ImageOps.grayscale(image)
    image.resize((156, 156), Image.LANCZOS)
    arr = np.array(image)
    return original, arr
